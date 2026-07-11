"""
Reproduction & fix-testing harness for flashinfer-ai/flashinfer#3904.

Capturing run() of a wrapper constructed with use_cuda_graph=False (the
default) into a CUDA graph silently produces wrong output after the next
plan() call, because default-mode plan() re-allocates its buffers while the
captured graph keeps reading the old addresses.

Usage:

  python repro-3904.py                # or: python repro-3904.py repro
      On a fix branch: raises RuntimeError at capture time (the guard).
      On main / older flashinfer: demonstrates the silent corruption.

  FLASHINFER_ALLOW_UNSAFE_GRAPH_CAPTURE=1 python repro-3904.py
      On a fix branch: bypasses the guard and demonstrates the same
      silent corruption the guard exists to prevent.

  python repro-3904.py overhead
      Benchmarks the guard's cost on the eager decode hot path. Works on
      any commit (main / any fix variant) -- check out two commits, run
      this on each, compare the numbers.

Reference numbers (RTX 4070, batch-1 fp16 decode):
  corruption:            replayed-vs-correct max diff ~2.26 (0 = correct)
  eager run(), no guard: ~10.4 us      raw capture query: ~0.21 us
"""

import statistics
import sys
import time

import torch

import flashinfer

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 32, 8, 128, 16


def make_wrapper():
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    # NOTE: use_cuda_graph defaults to False -- that is the whole bug.
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")


def plan(wrapper, kv_len):
    num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    wrapper.plan(
        torch.tensor([0, num_pages], dtype=torch.int32, device="cuda"),
        torch.arange(num_pages, dtype=torch.int32, device="cuda"),
        torch.tensor([(kv_len - 1) % PAGE_SIZE + 1], dtype=torch.int32, device="cuda"),
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=torch.float16,
    )


def make_inputs():
    kv_cache = torch.randn(
        128, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda"
    )
    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    return q, kv_cache


# --------------------------------------------------------------------- repro
def mode_repro():
    torch.manual_seed(0)
    wrapper = make_wrapper()
    q, kv_cache = make_inputs()

    # 1. plan for a short sequence, warm up eagerly
    plan(wrapper, kv_len=37)
    for _ in range(3):
        wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    # 2. capture run() -- freezes the CURRENT plan buffer addresses into the graph
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            out_captured = wrapper.run(q, kv_cache)
    except RuntimeError as e:
        print("guard raised at capture time (the fix is active):\n")
        print(f"  {e}\n")
        print("re-run with FLASHINFER_ALLOW_UNSAFE_GRAPH_CAPTURE=1 to see the")
        print("silent corruption this prevents")
        return

    # 3. sequences grew; the serving engine replans.
    #    Default mode: plan() re-allocates -> the graph's baked-in pointers
    #    now reference freed memory.
    plan(wrapper, kv_len=2048)
    _ = torch.randn(100_000, device="cuda")  # unrelated allocs recycle freed memory

    # 4. correct answer for the new plan, computed eagerly
    out_eager = wrapper.run(q, kv_cache)

    # 5. replay the captured graph -- reads stale plan buffers, no error raised
    graph.replay()
    torch.cuda.synchronize()

    diff = (out_captured - out_eager).abs().max().item()
    print(f"replayed-vs-correct max diff: {diff:.4f}   (0 would mean correct)")
    if diff > 1e-2:
        print("SILENT CORRUPTION REPRODUCED: the replayed graph returned garbage")
        print("attention output with no error anywhere -- this presents as a")
        print("model-quality bug (e.g. near-zero speculative-decoding acceptance).")
    else:
        print("outputs happened to match (allocator reuse is nondeterministic; ")
        print("try increasing the second plan's kv_len)")


# ------------------------------------------------------------------ overhead
def mode_overhead(iters=2000, repeats=5):
    torch.manual_seed(0)
    wrapper = make_wrapper()
    q, kv_cache = make_inputs()
    plan(wrapper, kv_len=37)

    def bench_run():
        for _ in range(200):
            wrapper.run(q, kv_cache)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            wrapper.run(q, kv_cache)
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1e6

    times = sorted(bench_run() for _ in range(repeats))
    med = statistics.median(times)
    print(f"eager run() per call:  median {med:.2f} us   "
          f"(min {times[0]:.2f}, max {times[-1]:.2f}, {repeats}x{iters} iters)")

    # raw cost of the underlying driver query (guard's irreducible part)
    if hasattr(torch.cuda, "is_current_stream_capturing"):
        n = 100_000
        t0 = time.time()
        for _ in range(n):
            torch.cuda.is_current_stream_capturing()
        raw = (time.time() - t0) / n * 1e6
        print(f"raw capture query:     {raw:.3f} us/call")

    # best-effort microbench of the guard expression as shipped (name varies
    # across fix iterations; silently skipped on commits without a guard)
    import flashinfer.utils as fu

    check = getattr(fu, "is_current_stream_capturing", None)
    if check is not None:
        n = 100_000
        t0 = time.time()
        for _ in range(n):
            if not wrapper._use_cuda_graph and check():
                pass
        guard = (time.time() - t0) / n * 1e6
        print(f"guard expression:      {guard:.3f} us/call")
    else:
        print("guard expression:      (no guard helper on this commit)")

    print("\ncompare 'eager run()' medians across commits; the guard's true")
    print("cost is that delta. During graph replay the cost is exactly zero")
    print("(host code does not execute on replay).")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "repro"
    if mode == "repro":
        mode_repro()
    elif mode == "overhead":
        mode_overhead()
    else:
        sys.exit(f"unknown mode {mode!r}; use: repro | overhead")
