"""
benchmarks/bench_quantized_gemm_sweep.py

Diagnostic sweep for fused_quantized_matmul vs naive_quantized_matmul,
mirroring bench_gating_sweep.py's methodology exactly: sweep n_tokens,
d_model, and num_experts independently to distinguish fixed-overhead from
genuine scaling problems, rather than trusting a single before/after
comparison (that's exactly the mistake the gating kernel's investigation
almost made -- an early partial comparison suggested "fixed overhead", and
only the full sweep revealed the real, opposite story).

Given quantized_gemm.py's kernel does one thread per (row, col) output
element with NO threadgroup-memory tiling (self-flagged in its own
docstring), the working hypothesis is that it re-reads each x row and each
weight column redundantly once per output element it contributes to --
total global memory traffic ~O(N*D*E) instead of a tiled matmul's
~O(N*D + D*E). If that's right, the ratio should get WORSE specifically as
num_experts and n_tokens grow (more redundant re-reads of the same rows/
columns), not just as d_model grows. This sweep is how we find out whether
that's actually true, rather than assuming it.

Usage:
    python3.11 benchmarks/bench_quantized_gemm_sweep.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "kernels"))

import mlx.core as mx  # noqa: E402
import numpy as np      # noqa: E402

from quantized_gemm import (   # noqa: E402
    quantize_weights, naive_quantized_matmul, fused_quantized_matmul,
)


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    elapsed = time.perf_counter() - start
    return elapsed / iters


def bench_one(rng, n_tokens, d_model, num_experts):
    x = mx.array(rng.normal(size=(n_tokens, d_model)).astype(np.float32))
    w = mx.array(rng.normal(size=(d_model, num_experts)).astype(np.float32))
    q, scale = quantize_weights(w)
    mx.eval(q, scale)

    naive_time = timeit(lambda: naive_quantized_matmul(x, q, scale))
    fused_time = timeit(lambda: fused_quantized_matmul(x, q, scale))
    return naive_time, fused_time


def print_row(n_tokens, d_model, num_experts, naive_t, fused_t):
    ratio = fused_t / naive_t
    print(f"{n_tokens:>9} {d_model:>8} {num_experts:>8}   "
          f"{naive_t*1e6:>9.1f} {fused_t*1e6:>10.1f}   {ratio:>6.2f}x")


def main():
    rng = np.random.default_rng(0)

    print(f"{'n_tokens':>9} {'d_model':>8} {'experts':>8}   "
          f"{'naive(us)':>9} {'fused(us)':>10}   {'ratio':>6}")
    print("-" * 64)

    # Sweep 1: n_tokens up, d_model/num_experts fixed at primary shape.
    for n_tokens in [1, 4, 16, 64, 256, 1024]:
        naive_t, fused_t = bench_one(rng, n_tokens, 1024, 16)
        print_row(n_tokens, 1024, 16, naive_t, fused_t)
    print()

    # Sweep 2: d_model up, n_tokens/num_experts fixed at primary shape.
    for d_model in [64, 256, 1024, 4096, 8192]:
        naive_t, fused_t = bench_one(rng, 256, d_model, 16)
        print_row(256, d_model, 16, naive_t, fused_t)
    print()

    # Sweep 3: num_experts up, n_tokens/d_model fixed at primary shape.
    for num_experts in [1, 2, 4, 8, 16, 32, 64]:
        naive_t, fused_t = bench_one(rng, 256, 1024, num_experts)
        print_row(256, 1024, num_experts, naive_t, fused_t)
    print()

    print("What to look for:")
    print("  Sweep 1 (n_tokens up):  ratio getting WORSE -> more redundant")
    print("    re-reads of each weight column (one per token), consistent")
    print("    with the no-tiling hypothesis.")
    print("  Sweep 2 (d_model up):   ratio getting worse -> per-element")
    print("    compute/memory cost scales badly (same axis that mattered")
    print("    most for the gating kernel's coalescing bug -- worth ruling")
    print("    out a similar access-pattern issue here too, even though a")
    print("    first read of the code didn't find one).")
    print("  Sweep 3 (num_experts up): ratio getting WORSE -> more redundant")
    print("    re-reads of each token's x row (one per expert output),")
    print("    the other half of the no-tiling hypothesis.")


if __name__ == "__main__":
    main()