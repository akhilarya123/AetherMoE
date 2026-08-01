"""
benchmarks/bench_quantized_gemm.py

Performance regression check: fused INT8-dequant-at-multiply-time kernel vs
the naive dequant-then-matmul path, per the Milestone 2 testing plan. Mirrors
bench_gating.py's structure exactly for consistency.

Per quantized_gemm.py's own docstring, this kernel launches one thread per
output element with no threadgroup-memory tiling -- so unlike the gating
kernel's original bug (a genuine memory-access-pattern mistake), there's no
known bug going in here. This benchmark exists to find out, with real
numbers, whether the lack of tiling actually costs as much as the docstring
warns it might -- not to confirm a suspicion.

MLX is LAZY -- every timing loop below does untimed warmup iterations and
calls mx.eval explicitly, or the "benchmark" would mostly measure graph-
building overhead rather than steady-state kernel latency.

Usage:
    python3.11 benchmarks/bench_quantized_gemm.py
    python3.11 benchmarks/bench_quantized_gemm.py --n-tokens 512 --d-model 4096 --num-experts 64
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tokens", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--regression-threshold", type=float, default=1.5,
                         help="Fail if fused is more than this factor SLOWER "
                              "than the naive dequant-then-matmul path "
                              "(default 1.5x).")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    x = mx.array(rng.normal(size=(args.n_tokens, args.d_model)).astype(np.float32))
    w = mx.array(rng.normal(size=(args.d_model, args.num_experts)).astype(np.float32))
    q, scale = quantize_weights(w)
    mx.eval(q, scale)  # materialize once, outside the timing loop

    print(f"shape: n_tokens={args.n_tokens} d_model={args.d_model} "
          f"num_experts={args.num_experts}")

    naive_time = timeit(lambda: naive_quantized_matmul(x, q, scale))
    fused_time = timeit(lambda: fused_quantized_matmul(x, q, scale))

    print(f"naive (dequant-then-matmul): {naive_time * 1e6:.1f} us/call")
    print(f"fused (dequant-at-multiply): {fused_time * 1e6:.1f} us/call")
    print(f"speedup:                     {naive_time / fused_time:.2f}x")

    if fused_time > naive_time * args.regression_threshold:
        print(f"\n[REGRESSION] fused kernel is {fused_time / naive_time:.2f}x "
              f"SLOWER than the naive dequant-then-matmul path, exceeding "
              f"the {args.regression_threshold}x threshold.")
        sys.exit(1)
    print("\n[OK] no performance regression detected")


if __name__ == "__main__":
    main()