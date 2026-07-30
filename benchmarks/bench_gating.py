"""
benchmarks/bench_gating.py

Performance regression check: fused gating kernel vs the unfused MLX
reference, per the Milestone 2 testing plan ("automated benchmark comparing
fused vs unfused kernel latency on every change, failing the build if a
regression exceeds a set threshold").

MLX is LAZY -- ops build a computation graph and don't execute until a value
is materialized (mx.eval, or converting to numpy/Python). Every timing loop
below calls mx.eval explicitly and does untimed warmup iterations first, or
the "benchmark" would mostly measure graph-building overhead and/or
first-call JIT/compile cost rather than steady-state kernel latency.

Usage:
    python benchmarks/bench_gating.py
    python benchmarks/bench_gating.py --n-tokens 512 --d-model 4096 --num-experts 64 --k 8
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "kernels"))

import mlx.core as mx  # noqa: E402
import numpy as np      # noqa: E402

from reference_gating import reference_gating   # noqa: E402
from fused_gating_metal import fused_gating       # noqa: E402


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
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--regression-threshold", type=float, default=1.5,
                         help="Fail if fused is more than this factor SLOWER "
                              "than reference (default 1.5x). Fused should "
                              "normally be faster or comparable -- this "
                              "threshold exists to catch an accidental "
                              "performance regression, not to require a "
                              "specific speedup.")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    hidden = mx.array(rng.normal(size=(args.n_tokens, args.d_model)).astype(np.float32))
    gate_w = mx.array((rng.normal(size=(args.d_model, args.num_experts)) * 0.1).astype(np.float32))
    gate_b = mx.array((rng.normal(size=(args.num_experts,)) * 0.01).astype(np.float32))

    print(f"shape: n_tokens={args.n_tokens} d_model={args.d_model} "
          f"num_experts={args.num_experts} k={args.k}")

    ref_time = timeit(lambda: reference_gating(hidden, gate_w, gate_b, args.k))
    fused_time = timeit(lambda: fused_gating(hidden, gate_w, gate_b, args.k))

    print(f"reference (unfused): {ref_time * 1e6:.1f} us/call")
    print(f"fused kernel:         {fused_time * 1e6:.1f} us/call")
    print(f"speedup:              {ref_time / fused_time:.2f}x")

    if fused_time > ref_time * args.regression_threshold:
        print(f"\n[REGRESSION] fused kernel is {fused_time / ref_time:.2f}x "
              f"SLOWER than the unfused reference, exceeding the "
              f"{args.regression_threshold}x threshold.")
        sys.exit(1)
    print("\n[OK] no performance regression detected")


if __name__ == "__main__":
    main()
