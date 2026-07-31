"""
benchmarks/bench_gating_sweep.py

Cheap diagnostic to run BEFORE opening Instruments: sweeps n_tokens and
d_model independently to tell apart two very different explanations for
the fused kernel's ~2.4-3x slowdown:

  (a) FIXED PER-CALL OVERHEAD (kernel launch, the fixed 256-thread dispatch,
      the single-thread top-k/softmax tail) -- if this dominates, the
      slowdown ratio should SHRINK as n_tokens/d_model grow, because the
      fixed cost gets amortized over more actual work.
  (b) GENUINE COMPUTE SCALING problem -- if this dominates, the slowdown
      ratio should stay roughly CONSTANT (or worsen) as shapes grow.

Prior data point already suggests (a): at n_tokens=256/d_model=1024 the
fused kernel was ~2.4-3x slower, but at n_tokens=512/d_model=4096 (8x more
total work) it was only ~1.4-1.9x slower -- ratio improving with size is
the signature of fixed overhead, not a scaling problem. This script checks
that more systematically across a wider sweep before we trust that read.

Usage:
    python3.11 benchmarks/bench_gating_sweep.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "kernels"))

import mlx.core as mx
import numpy as np

from reference_gating import reference_gating
from fused_gating_metal import fused_gating


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - start) / iters


def bench_one(n_tokens, d_model, num_experts, k):
    rng = np.random.default_rng(0)
    hidden = mx.array(rng.normal(size=(n_tokens, d_model)).astype(np.float32))
    gate_w = mx.array((rng.normal(size=(d_model, num_experts)) * 0.1).astype(np.float32))
    gate_b = mx.array((rng.normal(size=(num_experts,)) * 0.01).astype(np.float32))

    ref_t = timeit(lambda: reference_gating(hidden, gate_w, gate_b, k))
    fused_t = timeit(lambda: fused_gating(hidden, gate_w, gate_b, k))
    return ref_t, fused_t


def main():
    print(f"{'n_tokens':>9} {'d_model':>8} {'experts':>8} {'k':>3} "
          f"{'ref(us)':>10} {'fused(us)':>10} {'ratio':>8}")
    print("-" * 62)

    # Sweep 1: fix d_model/experts, vary n_tokens -- isolates per-token
    # fixed cost (one threadgroup launched per token).
    for n_tokens in [1, 4, 16, 64, 256, 1024]:
        ref_t, fused_t = bench_one(n_tokens, d_model=1024, num_experts=16, k=4)
        print(f"{n_tokens:>9} {1024:>8} {16:>8} {4:>3} "
              f"{ref_t*1e6:>10.1f} {fused_t*1e6:>10.1f} {fused_t/ref_t:>7.2f}x")

    print()

    # Sweep 2: fix n_tokens, vary d_model -- isolates per-element compute
    # scaling (the actual dot-product work per expert).
    for d_model in [64, 256, 1024, 4096, 8192]:
        ref_t, fused_t = bench_one(n_tokens=256, d_model=d_model, num_experts=16, k=4)
        print(f"{256:>9} {d_model:>8} {16:>8} {4:>3} "
              f"{ref_t*1e6:>10.1f} {fused_t*1e6:>10.1f} {fused_t/ref_t:>7.2f}x")

    print()

    # Sweep 3: fix n_tokens/d_model, vary num_experts -- isolates the cost
    # of the SIMDGROUPS_PER_TG=8 fixed dispatch vs actual expert count
    # (e.g. num_experts=1 still launches 8 simdgroups, 7 of which do
    # nothing -- if that waste matters, small num_experts should look
    # disproportionately bad).
    for num_experts in [1, 2, 4, 8, 16, 32, 64]:
        k = min(4, num_experts)
        ref_t, fused_t = bench_one(n_tokens=256, d_model=1024, num_experts=num_experts, k=k)
        print(f"{256:>9} {1024:>8} {num_experts:>8} {k:>3} "
              f"{ref_t*1e6:>10.1f} {fused_t*1e6:>10.1f} {fused_t/ref_t:>7.2f}x")

    print("\nWhat to look for:")
    print("  Sweep 1 (n_tokens up):  ratio SHRINKING as n_tokens grows -> fixed")
    print("    per-launch overhead confirmed (amortizes over more threadgroups).")
    print("  Sweep 2 (d_model up):   ratio SHRINKING as d_model grows -> the")
    print("    projection's compute cost is fine; overhead is elsewhere.")
    print("  Sweep 3 (num_experts):  ratio WORSE at small num_experts -> the")
    print("    fixed SIMDGROUPS_PER_TG=8 dispatch is wasting threads when there")
    print("    are fewer than 8 experts to compute.")


if __name__ == "__main__":
    main()
