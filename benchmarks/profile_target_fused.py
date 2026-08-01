"""
benchmarks/profile_target_fused.py

Instruments profiling target -- FUSED gating kernel only. Mirrors
profile_target_reference.py exactly (same shape, same iteration count,
same warmup pattern) so the two traces are directly comparable -- only the
implementation under the loop differs.

Usage:
    python3.11 benchmarks/profile_target_fused.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "kernels"))

import mlx.core as mx
import numpy as np

from fused_gating_metal import fused_gating

N_TOKENS, D_MODEL, NUM_EXPERTS, K = 256, 1024, 16, 4
ITERS = 20000


def main():
    rng = np.random.default_rng(0)
    hidden = mx.array(rng.normal(size=(N_TOKENS, D_MODEL)).astype(np.float32))
    gate_w = mx.array((rng.normal(size=(D_MODEL, NUM_EXPERTS)) * 0.1).astype(np.float32))
    gate_b = mx.array((rng.normal(size=(NUM_EXPERTS,)) * 0.01).astype(np.float32))

    print(f"[fused] shape n_tokens={N_TOKENS} d_model={D_MODEL} "
          f"num_experts={NUM_EXPERTS} k={K} -- running {ITERS} iterations. "
          f"Attach Instruments now.")

    for _ in range(10):
        mx.eval(fused_gating(hidden, gate_w, gate_b, K))

    start = time.perf_counter()
    for i in range(ITERS):
        mx.eval(fused_gating(hidden, gate_w, gate_b, K))
        if i % 2000 == 0:
            elapsed = time.perf_counter() - start
            print(f"  iter {i}/{ITERS}  ({elapsed:.1f}s elapsed)")

    print("[fused] done.")


if __name__ == "__main__":
    main()
