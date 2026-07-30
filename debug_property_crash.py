"""
debug_property_crash.py

Bisects the "Fatal Python error: Aborted" seen under
tests/python/test_gating_property.py. A hard process abort is a native
crash, not a Python exception -- Hypothesis can't catch it, shrink it, or
tell us which generated example triggered it; the process just dies. This
script replaces Hypothesis with a plain, deterministic sweep over small
shape combinations, printing (and flushing) each one *before* calling the
kernel. Whichever line is the LAST one printed (with no "OK" after it) is
the shape that crashes -- that's the information we actually need.

Run with:
    python3.11 debug_property_crash.py

If/when it crashes, just send me the full output up to and including the
crash -- the last "trying ..." line without a following "OK" line pinpoints
the exact (n_tokens, d_model, num_experts, k) that triggers it.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "kernels"))

import numpy as np
import mlx.core as mx
from fused_gating_metal import fused_gating

# Deliberately covers the same ranges as the Hypothesis strategy
# (gating_inputs: num_experts 1-32, k 1-num_experts, n_tokens 1-64,
# d_model 1-64), but as an exhaustive-ish small grid instead of random
# sampling, and in a deterministic, reproducible order.
NUM_EXPERTS_VALUES = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32]
N_TOKENS_VALUES = [1, 2, 3, 7, 8, 63, 64]
D_MODEL_VALUES = [1, 2, 3, 31, 32, 33, 63, 64]

def k_values_for(num_experts):
    # 1, num_experts, and a middle value -- covers the extremes, which is
    # where bugs like this usually live.
    vals = {1, num_experts}
    if num_experts > 2:
        vals.add(num_experts // 2)
    return sorted(vals)


def main():
    rng = np.random.default_rng(0)
    total = 0
    for num_experts in NUM_EXPERTS_VALUES:
        for k in k_values_for(num_experts):
            for n_tokens in N_TOKENS_VALUES:
                for d_model in D_MODEL_VALUES:
                    total += 1
                    print(f"trying n_tokens={n_tokens} d_model={d_model} "
                          f"num_experts={num_experts} k={k} ... ", end="", flush=True)

                    hidden = rng.normal(size=(n_tokens, d_model)).astype(np.float32)
                    gate_w = (rng.normal(size=(d_model, num_experts)) * 0.1).astype(np.float32)
                    gate_b = (rng.normal(size=(num_experts,)) * 0.01).astype(np.float32)

                    idx, val = fused_gating(mx.array(hidden), mx.array(gate_w),
                                             mx.array(gate_b), k)
                    idx_np, val_np = np.array(idx), np.array(val)  # forces evaluation

                    assert idx_np.shape == (n_tokens, k)
                    assert not np.isnan(val_np).any()

                    print("OK")
    print(f"\nAll {total} combinations completed without crashing.")


if __name__ == "__main__":
    main()
