"""
AetherMoE — src/kernels/reference_gating.py

Tier A: the "unfused" reference implementation of MoE gating, built entirely
from stable, well-documented MLX array ops (matmul, argsort, softmax). No
custom kernels here — this is deliberately the boring, low-risk version.

Why this exists even though Milestone 2's actual deliverable is the FUSED
kernel: every performance/correctness claim about the fused kernel is
relative to something. This is that something. It's also what the property
tests below run against a hand-checked oracle.

Math (proven in prototype/math_proto.py against a from-scratch NumPy
reimplementation before this file was written):
    logits   = hidden @ gate_weight + gate_bias        # [N, E]
    topk_idx = indices of the k largest logits per row  # [N, k]
    topk_val = softmax(logits[topk_idx])                # softmax over ONLY
                                                          # the k selected
                                                          # logits, not all E
"""

import mlx.core as mx


def reference_gating(hidden: mx.array, gate_weight: mx.array,
                      gate_bias: mx.array, k: int):
    """
    Args:
        hidden:      [N, D] token hidden states
        gate_weight: [D, E] gating linear-layer weight
        gate_bias:   [E]    gating linear-layer bias
        k:           number of experts to route each token to

    Returns:
        topk_idx: [N, k] int32  -- expert indices, sorted by descending logit
        topk_val: [N, k] float  -- softmax routing weights, sums to 1 per row
    """
    logits = hidden @ gate_weight + gate_bias  # [N, E]  -- "linear projection"

    # Top-k via full descending argsort + slice. Less efficient than a true
    # partial-selection algorithm for large E, but uses only mx.argsort,
    # which is guaranteed to exist and be correct -- exactly the point of
    # this being the "safe" reference tier.
    order = mx.argsort(-logits, axis=-1)  # descending order of logits
    topk_idx = order[:, :k]

    # Gather the logits at topk_idx. mx.take_along_axis exists in modern MLX;
    # if your installed version predates it, the commented fallback below
    # does the same gather using one-hot matmul (slower, but dependency-free
    # on take_along_axis specifically).
    topk_logits = mx.take_along_axis(logits, topk_idx, axis=-1)
    # --- fallback if take_along_axis is unavailable on your MLX version ---
    # one_hot = (mx.arange(logits.shape[-1])[None, None, :] == topk_idx[..., None]).astype(logits.dtype)
    # topk_logits = (one_hot * logits[:, None, :]).sum(axis=-1)

    topk_val = mx.softmax(topk_logits, axis=-1)  # softmax over just the k selected
    return topk_idx.astype(mx.int32), topk_val
