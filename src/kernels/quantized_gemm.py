"""
AetherMoE — src/kernels/quantized_gemm.py

Expert weights stored as INT8, dequantized to FP16 at multiply time inside
the GEMM inner loop, rather than materializing a full-precision copy of the
weight matrix up front. On a memory-constrained laptop this halves-to-quarters
the bytes moved from unified memory per expert weight, which matters more
than the extra ALU work of the unpack (GEMMs at these sizes are memory-bound,
not compute-bound).

Quantization scheme: per-output-column symmetric, proven correct against a
from-scratch NumPy implementation in prototype/math_proto.py before this
file was written:
    scale[e]  = max(|weight[:, e]|) / 127
    q[d, e]   = round(weight[d, e] / scale[e]), clipped to [-127, 127]
    dequant   = q[d, e] * scale[e]

Two tiers, same as the gating kernel:
  - quantize_weights / dequantize_naive / naive_quantized_matmul: plain MLX
    ops, dequantizes the WHOLE weight matrix before matmul. Low API risk,
    used as the correctness oracle.
  - fused_quantized_matmul: custom Metal kernel that dequantizes each weight
    element inline in the multiply-accumulate loop. This is the actual
    Milestone 2 deliverable; same API-risk caveat as fused_gating_metal.py
    applies here (see the HONESTY NOTE there) -- I could not run this.
"""

import mlx.core as mx


def quantize_weights(weight_fp32: mx.array):
    """weight_fp32: [D, E] -> (q_int8: [D, E], scale: [1, E])"""
    scale = mx.abs(weight_fp32).max(axis=0, keepdims=True) / 127.0
    scale = mx.maximum(scale, 1e-12)
    q = mx.round(weight_fp32 / scale)
    q = mx.clip(q, -127, 127).astype(mx.int8)
    return q, scale


def dequantize_naive(q_int8: mx.array, scale: mx.array) -> mx.array:
    """Materializes the full dequantized matrix -- this is the "expensive"
    baseline (reads int8, writes back full fp32/fp16 matrix) that the fused
    kernel's inline dequant is meant to avoid."""
    return q_int8.astype(mx.float32) * scale


def naive_quantized_matmul(x_fp32: mx.array, q_int8: mx.array,
                            scale: mx.array) -> mx.array:
    """Reference path: dequantize-then-matmul. Correct but does not save
    memory bandwidth (the whole point of the quantized kernel) since the
    full-precision weight matrix gets materialized first."""
    w_deq = dequantize_naive(q_int8, scale)
    return x_fp32 @ w_deq


# ---------------------------------------------------------------------------
# Fused kernel: dequantize-at-multiply-time
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = r"""
    uint row = thread_position_in_grid.y;   // output row (token)
    uint col = thread_position_in_grid.x;   // output column (expert output feature)

    uint N = n_[0];
    uint D = d_[0];
    uint E = e_[0];

    if (row >= N || col >= E) return;

    float acc = 0.0f;
    float s = scale[col];
    auto x_row = x + row * D;  // `auto`, not `const device float*` -- see the
                                // identical fix + explanation in
                                // fused_gating_metal.py's kernel source
    for (uint d = 0; d < D; ++d) {
        float w = float(q_weight[d * E + col]) * s;   // dequant AT MULTIPLY TIME
        acc += x_row[d] * w;
    }
    out[row * E + col] = acc;
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="fused_quantized_matmul",
            input_names=["x", "q_weight", "scale", "n_", "d_", "e_"],
            output_names=["out"],
            source=_KERNEL_SOURCE,
        )
    return _kernel


def fused_quantized_matmul(x_fp32: mx.array, q_int8: mx.array,
                            scale: mx.array) -> mx.array:
    """
    Args:
        x_fp32:  [N, D] activations
        q_int8:  [D, E] quantized weights
        scale:   [1, E] per-column dequant scale

    Returns:
        out: [N, E]

    NOTE: this launches one Metal thread per (row, col) output element --
    simple and correct, but NOT tiled/blocked, so it will not be competitive
    with MLX's built-in matmul on large shapes. That's an expected, explicit
    follow-up optimization (threadgroup-memory tiling), not something to
    "fix" as a bug on first pass -- get it numerically correct first, per
    the Milestone 2 Definition of Done, then optimize with Instruments
    evidence in hand.
    """
    n_tokens, d_model = x_fp32.shape
    d_model2, num_experts = q_int8.shape
    if d_model != d_model2:
        raise ValueError(f"shape mismatch: x has D={d_model}, q_weight has D={d_model2}")

    kernel = _get_kernel()

    n_arr = mx.array([n_tokens], dtype=mx.uint32)
    d_arr = mx.array([d_model], dtype=mx.uint32)
    e_arr = mx.array([num_experts], dtype=mx.uint32)

    # grid is (x=columns, y=rows) to match thread_position_in_grid.x/y usage
    # in the kernel source above.
    outputs = kernel(
        inputs=[x_fp32.astype(mx.float32), q_int8, scale.astype(mx.float32),
                n_arr, d_arr, e_arr],
        grid=(num_experts, n_tokens, 1),
        threadgroup=(min(32, num_experts), min(32, n_tokens), 1),
        output_shapes=[(n_tokens, num_experts)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]
