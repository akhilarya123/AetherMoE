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

*** v2 -- diagnosed from bench_quantized_gemm_sweep.py's actual output, same
discipline as the gating kernel's investigation (don't trust a single
before/after comparison -- two initial data points here looked roughly FLAT
across a 32x change in total work, which is exactly the misleading pattern
that almost led to the wrong conclusion for gating; the full 3-axis sweep
told a different, clearer story).

Sweep results: d_model is the dominant, clean, monotonic signal -- ratio
goes from 1.01x (fused matches naive) at D=64 to 3.07x at D=4096, then
plateaus ~3x by D=8192. n_tokens and num_experts sweeps were flat/noisy,
no clear trend -- the original "redundant re-reads from no tiling"
hypothesis did NOT hold up against the data.

Root cause, re-reading the v1 kernel with that in mind: it has ZERO
parallelism across the D-reduction. One thread computes an entire output
element alone, via `for (uint d = 0; d < D; ++d)`, with no help from other
threads. That's a more basic gap than the gating kernel ever had -- gating
at least split its D-reduction across a 32-lane SIMD group from the start
(before v4's memory-layout fix). A single thread doing D sequential
dequantize+FMA steps with no intra-element parallelism is exactly what
should scale linearly and badly with D, and plateau once both sides become
compute-bound -- consistent with the sweep.

Fix: one SIMD group (32 lanes) per output element instead of one thread,
splitting the D-loop across lanes with simd_sum -- direct port of the
pattern already proven in fused_gating_metal.py. q_weight is pre-transposed
to [E, D] (expert-major) up front, same as the gating v4 fix -- the naive
[D, E] column access here would hit the identical strided-memory bug gating
had before that fix, so this avoids rediscovering it from scratch.
SIMDGROUPS_PER_TG is a fixed 8 (not adaptive) -- deliberately not repeating
the threadgroup-size tuning exploration from the gating kernel, which
plateaued within measurement noise there; starting from the simple,
already-informed choice instead.
"""

import mlx.core as mx
import numpy as np

SIMDGROUPS_PER_TG = 8  # 8 simdgroups * 32 lanes = 256 threads/threadgroup,
                       # fixed -- see v2 postmortem above for why this is
                       # deliberately not adaptive.


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
    // v2: one SIMD group per output element (row, col), D-reduction split
    // across the 32 lanes, reduced with simd_sum -- direct port of the
    // proven fused_gating_metal.py pattern. One threadgroup per token
    // (row); SIMDGROUPS_PER_TG_CONST simdgroups each take a col, looping
    // if E > SIMDGROUPS_PER_TG_CONST.
    uint token_id = threadgroup_position_in_grid.x;
    uint local_idx = thread_position_in_threadgroup.x;
    uint sg = local_idx / 32u;
    uint lane = local_idx % 32u;

    uint N = n_[0];
    uint D = d_[0];
    uint E = e_[0];

    if (token_id >= N) return;

    auto x_row = x + token_id * D;  // `auto`, not a hardcoded address space --
                                     // see the identical fix + explanation in
                                     // fused_gating_metal.py's kernel source

    for (uint col = sg; col < E; col += SIMDGROUPS_PER_TG_CONST) {
        float s = scale[col];
        float partial = 0.0f;
        // q_weight here is EXPERT-MAJOR ([E, D], pre-transposed host-side --
        // see _get_transposed_qweight below). For fixed col, varying d across
        // the 32 lanes gives addresses col*D + lane_offset..+31 -- 1 apart,
        // coalesced. The original [D, E] column access (d*E+col) would have
        // been E int8s apart per lane -- the same strided-access bug the
        // gating kernel had before its v4 fix.
        for (uint d = lane; d < D; d += 32u) {
            float w = float(q_weight[col * D + d]) * s;   // dequant AT MULTIPLY TIME
            partial += x_row[d] * w;
        }
        float total = simd_sum(partial);
        if (lane == 0u) {
            out[token_id * E + col] = total;
        }
    }
"""

_kernel = None

# Cache of pre-transposed ([E, D], expert-major) INT8 copies of q_weight
# arrays, keyed by id(q_weight) -- same pattern and same rationale as
# fused_gating_metal.py's _weight_cache: the transpose is paid once per
# distinct weight array (a real model weight, reused across many calls in
# practice), not once per call. Keeps a reference to the original array to
# prevent id reuse after garbage collection serving a stale cache entry.
_qweight_cache = {}


def _get_transposed_qweight(q_int8: mx.array) -> mx.array:
    key = id(q_int8)
    cached = _qweight_cache.get(key)
    if cached is not None and cached[0] is q_int8:
        return cached[1]

    # Forced through numpy, same reasoning as the gating kernel's v4 fix:
    # mx.fast.metal_kernel hands the kernel a raw buffer pointer with no
    # stride information, so a lazy MLX transpose (which may just swap
    # strides instead of physically reordering memory) would make the
    # kernel read WRONG values here, not just be slow.
    q_np = np.array(q_int8, dtype=np.int8)          # [D, E]
    q_t_np = np.ascontiguousarray(q_np.T)             # [E, D], contiguous
    q_t = mx.array(q_t_np)

    _qweight_cache[key] = (q_int8, q_t)
    return q_t


def _get_kernel():
    global _kernel
    if _kernel is None:
        source = _KERNEL_SOURCE.replace("SIMDGROUPS_PER_TG_CONST", str(SIMDGROUPS_PER_TG))
        _kernel = mx.fast.metal_kernel(
            name="fused_quantized_matmul",
            input_names=["x", "q_weight", "scale", "n_", "d_", "e_"],
            output_names=["out"],
            source=source,
            header="#include <metal_stdlib>\nusing namespace metal;\n",
        )
    return _kernel


def fused_quantized_matmul(x_fp32: mx.array, q_int8: mx.array,
                            scale: mx.array) -> mx.array:
    """
    Args:
        x_fp32:  [N, D] activations
        q_int8:  [D, E] quantized weights (transposed internally to [E, D]
                 and cached -- see _get_transposed_qweight)
        scale:   [1, E] per-column dequant scale

    Returns:
        out: [N, E]

    v2: one SIMD group per output element, D-reduction parallelized across
    32 lanes with simd_sum, expert-major weight layout for coalesced
    access. See the v2 postmortem in this file's module docstring for the
    sweep evidence and reasoning that motivated this rewrite -- the v1
    one-thread-per-element design had zero parallelism across D at all.
    """
    n_tokens, d_model = x_fp32.shape
    d_model2, num_experts = q_int8.shape
    if d_model != d_model2:
        raise ValueError(f"shape mismatch: x has D={d_model}, q_weight has D={d_model2}")

    kernel = _get_kernel()
    q_weight_t = _get_transposed_qweight(q_int8)

    n_arr = mx.array([n_tokens], dtype=mx.uint32)
    d_arr = mx.array([d_model], dtype=mx.uint32)
    e_arr = mx.array([num_experts], dtype=mx.uint32)

    # grid is total threads (n_tokens * threadgroup size), NOT number of
    # threadgroups -- mx.fast.metal_kernel's grid follows Metal's
    # dispatchThreads semantics. This is the exact bug (#1 in HANDOVER.md)
    # already found and fixed once in fused_gating_metal.py; applying that
    # lesson here up front rather than rediscovering it.
    outputs = kernel(
        inputs=[x_fp32.astype(mx.float32), q_weight_t, scale.astype(mx.float32),
                n_arr, d_arr, e_arr],
        grid=(n_tokens * SIMDGROUPS_PER_TG * 32, 1, 1),
        threadgroup=(SIMDGROUPS_PER_TG * 32, 1, 1),
        output_shapes=[(n_tokens, num_experts)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]