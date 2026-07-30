"""
AetherMoE — src/kernels/fused_gating_metal.py

Tier B: fused gating as ONE Metal kernel launch (linear projection -> top-k
selection -> row-wise softmax), instead of the several separate MLX ops
in reference_gating.py.

*** KERNEL v2 -- see v1 postmortem below before touching the parallelism
strategy again ***

v1 parallelized across EXPERTS ONLY: one thread per expert, each thread
doing the entire d_model-length dot product as a single serial loop. Real
hardware measurement (not a guess): 2-2.7x SLOWER than MLX's built-in
reference implementation. The reason is straightforward once you look at
it -- for d_model=4096, that's 4096 sequential scalar multiply-adds on ONE
thread with zero vectorization, while MLX's matmul uses Apple's tuned GEMM
path. num_experts (8-64) was never the expensive dimension; d_model
(hundreds to thousands) is, and v1 didn't parallelize it at all.

v2 fixes this: each expert's dot product is computed by a 32-lane SIMD
group (not one thread), with the d_model reduction split across those 32
lanes and combined via `simd_sum` (a hardware reduction intrinsic, not a
manual tree-reduce -- one instruction). Multiple SIMD groups run per
threadgroup, each independently owning one expert; if num_experts exceeds
the number of resident SIMD groups, the kernel loops internally over
expert batches. This is still ONE kernel launch (fusion intact) -- the
looping happens inside the kernel body, not as repeated host-side launches.

*** HONESTY NOTE -- unverified again, for a different reason than v1 ***
v1's *launch mechanics* (mx.fast.metal_kernel's Python call signature) are
now confirmed working against your installed MLX version -- that risk is
retired. What's unverified in v2 is the Metal Shading Language itself:
`simd_sum` is a real MSL intrinsic, but I derived the lane/simdgroup indices
manually from `thread_position_in_threadgroup` (dividing/modulo by 32)
rather than using MSL's dedicated `[[thread_index_in_simdgroup]]` /
`[[simdgroup_index_in_threadgroup]]` attributes, because I don't know
whether MLX's kernel wrapper auto-injects those two the way it auto-injects
thread_position_in_threadgroup (which we now know it does). The manual
derivation should be equivalent AS LONG AS the threadgroup's linear thread
index maps to SIMD groups the same way I'm assuming (lane = index % 32,
simdgroup = index / 32) -- this is true for Apple GPUs' actual hardware
SIMD width (32), but if you get a compile error on `simd_sum` itself,
that's a missing include -- try adding `header="#include <metal_stdlib>\nusing namespace metal;\n"`
to the `mx.fast.metal_kernel(...)` call, or send me the exact compiler
error and I'll adjust.

*** v3 FIX (found via real testing on your Mac, not guessed) ***
Every v2 numerical test failure had the same signature: token/row 0 came
out correct, every other row was zero or garbage. That's the signature of
only ONE threadgroup ever actually running. Root cause: `grid` in
mx.fast.metal_kernel is TOTAL THREAD COUNT (Metal's dispatchThreads
semantics), not "number of threadgroups" -- I had it backwards. With
threadgroup=(256,1,1) and grid=(n_tokens,1,1) where n_tokens < 256 (true
for every test case run), MLX computed ceil(n_tokens/256) = 1 threadgroup
total, so only "token slot 0" was ever computed. Confirmed by cross-check:
quantized_gemm.py's grid was already total-thread-count by coincidence of
how that kernel happens to be structured, and its tests all passed cleanly
on the first try. Fix: grid must be n_tokens * threadgroup_size so that
ceil(grid / threadgroup) == n_tokens threadgroups actually get dispatched.
This likely also explains the hard process abort seen in the Hypothesis
property tests -- repeatedly asking the driver to run a threadgroup larger
than the requested grid may hit a harder failure mode on some shape/size
combinations, not just silently wrong output. Re-test after this fix to
confirm the abort is gone too; if it persists, that's new information, not
something this fix was expected to cover.

*** STILL OPEN: single-token case (n_tokens=1) picked expert 3 vs
reference's expert 4 -- this could NOT have been caused by the grid bug
above (a single-token dispatch was already exactly 1 threadgroup either
way, before and after the fix). My best guess is floating-point summation
order: the fused kernel reduces d_model via a 32-lane simd_sum, MLX's
reference reduction does its own (different) summation order, and if two
experts' logits happen to be very close for that specific random draw, a
tiny rounding difference could flip which one wins. Please re-run
specifically test_fused_matches_mlx_reference[1-16-8-1] after applying this
fix -- if it still fails, send me the raw logit values (not just the
top-1 pick) so we can tell a genuine bug apart from an honest near-tie.

  - One Metal threadgroup per token row.
  - Threadgroup size is now FIXED at SIMDGROUPS_PER_TG * 32 (256 threads),
    independent of num_experts -- a robustness improvement over v1, where
    threadgroup size scaled with num_experts.
  - Each SIMD group (32 lanes) computes one expert's full d_model dot
    product cooperatively, then loops to the next unhandled expert if
    num_experts > SIMDGROUPS_PER_TG.
  - Same top-k + softmax tail as v1 (unchanged -- that part was never the
    bottleneck; k and num_experts are both tiny, so a single thread doing
    that part costs nothing relative to the projection).
"""

import mlx.core as mx

MAX_EXPERTS = 256  # compile-time bound on the threadgroup-memory logits[] array
MAX_TOPK = 64
SIMDGROUPS_PER_TG = 8  # 8 simdgroups * 32 lanes = 256 threads/threadgroup, fixed regardless of num_experts

_KERNEL_SOURCE = r"""
    uint token_id = threadgroup_position_in_grid.x;
    uint local_idx = thread_position_in_threadgroup.x;
    uint lane = local_idx % 32u;          // this thread's lane within its 32-wide SIMD group
    uint sg = local_idx / 32u;            // which SIMD group (0..SIMDGROUPS_PER_TG-1) this thread belongs to

    threadgroup float logits[256];

    uint D = d_model_[0];
    uint E = num_experts[0];
    uint K = top_k[0];

    const device float* row = hidden + token_id * D;

    // Each SIMD group owns one expert at a time, looping if E > number of
    // resident SIMD groups. Different SIMD groups write different logits[e]
    // slots, so no synchronization is needed between loop iterations -- only
    // once, after the whole loop, before any thread reads the full array.
    for (uint e = sg; e < E; e += SIMDGROUPS_PER_TG_CONST) {
        float partial = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            partial += row[d] * gate_weight[d * E + e];
        }
        float dot = simd_sum(partial);   // 32-lane hardware reduction, single instruction
        if (lane == 0u) {
            logits[e] = dot + gate_bias[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_idx == 0u) {
        float local_logits[256];
        for (uint e = 0u; e < E; ++e) local_logits[e] = logits[e];

        float topk_vals[64];
        int topk_idx[64];
        for (uint kk = 0u; kk < K; ++kk) {
            float best_val = -INFINITY;
            int best_idx = -1;
            for (uint e = 0u; e < E; ++e) {
                if (local_logits[e] > best_val) {
                    best_val = local_logits[e];
                    best_idx = int(e);
                }
            }
            topk_vals[kk] = best_val;
            topk_idx[kk] = best_idx;
            local_logits[best_idx] = -INFINITY;
        }

        float max_val = topk_vals[0];
        float sum_exp = 0.0f;
        float exp_vals[64];
        for (uint kk = 0u; kk < K; ++kk) {
            exp_vals[kk] = exp(topk_vals[kk] - max_val);
            sum_exp += exp_vals[kk];
        }
        for (uint kk = 0u; kk < K; ++kk) {
            out_idx[token_id * K + kk] = topk_idx[kk];
            out_val[token_id * K + kk] = exp_vals[kk] / sum_exp;
        }
    }
"""
# SIMDGROUPS_PER_TG_CONST is textually substituted (not a real Metal constant
# reference) since this source is a plain string, not a templated file.
_KERNEL_SOURCE = _KERNEL_SOURCE.replace("SIMDGROUPS_PER_TG_CONST", str(SIMDGROUPS_PER_TG))

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="fused_gating",
            input_names=["hidden", "gate_weight", "gate_bias", "d_model_",
                         "num_experts", "top_k"],
            output_names=["out_idx", "out_val"],
            source=_KERNEL_SOURCE,
            header="#include <metal_stdlib>\nusing namespace metal;\n",
        )
    return _kernel


def fused_gating(hidden: mx.array, gate_weight: mx.array, gate_bias: mx.array,
                  k: int):
    """
    Args mirror reference_gating.reference_gating exactly, so the two are
    drop-in interchangeable for testing/benchmarking.

    Args:
        hidden:      [N, D] float32
        gate_weight: [D, E] float32
        gate_bias:   [E]    float32
        k:           number of experts per token

    Returns:
        topk_idx: [N, k] int32
        topk_val: [N, k] float32
    """
    n_tokens, d_model = hidden.shape
    num_experts = gate_weight.shape[1]

    if num_experts > MAX_EXPERTS:
        raise ValueError(
            f"num_experts={num_experts} exceeds this kernel's compile-time "
            f"MAX_EXPERTS={MAX_EXPERTS}; raise the threadgroup-memory array "
            f"size in _KERNEL_SOURCE (and MAX_EXPERTS above) if you need more."
        )
    if k > MAX_TOPK or k > num_experts:
        raise ValueError(f"k={k} invalid for num_experts={num_experts} "
                          f"(MAX_TOPK={MAX_TOPK})")

    kernel = _get_kernel()

    # Scalars passed as 1-element int32 arrays since Metal kernel buffers
    # need a concrete element type -- avoids relying on a specific
    # scalar-passing convention in the metal_kernel API that I can't verify.
    d_model_arr = mx.array([d_model], dtype=mx.uint32)
    num_experts_arr = mx.array([num_experts], dtype=mx.uint32)
    k_arr = mx.array([k], dtype=mx.uint32)

    outputs = kernel(
        inputs=[hidden.astype(mx.float32), gate_weight.astype(mx.float32),
                gate_bias.astype(mx.float32), d_model_arr, num_experts_arr, k_arr],
        # `grid` is TOTAL THREAD COUNT (Metal dispatchThreads semantics), not
        # the number of threadgroups -- MLX computes
        # num_threadgroups = ceil(grid / threadgroup) itself. We want exactly
        # n_tokens threadgroups (one per token row), so grid must be
        # n_tokens * threadgroup_size, NOT just n_tokens. Getting this wrong
        # silently launches only 1 threadgroup whenever n_tokens is smaller
        # than the threadgroup size (256) -- which was every test case here
        # -- computing token 0 correctly and leaving every other row
        # unwritten (zero or stale garbage). Confirmed against
        # quantized_gemm.py, whose grid happened to already be a total
        # thread count and whose tests passed cleanly.
        grid=(n_tokens * SIMDGROUPS_PER_TG * 32, 1, 1),
        threadgroup=(SIMDGROUPS_PER_TG * 32, 1, 1),
        output_shapes=[(n_tokens, k), (n_tokens, k)],
        output_dtypes=[mx.int32, mx.float32],
    )
    topk_idx, topk_val = outputs
    return topk_idx, topk_val
