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

*** v4 FIX -- diagnosed from bench_gating_sweep.py's actual output, not a
guess. The full sweep OVERTURNS the earlier "fixed overhead" hint (that
hint was based on a single before/after comparison, before the systematic
sweep existed): ratio gets WORSE as n_tokens grows (1.11x -> 4.69x) and
much worse as d_model grows (1.54x -> 9.31x, with the fused kernel's own
time growing 13.4x for a 128x increase in d_model against the reference's
2.2x) -- both are the signature of a genuine compute/memory scaling
problem, not per-call launch overhead, which would show the opposite
(shrinking) trend as work per call increases.

Root cause, found by re-reading the inner loop with that in mind:

    for (uint d = lane; d < D; d += 32u) {
        partial += row[d] * gate_weight[d * E + e];
    }

`row[d]` is coalesced -- for a fixed instant, the 32 lanes read
`row[lane_offset+0..31]`, addresses 1 apart. `gate_weight[d*E+e]` is NOT:
gate_weight is stored [D, E] row-major, so for a FIXED e (what one SIMD
group is doing) and d varying across the 32 lanes, addresses are `d*E+e`
-- E floats apart from each other, not 1. Fully strided, non-coalesced
memory access, on the one buffer whose access cost scales with both
d_model AND num_experts -- exactly the two sweeps that got dramatically
worse.

Fix: pre-transpose gate_weight to [E, D] (expert-major) once, so a SIMD
group working on expert e reads `gate_weight_T[e*D + lane_offset..+31]` --
contiguous, coalesced, same pattern as `row`. Since gate_weight is a model
weight reused across every gating call (and across every warmup+timed
iteration in the benchmark harness), the transposed copy is CACHED by
id(gate_weight) rather than rebuilt every call -- this matches how the
weight is actually used in real inference (loaded once, read many times),
not a benchmark-gaming trick.

HONESTY NOTE, unverified: mx.fast.metal_kernel hands the kernel a raw
buffer pointer, so a lazy MLX `.transpose()` (which may just swap strides
without physically reordering memory) would make the kernel read the
WRONG values here, not just be slow -- the kernel has no idea about
strides, only raw offsets. To avoid depending on undocumented MLX
materialization behavior, the transpose is forced through an explicit
numpy round-trip (mx.array -> np.ascontiguousarray(.T) -> mx.array),
which guarantees a real, contiguous [E, D] layout. This costs one extra
host<->device round trip the first time a given weight array is seen;
please confirm np.array(mlx_array) round-trips correctly on your installed
MLX version -- if it errors, paste the exact error and we'll adjust
(likely alternative: mx.eval() the transposed array first, then check
whether that alone produces a correct raw layout, though I'd want to
verify that empirically rather than assume it before trusting it in a
raw-pointer kernel).

Cache caveat, worth knowing about deliberately: keyed by id(gate_weight)
and holds a reference to the original array to prevent id reuse after
garbage collection. This is fine for this dev/benchmark context (a
handful of distinct weight arrays across a session) but is an unbounded
cache -- a long-running server juggling many distinct gating weights
would want an LRU or weak-reference-keyed version instead. Flagging this
now rather than letting it become a silent surprise later.

*** v5 -- lighter tuning pass, per explicit decision to try this before a
full simdgroup_matrix rewrite. Motivated by real Instruments evidence
(Metal System Trace, metal-gpu-intervals table), not a guess: with v4's
memory-coalescing fix in place, CPU encode time is now a wash between
reference and fused (~72us median, both), and CPU->GPU latency is only
~1.3x worse for fused -- but raw GPU busy duration is 4.2x worse
(515us vs 123us median at the primary shape). That's a genuine
compute-throughput gap, not overhead, and the likely full explanation is
that MLX's reference matmul dispatches onto Apple Silicon's dedicated
matrix-multiply hardware while this kernel's inner loop is a scalar
per-lane multiply-accumulate -- a fundamentally different throughput
ceiling that no amount of dispatch-shape tuning fully closes.

Given that, this pass targets ONLY the piece dispatch-shape tuning CAN
legitimately affect: SIMDGROUPS_PER_TG was a fixed 8 regardless of
num_experts, forcing needless serial waves over experts whenever
num_experts > 8 -- 2 waves for our primary shape (E=16), 8 waves for
E=64. Made adaptive: simdgroups_per_tg = min(num_experts, 32), capped at
32 because 32 simdgroups * 32 lanes = 1024 threads/threadgroup, which is
Apple GPU hardware's per-threadgroup thread limit -- not an arbitrary
number. E=16 now needs exactly 1 wave instead of 2; E=64 needs 2 instead
of 8.

HONEST EXPECTATION: this does NOT touch the per-expert d_model reduction
(each SIMD group still does D/32 sequential scalar FMAs per lane, e.g. 32
iterations at D=1024) -- which the GPU-interval evidence suggests is the
larger piece of the 515us at our primary shape (E=16 only needed 2 waves
even before this change, so halving that to 1 wave removes at most a
modest slice of the total, not the dominant one). This change should help
more clearly at large num_experts (E=64: 8->2 waves, a bigger relative
cut) than at E=16. Re-running the full sweep (not just the primary shape)
after this change will show honestly whether that expectation holds.

Deliberately NOT done this pass: float4-vectorized loads in the inner
loop. The clean way to do that requires reinterpret-casting `row` /
`gate_weight` to a float4 pointer, but their actual Metal address space
(`device` vs `constant`) is decided by MLX per-buffer-size -- exactly the
ambiguity that caused the v1->v2 `auto` fix. An explicit `device float4*`
cast could silently misbehave for small test fixtures MLX places in
`constant` space. Deferred rather than risking a repeat of that bug class
in a "lighter pass" meant to be low-risk; worth revisiting carefully (e.g.
by confirming MLX's actual size threshold for constant-vs-device
placement) if more speed is still needed after this change.

*** v5.1 -- real measurement on your Mac showed the v5 cap of 32 was too
aggressive: E=16 improved as expected (1.99x -> 1.69x), but E=32 and E=64
got WORSE (1.90x -> 2.13x, 2.48x -> 2.92x) versus the pre-v5 fixed-8
baseline. Likely explanation: a 1024-thread threadgroup (32 simdgroups) is
Apple GPU hardware's per-threadgroup max, which probably allows only one
such threadgroup resident per GPU core at a time -- trading away serial
expert-wave savings for worse occupancy across the 256 independent tokens'
worth of work. Lowered MAX_SIMDGROUPS_PER_TG to 16 (512 threads) to keep
the E<=16 win (identical behavior for E<=16, since min(E,16)==min(E,32) in
that range) while pulling E=32/64 back from the worst-occupancy regime.

*** v6 -- REVERTED v5/v5.1 back to a fixed SIMDGROUPS_PER_TG=8, per an
explicit decision after real measurement showed the adaptive scheme
plateaued within noise. Two concrete things drove this:
  1. E=16's dispatch shape was IDENTICAL between v5 (cap 32) and v5.1
     (cap 16) -- min(16,32)==min(16,16)==16 -- yet measured ratio moved
     from 1.69x to 1.80x with zero code change. That's a direct
     measurement of this benchmark's noise floor: roughly 10%+.
  2. Against that noise floor, v5.1's E=32 result (2.19x) was not
     distinguishable from v5's (2.13x) as either a recovery or a further
     regression relative to the pre-v5 baseline (1.90x) -- three fixed-8
     vs. adaptive comparisons at the primary shape all landed in the same
     ~1.7-2.1x band across versions.
Conclusion: threadgroup-size tuning was a real, well-motivated hypothesis
(and the Instruments evidence-based reasoning for trying it was sound),
but it did not produce a reproducible win at the shape that matters, and
it added real complexity (a compiled-kernel-per-variant cache, an extra
tunable cap). Reverting removes that complexity in exchange for
performance that is statistically indistinguishable from keeping it --
not a regression, a wash, and a wash isn't worth the extra moving parts.
The v4 fix (expert-major memory layout, the actual large win -- see v4
postmortem above) is UNCHANGED and fully retained by this revert; only
the v5/v5.1 adaptive-threadgroup experiment is undone.
"""

import mlx.core as mx
import numpy as np

MAX_EXPERTS = 256  # compile-time bound on the threadgroup-memory logits[] array
MAX_TOPK = 64
SIMDGROUPS_PER_TG = 8  # v6: reverted to fixed, per v6 postmortem above --
                       # 8 simdgroups * 32 lanes = 256 threads/threadgroup,
                       # regardless of num_experts.

_KERNEL_SOURCE = r"""
    uint token_id = threadgroup_position_in_grid.x;
    uint local_idx = thread_position_in_threadgroup.x;
    uint lane = local_idx % 32u;          // this thread's lane within its 32-wide SIMD group
    uint sg = local_idx / 32u;            // which SIMD group (0..SIMDGROUPS_PER_TG-1) this thread belongs to

    threadgroup float logits[256];

    uint D = d_model_[0];
    uint E = num_experts[0];
    uint K = top_k[0];

    auto row = hidden + token_id * D;  // `auto`, not `const device float*` -- MLX
                                        // places tiny buffers in the `constant`
                                        // address space instead of `device` as an
                                        // optimization, and hardcoding `device`
                                        // only matched the larger-buffer case.

    // Each SIMD group owns one expert at a time, looping if E > number of
    // resident SIMD groups. Different SIMD groups write different logits[e]
    // slots, so no synchronization is needed between loop iterations -- only
    // once, after the whole loop, before any thread reads the full array.
    for (uint e = sg; e < E; e += SIMDGROUPS_PER_TG_CONST) {
        float partial = 0.0f;
        // gate_weight here is EXPERT-MAJOR ([E, D], pre-transposed host-side --
        // see _get_transposed_weight below). For fixed e, varying d across the
        // 32 lanes gives addresses e*D + lane_offset..+31 -- 1 apart, coalesced.
        // The original [D, E] column access (d*E+e) was E floats apart per lane,
        // fully strided -- root cause of the scaling blowup seen in the sweep.
        for (uint d = lane; d < D; d += 32u) {
            partial += row[d] * gate_weight[e * D + d];
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
# v6: reverted to substituting SIMDGROUPS_PER_TG once at module load time
# (like v2-v4), since there's only one fixed value again -- no per-variant
# kernel cache needed.
_KERNEL_SOURCE = _KERNEL_SOURCE.replace("SIMDGROUPS_PER_TG_CONST", str(SIMDGROUPS_PER_TG))

_kernel = None  # v6: single compiled kernel again, no per-variant cache needed

# Cache of pre-transposed ([E, D], expert-major) copies of gate_weight arrays,
# keyed by id(gate_weight). We keep a reference to the original array alongside
# the transposed copy specifically to prevent Python reusing a freed id for a
# *different* array while the cache entry is still around -- without that, a
# coincidental id collision could silently serve the wrong transposed weight.
_weight_cache = {}

# Cache of the three tiny scalar shape-arrays, keyed by (d_model, num_experts, k)
# -- these are plain ints, not mutable arrays, so no id-collision risk here
# (unlike _weight_cache, which caches by identity of a mutable-looking array).
_scalar_cache = {}


def _get_transposed_weight(gate_weight: mx.array) -> mx.array:
    key = id(gate_weight)
    cached = _weight_cache.get(key)
    if cached is not None and cached[0] is gate_weight:
        return cached[1]

    # Forced through numpy rather than mx.transpose(): mx.fast.metal_kernel
    # hands the kernel a raw buffer pointer with no stride information, so a
    # lazy MLX transpose (which may just swap strides instead of physically
    # reordering memory) would make the kernel read WRONG values here, not
    # just be slow. np.ascontiguousarray forces a real, dense [E, D] copy.
    w_np = np.array(gate_weight, dtype=np.float32)      # [D, E]
    w_t_np = np.ascontiguousarray(w_np.T)                # [E, D], contiguous
    w_t = mx.array(w_t_np)

    _weight_cache[key] = (gate_weight, w_t)
    return w_t


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

    # v4: expert-major [E, D] layout so the kernel's per-lane reads are
    # coalesced (see the v4 postmortem above and the comment in
    # _KERNEL_SOURCE). Cached by id(gate_weight) so this transpose is paid
    # once per distinct weight array, not once per call.
    gate_weight_t = _get_transposed_weight(gate_weight)

    # Scalars passed as 1-element int32 arrays since Metal kernel buffers
    # need a concrete element type -- avoids relying on a specific
    # scalar-passing convention in the metal_kernel API that I can't verify.
    # These only depend on (d_model, num_experts, k), which are static for a
    # given deployment shape, so cache them the same way as the transposed
    # weight instead of reallocating three tiny mx.arrays every single call.
    scalar_key = (d_model, num_experts, k)
    scalar_arrs = _scalar_cache.get(scalar_key)
    if scalar_arrs is None:
        scalar_arrs = (
            mx.array([d_model], dtype=mx.uint32),
            mx.array([num_experts], dtype=mx.uint32),
            mx.array([k], dtype=mx.uint32),
        )
        _scalar_cache[scalar_key] = scalar_arrs
    d_model_arr, num_experts_arr, k_arr = scalar_arrs

    outputs = kernel(
        inputs=[hidden.astype(mx.float32), gate_weight_t,
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