"""
Pure-NumPy prototype of the Milestone 2 math, run and checked here (no MLX/
Metal available in this sandbox) so the ALGORITHM is proven correct before
translating it into MLX kernel code that I cannot execute myself.
"""
import numpy as np

def reference_gating(hidden, gate_weight, gate_bias, k):
    """hidden: [N, D], gate_weight: [D, E], gate_bias: [E] -> (idx[N,k] int, val[N,k] float)"""
    logits = hidden @ gate_weight + gate_bias           # [N, E]  <- "linear projection"
    order = np.argsort(-logits, axis=-1)                # descending
    topk_idx = order[:, :k]                             # [N, k]  <- "top-k selection"
    topk_logits = np.take_along_axis(logits, topk_idx, axis=-1)
    m = topk_logits.max(axis=-1, keepdims=True)
    e = np.exp(topk_logits - m)
    topk_val = e / e.sum(axis=-1, keepdims=True)        # <- "row-wise softmax over the k selected"
    return topk_idx, topk_val

def fused_kernel_simulation(hidden, gate_weight, gate_bias, k):
    """Simulates exactly what the per-threadgroup Metal kernel does: for each
    token row, serially compute all E logits, then iterative max-extraction
    top-k (mirrors the kernel's destructive-selection loop), then softmax.
    This should match reference_gating element-for-element EXCEPT in the
    presence of exact ties, where argsort's tie-break order and iterative
    max-extraction's tie-break order could differ -- that's an important
    edge case to test explicitly, not just assume away.
    """
    N, D = hidden.shape
    E = gate_weight.shape[1]
    out_idx = np.zeros((N, k), dtype=np.int64)
    out_val = np.zeros((N, k), dtype=np.float64)
    for n in range(N):
        logits = np.array([sum(hidden[n, d] * gate_weight[d, e] for d in range(D)) + gate_bias[e]
                            for e in range(E)])
        local = logits.copy()
        topk_vals = []
        topk_idx = []
        for kk in range(k):
            best_idx = int(np.argmax(local))   # np.argmax ties -> lowest index, same as the
                                                 # kernel's "> best_val" (strict) linear scan
            topk_vals.append(local[best_idx])
            topk_idx.append(best_idx)
            local[best_idx] = -np.inf
        topk_vals = np.array(topk_vals)
        m = topk_vals.max()
        e = np.exp(topk_vals - m)
        val = e / e.sum()
        out_idx[n] = topk_idx
        out_val[n] = val
    return out_idx, out_val

def quantize_int8(weight_fp32, axis=0):
    """Per-column (per-output-feature) symmetric int8 quantization.
    scale[e] = max(|weight[:,e]|) / 127
    q[d,e] = round(weight[d,e] / scale[e]), clipped to int8 range.
    Dequantize-at-multiply-time means: during the GEMM inner loop we compute
    q[d,e] (int8) * scale[e] on the fly rather than materializing a full
    fp32/fp16 copy of the weight matrix up front -- that's the whole point
    (memory bandwidth savings), so the *numerical* contract we must prove
    correct is just: dequant(quantize(w)) ~= w within quantization error,
    and matmul using per-multiply dequant == matmul using precomputed dequant.
    """
    scale = np.abs(weight_fp32).max(axis=axis, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-12)  # avoid div-by-zero for an all-zero column
    q = np.round(weight_fp32 / scale).clip(-127, 127).astype(np.int8)
    return q, scale

def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale

def quantized_matmul_perentry(x_fp32, q_weight_int8, scale):
    """Simulates the fused kernel's inner loop: dequantize each int8 weight
    element to fp32 (kernel would use fp16) at the moment it's used in the
    multiply-accumulate, rather than dequantizing the whole matrix first."""
    N, D = x_fp32.shape
    D2, E = q_weight_int8.shape
    assert D == D2
    out = np.zeros((N, E), dtype=np.float64)
    for n in range(N):
        for e in range(E):
            acc = 0.0
            for d in range(D):
                w = q_weight_int8[d, e] * scale[0, e]   # dequant at multiply time
                acc += x_fp32[n, d] * w
            out[n, e] = acc
    return out


def run_checks():
    rng = np.random.default_rng(0)

    # --- 1. Reference gating math is internally consistent (softmax sums to 1, etc.) ---
    N, D, E, k = 5, 16, 8, 3
    hidden = rng.normal(size=(N, D)).astype(np.float32)
    gate_w = rng.normal(size=(D, E)).astype(np.float32) * 0.1
    gate_b = rng.normal(size=(E,)).astype(np.float32) * 0.01

    idx_ref, val_ref = reference_gating(hidden, gate_w, gate_b, k)
    assert np.allclose(val_ref.sum(axis=-1), 1.0, atol=1e-6), "softmax weights must sum to 1"
    assert idx_ref.shape == (N, k) and val_ref.shape == (N, k)
    print("[OK] reference_gating: softmax normalizes to 1, shapes correct")

    # --- 2. Fused-kernel simulation matches reference bit-for-bit on random inputs ---
    idx_fused, val_fused = fused_kernel_simulation(hidden, gate_w, gate_b, k)
    assert np.array_equal(idx_ref, idx_fused), f"index mismatch:\n{idx_ref}\nvs\n{idx_fused}"
    assert np.allclose(val_ref, val_fused, atol=1e-6), "value mismatch on random (no-tie) inputs"
    print("[OK] fused_kernel_simulation matches reference_gating exactly on random inputs")

    # --- 3. Tie-breaking edge case: two experts with IDENTICAL logits ---
    hidden_tie = np.zeros((1, D), dtype=np.float32)
    gate_w_tie = np.zeros((D, E), dtype=np.float32)
    gate_b_tie = np.zeros((E,), dtype=np.float32)
    gate_b_tie[2] = 5.0
    gate_b_tie[5] = 5.0  # experts 2 and 5 tied for the top logit
    idx_ref_t, val_ref_t = reference_gating(hidden_tie, gate_w_tie, gate_b_tie, k=2)
    idx_fused_t, val_fused_t = fused_kernel_simulation(hidden_tie, gate_w_tie, gate_b_tie, k=2)
    print(f"[TIE CHECK] reference picks {idx_ref_t}, fused-sim picks {idx_fused_t} "
          f"(both should pick experts {{2,5}} as the top-2 set)")
    assert set(idx_ref_t[0].tolist()) == {2, 5}
    assert set(idx_fused_t[0].tolist()) == {2, 5}
    print("[OK] both implementations select the correct SET on ties, "
          "even if internal order differs -- this is the actual correctness "
          "contract for MoE routing (order among tied experts doesn't matter, "
          "membership does)")

    # --- 4. k == num_experts edge case (every expert selected) ---
    idx_all, val_all = reference_gating(hidden, gate_w, gate_b, k=E)
    assert set(idx_all[0].tolist()) == set(range(E))
    assert np.allclose(val_all.sum(axis=-1), 1.0, atol=1e-6)
    print("[OK] k == num_experts: full softmax over all experts, still normalizes")

    # --- 5. k == 1 edge case ---
    idx_1, val_1 = reference_gating(hidden, gate_w, gate_b, k=1)
    assert np.allclose(val_1, 1.0), "softmax over a single logit must be exactly 1.0"
    print("[OK] k == 1: softmax of a single value is exactly 1.0")

    # --- 6. INT8 quantization round-trip error is bounded and small ---
    D2, E2 = 64, 8
    w = rng.normal(size=(D2, E2)).astype(np.float32)
    q, scale = quantize_int8(w, axis=0)
    w_deq = dequantize_int8(q, scale)
    max_abs_err = np.abs(w - w_deq).max()
    max_possible_err = (scale / 2).max()  # rounding to nearest int -> error <= half a quantization step
    assert max_abs_err <= max_possible_err + 1e-6, (max_abs_err, max_possible_err)
    print(f"[OK] int8 quantize/dequantize round-trip: max abs error {max_abs_err:.6f} "
          f"<= theoretical bound {max_possible_err:.6f}")

    # --- 7. Per-multiply dequant gives the SAME result as dequant-then-matmul ---
    x = rng.normal(size=(4, D2)).astype(np.float32)
    out_perentry = quantized_matmul_perentry(x, q, scale)
    out_dequant_first = x.astype(np.float64) @ w_deq.astype(np.float64)
    assert np.allclose(out_perentry, out_dequant_first, atol=1e-6), \
        "dequant-at-multiply-time must be numerically identical to dequant-then-matmul"
    print("[OK] quantized_matmul_perentry (kernel's approach) == dequantize-then-matmul "
          "(reference approach) -- confirms fusing dequant into the multiply loop "
          "changes nothing numerically, only when/where the dequant happens")

    # --- 8. Quantized matmul vs the ORIGINAL fp32 weights: bounded relative error ---
    out_fp32 = x.astype(np.float64) @ w.astype(np.float64)
    rel_err = np.abs(out_perentry - out_fp32) / (np.abs(out_fp32) + 1e-8)
    print(f"[INFO] quantized vs fp32 relative error: mean={rel_err.mean():.4f}, "
          f"max={rel_err.max():.4f} (expected to be small but nonzero -- this IS the "
          f"quantization tradeoff, not a bug)")

    print("\nAll math checks passed.")


if __name__ == "__main__":
    run_checks()
