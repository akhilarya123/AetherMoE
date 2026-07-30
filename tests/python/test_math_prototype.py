"""
tests/python/test_math_prototype.py

Wraps prototype/math_proto.py's checks as real pytest cases. Requires only
NumPy -- no MLX, no PyTorch, no Apple Silicon. This is the one test file in
Milestone 2 that I could actually execute myself, and did (see chat history);
it exists in the suite so it keeps running as a fast, dependency-free sanity
gate on every future change, independent of whether MLX/Metal are available
in whatever environment runs `pytest`.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "prototype"))
from math_proto import (                                    # noqa: E402
    reference_gating, fused_kernel_simulation,
    quantize_int8, dequantize_int8, quantized_matmul_perentry,
)


def test_softmax_normalizes():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(5, 16)).astype(np.float32)
    gate_w = rng.normal(size=(16, 8)).astype(np.float32) * 0.1
    gate_b = rng.normal(size=(8,)).astype(np.float32) * 0.01
    _, val = reference_gating(hidden, gate_w, gate_b, k=3)
    assert np.allclose(val.sum(axis=-1), 1.0, atol=1e-6)


def test_fused_simulation_matches_reference():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(5, 16)).astype(np.float32)
    gate_w = rng.normal(size=(16, 8)).astype(np.float32) * 0.1
    gate_b = rng.normal(size=(8,)).astype(np.float32) * 0.01
    idx_ref, val_ref = reference_gating(hidden, gate_w, gate_b, k=3)
    idx_fused, val_fused = fused_kernel_simulation(hidden, gate_w, gate_b, k=3)
    assert np.array_equal(idx_ref, idx_fused)
    assert np.allclose(val_ref, val_fused, atol=1e-6)


def test_tie_breaking_selects_correct_set():
    D, E = 8, 8
    hidden = np.zeros((1, D), dtype=np.float32)
    gate_w = np.zeros((D, E), dtype=np.float32)
    gate_b = np.zeros((E,), dtype=np.float32)
    gate_b[2] = 5.0
    gate_b[5] = 5.0
    idx, _ = reference_gating(hidden, gate_w, gate_b, k=2)
    assert set(idx[0].tolist()) == {2, 5}


def test_quantize_dequantize_round_trip_bounded():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(64, 8)).astype(np.float32)
    q, scale = quantize_int8(w, axis=0)
    w_deq = dequantize_int8(q, scale)
    max_err = np.abs(w - w_deq).max()
    bound = (scale / 2).max()
    assert max_err <= bound + 1e-6


def test_perentry_dequant_matches_dequant_then_matmul():
    rng = np.random.default_rng(2)
    D, E = 32, 8
    w = rng.normal(size=(D, E)).astype(np.float32)
    q, scale = quantize_int8(w, axis=0)
    w_deq = dequantize_int8(q, scale)
    x = rng.normal(size=(4, D)).astype(np.float32)
    out_perentry = quantized_matmul_perentry(x, q, scale)
    out_dequant_first = x.astype(np.float64) @ w_deq.astype(np.float64)
    assert np.allclose(out_perentry, out_dequant_first, atol=1e-6)
