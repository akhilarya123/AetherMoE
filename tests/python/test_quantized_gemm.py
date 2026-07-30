"""
tests/python/test_quantized_gemm.py

Correctness tests for the INT8 quantized expert GEMM. Tolerance here is
looser than the gating tests' 1e-4 -- quantization is LOSSY by design, so
the contract we're testing is "the fused kernel's per-multiply dequant
matches the naive dequant-then-matmul path" (should be near machine
precision, since both do the same arithmetic) and separately "quantized
output stays within a sane bound of the true fp32 output" (necessarily
loose, since 8-bit quantization is the whole tradeoff).

Run with: pytest tests/python/test_quantized_gemm.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "kernels"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "validation"))

mx = pytest.importorskip("mlx.core", reason="MLX not installed -- requires Apple Silicon")
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from quantized_gemm import (                              # noqa: E402
    quantize_weights, naive_quantized_matmul, fused_quantized_matmul,
)
from pytorch_reference import pytorch_quantized_matmul     # noqa: E402

FUSED_VS_NAIVE_TOLERANCE = 1e-3   # both do the identical arithmetic; should be near machine precision
FP32_RELATIVE_TOLERANCE = 0.35    # quantization IS lossy -- this bounds "not egregiously wrong", not "exact"


def _random_case(rng, n_tokens, d_model, num_experts):
    x = rng.normal(size=(n_tokens, d_model)).astype(np.float32)
    w = rng.normal(size=(d_model, num_experts)).astype(np.float32)
    return x, w


@pytest.mark.parametrize("n_tokens,d_model,num_experts", [
    (4, 32, 8),
    (16, 64, 16),
    (1, 16, 4),
])
def test_fused_matches_naive_dequant_then_matmul(n_tokens, d_model, num_experts):
    rng = np.random.default_rng(11)
    x, w = _random_case(rng, n_tokens, d_model, num_experts)

    q, scale = quantize_weights(mx.array(w))
    naive_out = np.array(naive_quantized_matmul(mx.array(x), q, scale))
    fused_out = np.array(fused_quantized_matmul(mx.array(x), q, scale))

    assert np.allclose(naive_out, fused_out, atol=FUSED_VS_NAIVE_TOLERANCE), (
        f"fused kernel diverges from naive dequant-then-matmul beyond "
        f"{FUSED_VS_NAIVE_TOLERANCE} -- these two should compute the exact "
        f"same arithmetic just at different times, so any gap here points "
        f"to a bug in the fused kernel's inner loop, not quantization loss.\n"
        f"max abs diff: {np.abs(naive_out - fused_out).max()}")


def test_fused_matches_pytorch_reference():
    rng = np.random.default_rng(12)
    x, w = _random_case(rng, 8, 32, 8)

    q, scale = quantize_weights(mx.array(w))
    fused_out = np.array(fused_quantized_matmul(mx.array(x), q, scale))
    pt_out = pytorch_quantized_matmul(x, np.array(q), np.array(scale))

    assert np.allclose(fused_out, pt_out, atol=FUSED_VS_NAIVE_TOLERANCE), (
        f"fused MLX kernel vs PyTorch mps reference mismatch beyond "
        f"{FUSED_VS_NAIVE_TOLERANCE}. max abs diff: "
        f"{np.abs(fused_out - pt_out).max()}")


def test_quantized_output_within_bound_of_true_fp32():
    """Not a tight-tolerance test -- confirms quantization error is bounded
    and doesn't blow up, without pretending 8-bit quantization is lossless."""
    rng = np.random.default_rng(13)
    x, w = _random_case(rng, 8, 64, 16)

    q, scale = quantize_weights(mx.array(w))
    fused_out = np.array(fused_quantized_matmul(mx.array(x), q, scale))
    true_out = x @ w  # fp32 ground truth, no quantization at all

    rel_err = np.abs(fused_out - true_out) / (np.abs(true_out) + 1e-6)
    assert rel_err.mean() < FP32_RELATIVE_TOLERANCE, (
        f"mean relative error {rel_err.mean():.4f} exceeds sanity bound "
        f"{FP32_RELATIVE_TOLERANCE} -- either the quantization scheme has a "
        f"bug, or this scale of weights genuinely needs a different scheme "
        f"(e.g. per-row instead of per-column, or fewer than 8 bits' worth "
        f"of dynamic range being wasted).")


def test_all_zero_weight_column_does_not_produce_nan():
    """Edge case: a weight column of all zeros -> scale would be 0 without
    the epsilon floor in quantize_weights, causing a divide-by-zero NaN."""
    d_model, num_experts = 16, 4
    w = np.zeros((d_model, num_experts), dtype=np.float32)
    w[:, 1] = np.random.default_rng(14).normal(size=d_model)  # only column 1 is nonzero

    q, scale = quantize_weights(mx.array(w))
    x = np.random.default_rng(15).normal(size=(4, d_model)).astype(np.float32)
    out = np.array(fused_quantized_matmul(mx.array(x), q, scale))

    assert not np.isnan(out).any(), "all-zero weight column produced NaN output"
    assert np.allclose(out[:, 0], 0.0, atol=1e-5), "all-zero column should produce all-zero output"
