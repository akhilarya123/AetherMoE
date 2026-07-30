"""
tests/python/test_gating_numerical.py

Three-way cross-check: fused Metal kernel vs MLX reference vs PyTorch mps
reference. Tolerance is 1e-4 per the Milestone 2 spec, applied to the
softmax VALUES; index agreement is checked as exact SET equality (not
positional equality) because tie-breaking order between argsort (MLX
reference) and torch.topk (PyTorch reference) and the kernel's linear-scan
max-extraction can legitimately differ on exact ties without either being
wrong -- see prototype/math_proto.py's tie-break check for why set equality
is the correct contract here, not tuple equality.

Run with: pytest tests/python/test_gating_numerical.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "kernels"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "validation"))

mx = pytest.importorskip("mlx.core", reason="MLX not installed -- this suite requires Apple Silicon")
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from reference_gating import reference_gating          # noqa: E402
from fused_gating_metal import fused_gating             # noqa: E402
from pytorch_reference import pytorch_gating            # noqa: E402

TOLERANCE = 1e-4


def _random_inputs(rng, n_tokens, d_model, num_experts):
    hidden = rng.normal(size=(n_tokens, d_model)).astype(np.float32)
    gate_w = (rng.normal(size=(d_model, num_experts)) * 0.1).astype(np.float32)
    gate_b = (rng.normal(size=(num_experts,)) * 0.01).astype(np.float32)
    return hidden, gate_w, gate_b


def _sets_equal_per_row(idx_a, idx_b):
    return all(set(a.tolist()) == set(b.tolist()) for a, b in zip(idx_a, idx_b))


@pytest.mark.parametrize("n_tokens,d_model,num_experts,k", [
    (8, 32, 8, 2),
    (16, 64, 16, 4),
    (1, 16, 8, 1),
    (32, 128, 32, 8),
])
def test_fused_matches_mlx_reference(n_tokens, d_model, num_experts, k):
    rng = np.random.default_rng(42)
    hidden, gate_w, gate_b = _random_inputs(rng, n_tokens, d_model, num_experts)

    ref_idx, ref_val = reference_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k)
    fused_idx, fused_val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k)

    ref_idx_np, ref_val_np = np.array(ref_idx), np.array(ref_val)
    fused_idx_np, fused_val_np = np.array(fused_idx), np.array(fused_val)

    assert _sets_equal_per_row(ref_idx_np, fused_idx_np), (
        f"expert SET mismatch:\nreference={ref_idx_np}\nfused={fused_idx_np}")

    # Values aren't directly comparable position-by-position unless index
    # order also matches, so sort both by value descending before comparing.
    ref_sorted = np.sort(ref_val_np, axis=-1)[:, ::-1]
    fused_sorted = np.sort(fused_val_np, axis=-1)[:, ::-1]
    assert np.allclose(ref_sorted, fused_sorted, atol=TOLERANCE), (
        f"softmax value mismatch beyond tolerance {TOLERANCE}:\n"
        f"reference={ref_sorted}\nfused={fused_sorted}")


@pytest.mark.parametrize("n_tokens,d_model,num_experts,k", [
    (8, 32, 8, 2),
    (16, 64, 16, 4),
])
def test_fused_matches_pytorch_reference(n_tokens, d_model, num_experts, k):
    rng = np.random.default_rng(7)
    hidden, gate_w, gate_b = _random_inputs(rng, n_tokens, d_model, num_experts)

    fused_idx, fused_val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k)
    pt_idx, pt_val = pytorch_gating(hidden, gate_w, gate_b, k)

    fused_idx_np, fused_val_np = np.array(fused_idx), np.array(fused_val)

    assert _sets_equal_per_row(fused_idx_np, pt_idx), (
        f"expert SET mismatch vs PyTorch:\nfused={fused_idx_np}\npytorch={pt_idx}")

    fused_sorted = np.sort(fused_val_np, axis=-1)[:, ::-1]
    pt_sorted = np.sort(pt_val, axis=-1)[:, ::-1]
    assert np.allclose(fused_sorted, pt_sorted, atol=TOLERANCE), (
        f"softmax value mismatch vs PyTorch beyond tolerance {TOLERANCE}:\n"
        f"fused={fused_sorted}\npytorch={pt_sorted}")


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(1)
    hidden, gate_w, gate_b = _random_inputs(rng, 10, 32, 8)
    _, val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k=3)
    val_np = np.array(val)
    assert np.allclose(val_np.sum(axis=-1), 1.0, atol=1e-5)
