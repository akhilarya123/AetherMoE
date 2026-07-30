"""
tests/python/test_gating_property.py

Property-based tests per the Milestone 2 testing plan: "random top-k values,
batch sizes, and expert counts to catch edge cases (ties in top-k, empty
experts, extreme batch sizes)."

Properties checked (things that must ALWAYS hold, for any valid input,
regardless of the specific numbers Hypothesis generates):
  1. Output shapes are always [N, k].
  2. Softmax weights always sum to 1 per row.
  3. Every returned expert index is in range [0, num_experts).
  4. No expert index repeats within a single token's top-k (can't route a
     token to the same expert twice).
  5. Fused kernel and MLX reference always agree on the SET of selected
     experts (see test_gating_numerical.py for why set, not tuple).

Run with: pytest tests/python/test_gating_property.py -v
(add --hypothesis-seed=<N> to reproduce a specific failure)
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "kernels"))

mx = pytest.importorskip("mlx.core", reason="MLX not installed -- this suite requires Apple Silicon")
hypothesis = pytest.importorskip("hypothesis", reason="pip install hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from reference_gating import reference_gating   # noqa: E402
from fused_gating_metal import fused_gating      # noqa: E402


@st.composite
def gating_inputs(draw, min_experts=1, max_experts=32):
    num_experts = draw(st.integers(min_value=min_experts, max_value=max_experts))
    k = draw(st.integers(min_value=1, max_value=num_experts))  # k can't exceed num_experts
    n_tokens = draw(st.integers(min_value=1, max_value=64))
    d_model = draw(st.integers(min_value=1, max_value=64))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(n_tokens, d_model)).astype(np.float32)
    gate_w = (rng.normal(size=(d_model, num_experts)) * 0.1).astype(np.float32)
    gate_b = (rng.normal(size=(num_experts,)) * 0.01).astype(np.float32)
    return hidden, gate_w, gate_b, k, num_experts, n_tokens


@given(gating_inputs())
@settings(max_examples=100, deadline=None)
def test_fused_gating_invariants(inputs):
    hidden, gate_w, gate_b, k, num_experts, n_tokens = inputs

    idx, val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k)
    idx_np, val_np = np.array(idx), np.array(val)

    # 1. Shape
    assert idx_np.shape == (n_tokens, k)
    assert val_np.shape == (n_tokens, k)

    # 2. Softmax normalization
    assert np.allclose(val_np.sum(axis=-1), 1.0, atol=1e-4), \
        f"rows don't sum to 1: {val_np.sum(axis=-1)}"

    # 3. Index range
    assert (idx_np >= 0).all() and (idx_np < num_experts).all(), \
        f"index out of range [0,{num_experts}): {idx_np}"

    # 4. No duplicate expert within a row
    for row in idx_np:
        assert len(set(row.tolist())) == len(row), f"duplicate expert in row: {row}"

    # 5. Values are valid probabilities
    assert (val_np >= 0).all() and (val_np <= 1.0 + 1e-6).all()


@given(gating_inputs())
@settings(max_examples=50, deadline=None)
def test_fused_matches_reference_expert_set(inputs):
    hidden, gate_w, gate_b, k, num_experts, n_tokens = inputs

    ref_idx, _ = reference_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k)
    fused_idx, _ = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k)

    ref_np, fused_np = np.array(ref_idx), np.array(fused_idx)
    for r, f in zip(ref_np, fused_np):
        assert set(r.tolist()) == set(f.tolist()), (
            f"expert set mismatch: reference picked {sorted(r.tolist())}, "
            f"fused picked {sorted(f.tolist())}")


# --- Explicit edge cases beyond what random generation reliably hits ---

def test_k_equals_num_experts_selects_everyone():
    """k == num_experts: every expert must be selected (softmax over all)."""
    rng = np.random.default_rng(3)
    num_experts = 8
    hidden = rng.normal(size=(4, 16)).astype(np.float32)
    gate_w = rng.normal(size=(16, num_experts)).astype(np.float32)
    gate_b = rng.normal(size=(num_experts,)).astype(np.float32)

    idx, val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k=num_experts)
    idx_np, val_np = np.array(idx), np.array(val)
    for row in idx_np:
        assert set(row.tolist()) == set(range(num_experts))
    assert np.allclose(val_np.sum(axis=-1), 1.0, atol=1e-5)


def test_single_expert_no_choice():
    """num_experts == 1, k == 1: only one possible choice, weight must be 1.0."""
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(5, 16)).astype(np.float32)
    gate_w = rng.normal(size=(16, 1)).astype(np.float32)
    gate_b = rng.normal(size=(1,)).astype(np.float32)

    idx, val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k=1)
    idx_np, val_np = np.array(idx), np.array(val)
    assert (idx_np == 0).all()
    assert np.allclose(val_np, 1.0, atol=1e-6)


def test_exact_tie_between_experts_selects_correct_set():
    """Two experts with an EXACTLY identical logit -- selection must include
    both (set membership), even though which one is "first" is arbitrary."""
    d_model = 8
    num_experts = 6
    hidden = np.zeros((1, d_model), dtype=np.float32)
    gate_w = np.zeros((d_model, num_experts), dtype=np.float32)
    gate_b = np.zeros((num_experts,), dtype=np.float32)
    gate_b[1] = 3.0
    gate_b[4] = 3.0  # exact tie for the top logit

    idx, _ = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k=2)
    idx_np = np.array(idx)
    assert set(idx_np[0].tolist()) == {1, 4}


def test_extreme_batch_size_one_token():
    hidden = np.random.default_rng(5).normal(size=(1, 32)).astype(np.float32)
    gate_w = np.random.default_rng(6).normal(size=(32, 8)).astype(np.float32)
    gate_b = np.zeros((8,), dtype=np.float32)
    idx, val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k=2)
    assert np.array(idx).shape == (1, 2)
    assert np.allclose(np.array(val).sum(axis=-1), 1.0, atol=1e-5)


def test_large_batch_size():
    rng = np.random.default_rng(8)
    n_tokens = 2048
    hidden = rng.normal(size=(n_tokens, 64)).astype(np.float32)
    gate_w = rng.normal(size=(64, 16)).astype(np.float32) * 0.1
    gate_b = rng.normal(size=(16,)).astype(np.float32) * 0.01
    idx, val = fused_gating(mx.array(hidden), mx.array(gate_w), mx.array(gate_b), k=4)
    assert np.array(idx).shape == (n_tokens, 4)
    assert np.allclose(np.array(val).sum(axis=-1), 1.0, atol=1e-4)
