"""
AetherMoE — validation/pytorch_reference.py

Independent ground-truth implementations in PyTorch, run on Apple Silicon's
`mps` backend. The point of this file is INDEPENDENCE: reference_gating.py
and pytorch_gating below should never share a bug, because they're written
against two different frameworks' primitives. Agreement between MLX and
PyTorch is much stronger evidence of correctness than MLX-vs-MLX agreement
would be.

If `mps` isn't available (e.g. running this file on non-Apple hardware, or
an older PyTorch build), falls back to CPU -- still numerically valid as a
reference, just not exercising the GPU path.
"""

import torch


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pytorch_gating(hidden, gate_weight, gate_bias, k):
    """
    Args (numpy arrays or anything torch.as_tensor accepts):
        hidden:      [N, D]
        gate_weight: [D, E]
        gate_bias:   [E]
        k: int

    Returns: (topk_idx [N,k] int64 numpy, topk_val [N,k] float32 numpy)
    """
    device = _device()
    h = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    w = torch.as_tensor(gate_weight, dtype=torch.float32, device=device)
    b = torch.as_tensor(gate_bias, dtype=torch.float32, device=device)

    logits = h @ w + b                                   # [N, E]
    topk_val_raw, topk_idx = torch.topk(logits, k, dim=-1)  # torch.topk ties -> lower index first
    topk_val = torch.softmax(topk_val_raw, dim=-1)

    return topk_idx.cpu().numpy(), topk_val.cpu().numpy()


def pytorch_quantized_matmul(x, q_int8, scale):
    """
    Args:
        x:      [N, D] float
        q_int8: [D, E] int8
        scale:  [1, E] float
    Returns: [N, E] float32 numpy
    """
    device = _device()
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    q_t = torch.as_tensor(q_int8, dtype=torch.float32, device=device)  # widen for the multiply
    s_t = torch.as_tensor(scale, dtype=torch.float32, device=device)

    w_deq = q_t * s_t  # [D, E]
    out = x_t @ w_deq
    return out.cpu().numpy()


if __name__ == "__main__":
    import numpy as np

    print(f"torch: mps available = {torch.backends.mps.is_available()}")
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(4, 16)).astype(np.float32)
    gate_w = rng.normal(size=(16, 8)).astype(np.float32) * 0.1
    gate_b = rng.normal(size=(8,)).astype(np.float32) * 0.01

    idx, val = pytorch_gating(hidden, gate_w, gate_b, k=3)
    print("topk_idx:\n", idx)
    print("topk_val:\n", val)
    print("row sums (should be ~1.0):", val.sum(axis=-1))
