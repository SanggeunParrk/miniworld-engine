"""PyTorch reference for tm2: ``sigmoid(x_gate @ W_gate) * (x_out @ W_out)``.

The kernel is checked against the functional form (:func:`tm2_pytorch`) on both forward output and backward gradients.
"""

from __future__ import annotations

import torch


def tm2_pytorch(
    x_gate: torch.Tensor,  # (..., D)
    x_out: torch.Tensor,  # (..., D)
    W_gate: torch.Tensor,  # (D, D)
    W_out: torch.Tensor,  # (D, D)
) -> torch.Tensor:
    return torch.sigmoid(x_gate @ W_gate) * (x_out @ W_out)
