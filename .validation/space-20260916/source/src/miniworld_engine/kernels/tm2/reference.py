"""PyTorch reference for tm2: ``sigmoid(x_gate @ W_gate) * (x_out @ W_out)``.

Provided as an ``nn.Module`` (owns ``W_gate`` / ``W_out`` as parameters) so the
kernel can be checked against it on both forward output and backward gradients.
A functional form (:func:`tm2_pytorch`) is kept for callers holding the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def tm2_pytorch(
    x_gate: torch.Tensor,  # (..., D)
    x_out: torch.Tensor,  # (..., D)
    W_gate: torch.Tensor,  # (D, D)
    W_out: torch.Tensor,  # (D, D)
) -> torch.Tensor:
    return torch.sigmoid(x_gate @ W_gate) * (x_out @ W_out)


class TM2Reference(nn.Module):
    """nn.Module reference for tm2 (forward + backward ground truth)."""

    def __init__(self, d: int, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        scale = d**-0.5
        self.W_gate = nn.Parameter(torch.randn(d, d, dtype=dtype) * scale)
        self.W_out = nn.Parameter(torch.randn(d, d, dtype=dtype) * scale)

    def forward(self, x_gate: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        """Forward pass; differentiable in both inputs and both weights."""
        return tm2_pytorch(x_gate, x_out, self.W_gate, self.W_out)
