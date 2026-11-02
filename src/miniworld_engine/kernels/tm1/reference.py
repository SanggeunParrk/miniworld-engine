"""PyTorch reference for the tm1 fused dual sigmoid-gate kernel.

Matches ``triton_tm1`` semantics: ``(sigmoid(x@WLg) * (x@WL),
sigmoid(x@WRg) * (x@WR))``. The reference is the plain functional form (:func:`tm1_pytorch`): a kernel is checked against it on both the forward output and the backward gradients.
"""

from __future__ import annotations

import torch


def tm1_pytorch(
    x: torch.Tensor,  # (..., D)
    WL: torch.Tensor,  # (D, D)
    WLg: torch.Tensor,  # (D, D)
    WR: torch.Tensor,  # (D, D)
    WRg: torch.Tensor,  # (D, D)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(sigmoid(x@WLg) * (x@WL), sigmoid(x@WRg) * (x@WR))``."""
    left = torch.sigmoid(x @ WLg) * (x @ WL)
    right = torch.sigmoid(x @ WRg) * (x @ WR)
    return left, right
