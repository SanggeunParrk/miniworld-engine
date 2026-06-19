"""PyTorch reference for the tm1 fused dual sigmoid-gate kernel.

Matches `team_gm.modules.kernels.tm1.triton_tm1` semantics. Weight tensors are
already transposed by the caller (the layer passes ``Linear.weight.T``), so
they are the "right-side" operands of a plain matmul.
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
