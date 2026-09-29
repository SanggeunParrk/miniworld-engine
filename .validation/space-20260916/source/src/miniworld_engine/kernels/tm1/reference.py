"""PyTorch reference for the tm1 fused dual sigmoid-gate kernel.

Matches ``triton_tm1`` semantics: ``(sigmoid(x@WLg) * (x@WL),
sigmoid(x@WRg) * (x@WR))``. Provided as an ``nn.Module`` (owns the four weight
matrices as parameters) so a kernel can be checked against it on both the
forward output and the backward gradients, e.g.::

    ref = TM1Reference(D).cuda().to(torch.bfloat16)
    lo, ro = ref(x)                       # reference forward
    lk, rk = triton_tm1(x, ref.WL, ref.WLg, ref.WR, ref.WRg)  # kernel forward
    (lo.sum() + ro.sum()).backward()      # reference backward -> ref.WL.grad, x.grad

A plain functional form (:func:`tm1_pytorch`) is kept for callers that already
hold the weight tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


class TM1Reference(nn.Module):
    """nn.Module reference for tm1 (forward + backward ground truth)."""

    def __init__(self, d: int, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        scale = d**-0.5

        def w() -> nn.Parameter:
            return nn.Parameter(torch.randn(d, d, dtype=dtype) * scale)

        self.WL, self.WLg, self.WR, self.WRg = w(), w(), w(), w()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass; differentiable in ``x`` and all four weights."""
        return tm1_pytorch(x, self.WL, self.WLg, self.WR, self.WRg)
