"""PyTorch reference for the fused trimul input-projection kernel.

Produces the three input projections of the triangle multiplicative update::

    left  = sigmoid(x @ WLg) * (x @ WL)     -> [B, D, L, L]
    right = sigmoid(x @ WRg) * (x @ WR)     -> [B, D, L, L]
    gate  = sigmoid(x @ Wg)                 -> [B, L, L, D]

``left``/``right`` come out in ``[B, D, L, L]`` so the downstream outgoing
contraction is a flat batched matmul (``einsum("bdik,bdjk->bdij")`` == bmm) with
no transpose. ``gate`` stays in ``[B, L, L, D]`` because it is multiplied against
the ``[B, L, L, D]`` output at the very end (it never enters the bmm).

Weight convention mirrors ``tm1``: each ``W*`` is ``(D, D)`` used as ``x @ W``,
i.e. the caller passes ``nn.Linear.weight.T``.
"""

from __future__ import annotations

import torch


def trimul_inproj_pytorch(
    x: torch.Tensor,  # (B, L, L, D)
    WL: torch.Tensor,  # (D, D)  — to_left.weight.T
    WLg: torch.Tensor,  # (D, D)  — to_left_gate.weight.T
    WR: torch.Tensor,  # (D, D)  — to_right.weight.T
    WRg: torch.Tensor,  # (D, D)  — to_right_gate.weight.T
    Wg: torch.Tensor,  # (D, D)  — to_gate.weight.T
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(left_bdll, right_bdll, gate_blld)``."""
    left = torch.sigmoid(x @ WLg) * (x @ WL)  # (B, L, L, D)
    right = torch.sigmoid(x @ WRg) * (x @ WR)  # (B, L, L, D)
    gate = torch.sigmoid(x @ Wg)  # (B, L, L, D), stays blld
    left_bdll = left.permute(0, 3, 1, 2).contiguous()  # (B, D, L, L)
    right_bdll = right.permute(0, 3, 1, 2).contiguous()  # (B, D, L, L)
    return left_bdll, right_bdll, gate
