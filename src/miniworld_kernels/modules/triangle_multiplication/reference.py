"""PyTorch reference for the full triangle multiplicative update.

Signature matches ``cuequivariance_torch.triangle_multiplicative_update`` and
the perf/trimul ``fused_triangle_multiplicative_update_dtv1`` kernel:

    x_normed = LayerNorm(x; norm_in)
    left     = sigmoid(x_normed @ g_in_left.T) * (x_normed @ p_in_left.T)
    right    = sigmoid(x_normed @ g_in_right.T) * (x_normed @ p_in_right.T)
    # outgoing:  einsum('bikd,bjkd->bijd', left, right)
    # incoming:  einsum('bkid,bkjd->bijd', left, right)
    out_n    = LayerNorm(contraction; norm_out)
    y        = sigmoid(x_normed @ g_out.T) * (out_n @ p_out.T)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def triangle_multiplicative_update_pytorch(
    x: torch.Tensor,  # (B, I, J, D)
    direction: str,  # "outgoing" or "incoming"
    mask: torch.Tensor | None,  # (B, I, J) bool 2D pair mask, or None
    norm_in_weight: torch.Tensor,  # (D,)
    norm_in_bias: torch.Tensor,  # (D,)
    p_in_weight: torch.Tensor,  # (2D, D)  [p_left; p_right]
    g_in_weight: torch.Tensor,  # (2D, D)  [g_left; g_right]
    norm_out_weight: torch.Tensor,  # (D,)
    norm_out_bias: torch.Tensor,  # (D,)
    p_out_weight: torch.Tensor,  # (D, D)
    g_out_weight: torch.Tensor,  # (D, D)
    eps: float = 1e-5,
) -> torch.Tensor:
    D = x.shape[-1]
    x_normed = F.layer_norm(x, (D,), norm_in_weight, norm_in_bias, eps)

    a_full = x_normed @ p_in_weight.T  # (..., 2D)
    g_full = x_normed @ g_in_weight.T
    a_full = torch.sigmoid(g_full) * a_full
    left, right = a_full[..., :D], a_full[..., D:]

    if mask is not None:
        # dtv1 takes a pre-computed 2D pair mask of shape (B, I, J).
        m = mask.to(left.dtype).unsqueeze(-1)
        left = left * m
        right = right * m

    if direction == "outgoing":
        contraction = torch.einsum("bikd,bjkd->bijd", left, right)
    elif direction == "incoming":
        contraction = torch.einsum("bkid,bkjd->bijd", left, right)
    else:
        raise ValueError(f"direction must be outgoing|incoming, got {direction!r}")

    out_n = F.layer_norm(contraction, (D,), norm_out_weight, norm_out_bias, eps)
    y = torch.sigmoid(x_normed @ g_out_weight.T) * (out_n @ p_out_weight.T)
    return y
