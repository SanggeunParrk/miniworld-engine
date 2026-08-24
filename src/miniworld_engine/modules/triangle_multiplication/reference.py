"""PyTorch reference for the full triangle multiplicative update.

``TriangleMultiplicationReference`` is an ``nn.Module`` ground truth: it owns the
same parameters (same names/shapes) as the kernel-backed
:class:`~miniworld_engine.modules.triangle_multiplication.module.TriangleMultiplication`,
so a kernel implementation can be checked against it on both forward output AND
backward gradients::

    ref = TriangleMultiplicationReference(d_pair).cuda()
    mod = TriangleMultiplication(d_pair, implementation=ImplementationType.TRITON).cuda()
    mod.load_state_dict(ref.state_dict())          # share weights
    y_ref = ref(pair, mask);  y_ref.sum().backward()
    y_mod = mod(pair, mask)                          # compare fwd + param.grad (bwd)

A functional form (:func:`triangle_multiplicative_update_pytorch`, the
cuequivariance / dtv1 weight API with concatenated p_in/g_in) is kept for
baseline comparisons that already hold raw weight tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear


class TriangleMultiplicationReference(nn.Module):
    """Pure-PyTorch nn.Module reference (forward + backward ground truth).

    Mirrors the parameter layout of the kernel-backed ``TriangleMultiplication``
    so ``state_dict`` transfers directly between the two.
    """

    def __init__(
        self,
        d_pair: int = 128,
        *,
        d_hidden: int | None = None,
        outgoing: bool = True,
    ) -> None:
        super().__init__()
        self.outgoing = outgoing
        if d_hidden is None:
            d_hidden = d_pair

        self.ln_pair = LayerNorm(d_pair)
        self.to_left = Linear(d_pair, d_pair, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_right = Linear(d_pair, d_pair, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d_pair, bias=False, init="zero")

        self.ln_out = LayerNorm(d_pair)
        self.to_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_out = Linear(d_hidden, d_pair, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass; fully differentiable for fwd/bwd comparison."""
        x_normed = self.ln_pair(pair)
        left = sigmoid_gate(self.to_left_gate(x_normed), self.to_left(x_normed))
        right = sigmoid_gate(self.to_right_gate(x_normed), self.to_right(x_normed))

        if mask is not None:
            mask_2d = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)
            left = left * mask_2d
            right = right * mask_2d

        if self.outgoing:
            out = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            out = torch.einsum("bkid,bkjd->bijd", left, right)

        out = self.ln_out(out)
        return sigmoid_gate(self.to_gate(x_normed), self.to_out(out))


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
    """Functional reference in the cuequivariance / dtv1 weight API."""
    D = x.shape[-1]
    x_normed = F.layer_norm(x, (D,), norm_in_weight, norm_in_bias, eps)

    a_full = x_normed @ p_in_weight.T  # (..., 2D)
    g_full = x_normed @ g_in_weight.T
    a_full = torch.sigmoid(g_full) * a_full
    left, right = a_full[..., :D], a_full[..., D:]

    if mask is not None:
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
    return torch.sigmoid(x_normed @ g_out_weight.T) * (out_n @ p_out_weight.T)
