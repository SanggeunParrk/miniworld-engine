"""PyTorch reference for the fused SwiGLU Transition kernel.

Mirrors the whole-op entry point ``kernels.transition.whole_op.transition`` (and the
kernel it wraps, ``kernels.triton_transition_fused``): LayerNorm folded into a SwiGLU
expand pair followed by the squeeze projection, over the last dimension::

    xn = LayerNorm(x; ln_in_weight, ln_in_bias, eps)   # (..., D)
    a  = xn @ expand_a_weight.T                        # (..., n*D)
    b  = xn @ expand_b_weight.T                        # (..., n*D)
    h  = silu(a) * b = a * sigmoid(a) * b              # SwiGLU, (..., n*D)
    y  = h @ squeeze_weight.T                          # (..., D)
    y  = y + x                                         # only if add_residual

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``), matching the
kernel argument layout. ``add_residual`` reproduces the kernel's fused residual
epilogue; ``modules.Transition`` always requests it, the whole-op does not.

Provided as an ``nn.Module`` (:class:`TransitionReference`, owns every weight as a
parameter) so a kernel can be checked on both forward output and backward gradients::

    ref = TransitionReference(d_hidden=128, n=4).cuda()
    y = ref(x)                                         # reference forward
    yk = transition(                                   # kernel forward
        x, ln_in_weight=ref.ln_in_weight, ln_in_bias=ref.ln_in_bias,
        expand_a_weight=ref.expand_a_weight, expand_b_weight=ref.expand_b_weight,
        squeeze_weight=ref.squeeze_weight, n=4,
    )
    y.sum().backward()                                 # -> ref.expand_a_weight.grad, x.grad

A plain functional form (:func:`transition_pytorch`) is kept for callers that already
hold the weight tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def transition_pytorch(
    x: torch.Tensor,  # (..., D)
    ln_in_weight: torch.Tensor,  # (D,)
    ln_in_bias: torch.Tensor,  # (D,)
    expand_a_weight: torch.Tensor,  # (n*D, D)
    expand_b_weight: torch.Tensor,  # (n*D, D)
    squeeze_weight: torch.Tensor,  # (D, n*D)
    n: int,
    eps: float = 1e-5,
    add_residual: bool = False,
) -> torch.Tensor:
    """Compute ``squeeze(silu(a) * b)`` over ``LayerNorm(x)``. Returns ``x``'s shape.

    ``n`` is carried only for signature parity with the kernel (the expansion factor is
    already implied by the weight shapes). The LayerNorm affine params are cast to the
    activation dtype: the module keeps them fp32-pinned while the trunk runs bf16, and
    the kernel likewise upcasts internally and normalises back to ``x.dtype``.
    """
    xn = F.layer_norm(
        x,
        (x.shape[-1],),
        ln_in_weight.to(x.dtype),
        ln_in_bias.to(x.dtype),
        eps,
    )
    a = F.linear(xn, expand_a_weight)
    b = F.linear(xn, expand_b_weight)
    h = F.silu(a) * b
    out = F.linear(h, squeeze_weight)
    return out + x if add_residual else out


class TransitionReference(nn.Module):
    """nn.Module reference for the SwiGLU transition (forward + backward ground truth)."""

    def __init__(
        self,
        d_hidden: int = 128,
        n: int = 4,
        *,
        eps: float = 1e-5,
        add_residual: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.n = n
        self.eps = eps
        self.add_residual = add_residual
        nd = n * d_hidden

        def w(out_features: int, in_features: int) -> nn.Parameter:
            scale = in_features**-0.5
            t = torch.randn(out_features, in_features, device=device, dtype=dtype)
            return nn.Parameter(t * scale)

        # LayerNorm affine is created in fp32 rather than the trunk dtype, mirroring the
        # fp32-pinned norm params of ``modules.Transition``; forward casts them to the
        # activation dtype.
        self.ln_in_weight = nn.Parameter(torch.ones(d_hidden, device=device))
        self.ln_in_bias = nn.Parameter(torch.zeros(d_hidden, device=device))
        self.expand_a_weight = w(nd, d_hidden)
        self.expand_b_weight = w(nd, d_hidden)
        self.squeeze_weight = w(d_hidden, nd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; differentiable in ``x`` and every weight."""
        return transition_pytorch(
            x,
            self.ln_in_weight,
            self.ln_in_bias,
            self.expand_a_weight,
            self.expand_b_weight,
            self.squeeze_weight,
            self.n,
            self.eps,
            self.add_residual,
        )
