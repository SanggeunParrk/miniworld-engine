"""The torch reference the rmsnorm kernels are checked against.

``F.rms_norm`` is the same function; it is written out here so the checker compares against the
formula rather than against another implementation that could move underneath it.
"""

from __future__ import annotations

import torch


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor | None = None,
                      eps: float = 1e-5) -> torch.Tensor:
    """``x / sqrt(mean(x^2) + eps) * weight`` over the last axis, computed in fp32."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def rmsnorm_modulate_reference(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                               weight: torch.Tensor | None = None,
                               eps: float = 1e-5) -> torch.Tensor:
    """``rmsnorm(x) * (1 + scale) + shift``, the adaLN-Zero modulate, in fp32 throughout.

    The normalization is inlined rather than delegated to `rmsnorm_reference` for the reason that
    function's caller in `rmsnorm_adamod_reference` gives: it rounds its result to the input
    dtype, which is correct when the normalization IS the op and wrong for an intermediate the
    fused kernel keeps in fp32.
    """
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        normed = normed * weight.float()
    return (normed * (1.0 + scale.float()) + shift.float()).to(x.dtype)


def rmsnorm_adamod_reference(q: torch.Tensor, c: torch.Tensor, w_scale: torch.Tensor,
                             w_shift: torch.Tensor, w_gate: torch.Tensor,
                             weight: torch.Tensor | None = None,
                             eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rmsnorm(q) * (1 + c@Wsc^T) + c@Wsh^T,  c@Wg^T)`` -- the projection written out.

    Spelled as two `Linear`s and a modulate -- the arrangement the fused kernel replaces -- so a
    disagreement points at the fusion rather than at a differently-factored reference.
    """
    # The normalization is spelled out rather than delegated to `rmsnorm_reference`, which
    # rounds its result to the input dtype: that rounding is correct when the normalization IS
    # the op, and wrong here, where it is an intermediate the fused kernel keeps in fp32. Calling
    # it made the reference the less accurate of the two and put that gap into `dweight`.
    qf = q.float()
    normed = qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        normed = normed * weight.float()
    scale = torch.nn.functional.linear(c.float(), w_scale.float())
    shift = torch.nn.functional.linear(c.float(), w_shift.float())
    y = (normed * (1.0 + scale) + shift).to(q.dtype)
    return y, torch.nn.functional.linear(c.float(), w_gate.float()).to(q.dtype)
