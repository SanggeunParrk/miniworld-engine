"""PyTorch reference for the fused LayerNorm + pair-mask kernel.

Mirrors ``kernels/fused_ln_mask/cute/fused_ln_mask.py::fused_ln_mask(x, weight, bias,
mask, eps)`` — same tensors, same order, same output shape.

``x`` is ``(B, L, L, D)`` and ``mask`` is a 2D PAIR mask ``(B, L, L)`` (bool or float),
one scalar per row of the flattened ``(B*L*L, D)`` matrix the kernel walks. The
LayerNorm reduces over the LAST axis ``D`` only — mean and variance are per
``(b, r, c)`` row, so the mask never enters the reduction. It is applied afterwards, as
a plain per-row scale on the already-affine result::

    mean[b,r,c] = (1/D) * sum_d x[b,r,c,d]
    var[b,r,c]  = (1/D) * sum_d (x[b,r,c,d] - mean[b,r,c])**2
    normed      = (x[b,r,c,d] - mean) / sqrt(var + eps) * weight[d] + bias[d]
    out[b,r,c,d] = normed * mask[b,r,c]

Because the multiply comes after the affine, a masked-out row is exactly zero including
the LayerNorm ``bias`` term, and the mask contributes a gradient path of its own when it
is a float tensor.

Dtypes. Data stays in the caller's dtype (bf16 in production) while the reduction, the
normalization, the affine and the mask multiply all run in fp32, with a single rounding
back to ``x.dtype`` at the store — matching the kernel, which loads everything
``.to(tl.float32)`` and casts only in ``tl.store``. The launcher casts ``mask`` to
``x.dtype`` before the kernel sees it, so a float mask is rounded to bf16 first; that
cast is reproduced here.

The kernel's winning (covering-tile) branch uses the centred variance
``sum_d (x - mean)**2 / D``, which is what this file implements; its tiled branch uses
the algebraically equivalent ``sum_d x**2 / D - mean**2`` so each tile is read once.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fused_ln_mask_pytorch(
    x: torch.Tensor,  # (B, L, L, D)
    weight: torch.Tensor,  # (D,)
    bias: torch.Tensor,  # (D,)
    mask: torch.Tensor,  # (B, L, L) bool or float — one scalar per row
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm over the last axis, then a per-row multiply by ``mask``.

    Returns ``(B, L, L, D)`` in ``x.dtype``. Differentiable in ``x``, ``weight``,
    ``bias`` and (when it is a floating tensor) ``mask``.
    """
    # fp32 accumulation for bf16/fp16/fp32 data; fp64 inputs (gradcheck) keep fp64.
    acc = torch.promote_types(x.dtype, torch.float32)
    xf = x.to(acc)
    mean = xf.mean(dim=-1, keepdim=True)
    centred = xf - mean
    var = centred.square().mean(dim=-1, keepdim=True)
    normed = centred * torch.rsqrt(var + eps) * weight.to(acc) + bias.to(acc)

    # The launcher stores the mask in x's dtype, so a float mask loses precision before
    # it is ever multiplied in; round it the same way.
    scale = mask.to(x.dtype).to(acc).unsqueeze(-1)
    return (normed * scale).to(x.dtype)


class FusedLNMaskReference(nn.Module):
    """nn.Module reference for fused_ln_mask (forward + backward ground truth).

    Owns the LayerNorm ``weight``/``bias`` as ``nn.Parameter`` so a kernel can be
    checked against it on both the forward output and the backward gradients.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x`` : (B, L, L, D), ``mask`` : (B, L, L) -> (B, L, L, D)."""
        return fused_ln_mask_pytorch(x, self.weight, self.bias, mask, self.eps)
