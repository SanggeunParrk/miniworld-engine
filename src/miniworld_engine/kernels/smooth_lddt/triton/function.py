"""The portable (Triton) backend of ``SmoothLddtLoss``: the prep / fwd / bwd kernels of ``main.py`` as an autograd function. Inputs are the
padded, prepared tensors of the module (``modules/smooth_lddt.py``): ``x`` [A, B, N, 3] fp32 / bf16, ``gt`` [B, N, 3] fp32, ``nuc`` /
``mask`` [B, N] int8, N % 128 == 0."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.smooth_lddt.triton.main import (
    smooth_lddt_bwd_portable,
    smooth_lddt_fwd_portable,
    smooth_lddt_prep_portable,
)


class PortableSmoothLddtFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gt, nuc, mask):
        """``(sum of the pair scores [A, B], number of ground-truth pairs [B])``; the second output carries no gradient."""
        ctx.in_dtype = x.dtype
        x32 = x.contiguous()   # fp32 or bf16: the kernels load either and compute in fp32
        tile, cnt = smooth_lddt_prep_portable(gt, nuc, mask)   # the pair set of the ground truth: once per call, shared by every augment
        denom = cnt.sum(dim=(1, 2)).clamp(min=1).to(torch.float32)
        ctx.mark_non_differentiable(denom)
        ctx.save_for_backward(x32, gt, nuc, mask, tile)
        # per-row scores, added in torch's fixed order
        return smooth_lddt_fwd_portable(x32, gt, nuc, mask, tile).sum(dim=1).view(x.shape[0], x.shape[1]), denom

    @staticmethod
    def backward(ctx, grad_sum, _grad_denom):
        x32, gt, nuc, mask, tile = ctx.saved_tensors
        gx = smooth_lddt_bwd_portable(x32, gt, nuc, mask, tile, grad_sum.contiguous().view(-1).float())
        return gx.to(ctx.in_dtype), None, None, None


def smooth_lddt_triton(x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``mean_{a, b} (1 - smooth lDDT)`` (fp32 scalar), differentiable in ``x``."""
    total, denom = PortableSmoothLddtFunction.apply(x, gt, nuc, mask)   # [A, B] sums of the pair scores, [B] pairs of the ground truth
    return (1.0 - total / denom).mean()
