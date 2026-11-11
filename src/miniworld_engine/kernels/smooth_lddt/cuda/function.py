"""The CUDA backend of ``SmoothLddtLoss``: prep -> fwd (-> bwd) launches of ``sm100`` (one cubin per arch: sm_90a H100, sm_100a B200) as an
autograd function. Inputs are the padded, prepared tensors of the module (``modules/smooth_lddt.py``): ``x`` [A, B, N, 3] fp32 / bf16,
``gt`` [B, N, 3] fp32, ``nuc`` / ``mask`` [B, N] int8, N % 128 == 0."""

from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

BLK = 128


def _smooth_lddt_prep_fake(gt, nuc, mask):
    """[tile flags [B, NB, NB] int8, per-row pair counts [B, N] int32, pair bit masks FM and UM [B, N, N // 32] int32]."""
    b, n = mask.shape
    nb = n // BLK
    return [
        torch.empty((b, nb, nb), dtype=torch.int8, device=gt.device),
        torch.empty((b, n), dtype=torch.int32, device=gt.device),
        torch.empty((b, n, n // 32), dtype=torch.int32, device=gt.device),
        torch.empty((b, n, n // 32), dtype=torch.int32, device=gt.device),
    ]


@opaque(fake=_smooth_lddt_prep_fake, name="smooth_lddt_prep")
def smooth_lddt_prep(gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """Ground-truth pair set of gt [B, N, 3] fp32 (N % 128 == 0): [tile flags, per-row pair counts, FM, UM bit masks]."""
    from miniworld_engine.kernels.smooth_lddt.cuda import sm100

    return list(sm100.prep(gt, nuc, mask))


def _smooth_lddt_fwd_fake(x, gt, tile, fm):
    """Partial sums [A * B, NB] fp32."""
    return torch.empty((x.shape[0] * x.shape[1], x.shape[2] // BLK), dtype=torch.float32, device=x.device)


@opaque(fake=_smooth_lddt_fwd_fake, name="smooth_lddt_fwd")
def smooth_lddt_fwd(x: torch.Tensor, gt: torch.Tensor, tile: torch.Tensor, fm: torch.Tensor) -> torch.Tensor:
    """Partial sums of the pair scores per (augment * batch, row block), [A * B, NB] fp32."""
    from miniworld_engine.kernels.smooth_lddt.cuda import sm100

    return sm100.forward(x, gt, tile, fm)


def _smooth_lddt_bwd_fake(x, gt, nuc, tile, um, go):
    """The coordinate gradient, the shape of x (fp32)."""
    return torch.empty_like(x)


@opaque(fake=_smooth_lddt_bwd_fake, name="smooth_lddt_bwd")
def smooth_lddt_bwd(
    x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, tile: torch.Tensor, um: torch.Tensor, go: torch.Tensor
) -> torch.Tensor:
    """d(sum of the pair scores)/dx [A, B, N, 3] fp32 for the upstream gradient ``go`` [A * B] fp32."""
    from miniworld_engine.kernels.smooth_lddt.cuda import sm100

    return sm100.backward(x, gt, nuc, tile, um, go)


class SmoothLddtFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gt, nuc, tile, fm, um):
        ctx.in_dtype = x.dtype
        x32 = x.float().contiguous()
        ctx.save_for_backward(x32, gt, nuc, tile, um)
        return smooth_lddt_fwd(x32, gt, tile, fm).sum(dim=1).view(x.shape[0], x.shape[1])

    @staticmethod
    def backward(ctx, grad_sum):
        x32, gt, nuc, tile, um = ctx.saved_tensors
        gx = smooth_lddt_bwd(x32, gt, nuc, tile, um, grad_sum.contiguous().view(-1).float())
        return gx.to(ctx.in_dtype), None, None, None, None, None


def smooth_lddt_cuda(x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``mean_{a, b} (1 - smooth lDDT)`` (fp32 scalar), differentiable in ``x``."""
    tile, cnt, fm, um = smooth_lddt_prep(gt, nuc, mask)
    denom = cnt.sum(dim=1).clamp(min=1).to(torch.float32)  # [B] pairs of the ground truth
    total = SmoothLddtFunction.apply(x, gt, nuc, tile, fm, um)  # [A, B] sum of the pair scores
    return (1.0 - total / denom).mean()
