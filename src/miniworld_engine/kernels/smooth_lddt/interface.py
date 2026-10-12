"""Smooth lDDT of the diffusion loss (kernel family ``smooth_lddt``): AlphaFold 3 Algorithm 27,

    L = mean_{augment a, batch b} (1 - lddt_ab),   lddt_ab = sum_{pairs (i, j)} keep_ij * mean_t sigmoid(t - |d_pred_ij - d_gt_ij|) / sum keep,

t in (0.5, 1, 2, 4) A, ``keep`` = both atoms resolved, i != j, ``d_gt(i, j) < 30 A`` for a nucleotide atom i and ``15 A`` otherwise.

The reference (and the old PyTorch loss) builds an [N, N] tensor per augment (48 x 4096^2 pairs at the diffusion training shape: seconds
of elementwise passes and 3 GB intermediates). The kernels build none: distances come from the coordinates tile by tile, the four
sigmoids share one exponential, and tiles with no pair inside the radius are skipped. The augments are rigid motions of one structure,
so the ground-truth pair set, its normaliser and the tile flags are computed once per call. The backward is analytic, deterministic
(every atom collects the gradient of both of its roles in a pair; no atomics) and CUDA-graph capturable.

Served: B200 (sm_100a). The kernels compute in fp32; ``x_pred`` may be fp32 or bf16 (cast outside autograd, the gradient goes back in its
dtype). :func:`refusal` says why anything else is not served; it never raises.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

BLK = 128


def refusal(x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> str | None:
    """None if the fused op can run this call, else why it cannot. Only metadata is read (traces under ``torch.compile``)."""
    if not x_pred.is_cuda:
        return "the input is not on a CUDA device"
    if x_pred.dtype not in (torch.float32, torch.bfloat16):
        return f"x_pred must be fp32 or bf16, got {x_pred.dtype}"
    if x_pred.dim() != 4 or x_pred.shape[-1] != 3:
        return f"x_pred must be [A, B, N, 3], got {tuple(x_pred.shape)}"
    a, b, n, _ = x_pred.shape
    if tuple(gt.shape) != (b, n, 3):
        return f"gt must be [B, N, 3] = {(b, n, 3)}, got {tuple(gt.shape)}"
    if tuple(is_nuc.shape) != (b, n) or tuple(mask.shape) != (b, n):
        return f"is_nuc / mask must be [B, N] = {(b, n)}, got {tuple(is_nuc.shape)} / {tuple(mask.shape)}"
    if torch.cuda.get_device_capability(x_pred.device) != (10, 0):
        return "served on B200 (sm_100a) only"
    return None


def _smooth_lddt_prep_fake(gt, nuc, mask):
    """[tile flags [B, NB, NB] int8, per-row pair counts [B, N] int32]."""
    b, n = mask.shape
    nb = n // BLK
    return [torch.empty((b, nb, nb), dtype=torch.int8, device=gt.device), torch.empty((b, n), dtype=torch.int32, device=gt.device)]


@opaque(fake=_smooth_lddt_prep_fake, name="smooth_lddt_prep")
def smooth_lddt_prep(gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """[tile flags [B, NB, NB] int8, per-row pair counts [B, N] int32] of the ground truth gt [B, N, 3] fp32 (N % 128 == 0)."""
    from miniworld_engine.kernels.smooth_lddt.cuda import sm100

    return list(sm100.prep(gt, nuc, mask))


def _smooth_lddt_fwd_fake(x, gt, nuc, mask, tile):
    """Partial sums [A * B, NB] fp32."""
    return torch.empty((x.shape[0] * x.shape[1], x.shape[2] // BLK), dtype=torch.float32, device=x.device)


@opaque(fake=_smooth_lddt_fwd_fake, name="smooth_lddt_fwd")
def smooth_lddt_fwd(x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor, tile: torch.Tensor) -> torch.Tensor:
    """Partial sums of the pair scores per (augment * batch, row block), [A * B, NB] fp32."""
    from miniworld_engine.kernels.smooth_lddt.cuda import sm100

    return sm100.forward(x, gt, nuc, mask, tile)


def _smooth_lddt_bwd_fake(x, gt, nuc, mask, tile, go):
    """The coordinate gradient, the shape of x (fp32)."""
    return torch.empty_like(x)


@opaque(fake=_smooth_lddt_bwd_fake, name="smooth_lddt_bwd")
def smooth_lddt_bwd(x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor, tile: torch.Tensor,
                    go: torch.Tensor) -> torch.Tensor:
    """d(sum of the pair scores)/dx [A, B, N, 3] fp32 for the upstream gradient ``go`` [A * B] fp32."""
    from miniworld_engine.kernels.smooth_lddt.cuda import sm100

    return sm100.backward(x, gt, nuc, mask, tile, go)


class SmoothLddtFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gt, nuc, mask, tile):
        ctx.in_dtype = x.dtype
        x32 = x.float().contiguous()
        ctx.save_for_backward(x32, gt, nuc, mask, tile)
        return smooth_lddt_fwd(x32, gt, nuc, mask, tile).sum(dim=1).view(x.shape[0], x.shape[1])

    @staticmethod
    def backward(ctx, grad_sum):
        x32, gt, nuc, mask, tile = ctx.saved_tensors
        gx = smooth_lddt_bwd(x32, gt, nuc, mask, tile, grad_sum.contiguous().view(-1).float())
        return gx.to(ctx.in_dtype), None, None, None, None


def smooth_lddt(x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``mean_{a, b} (1 - smooth lDDT)``: differentiable in ``x_pred`` [A, B, N, 3]; ``gt`` [B, N, 3] is the (un-noised, un-augmented) structure,
    ``is_nuc`` / ``mask`` [B, N] bool (nucleotide atoms, resolved atoms). Check :func:`refusal` first."""
    a, b, n, _ = x_pred.shape
    pad = (-n) % BLK
    if pad:
        x_pred = torch.nn.functional.pad(x_pred, (0, 0, 0, pad))
        gt = torch.nn.functional.pad(gt, (0, 0, 0, pad))
        is_nuc = torch.nn.functional.pad(is_nuc, (0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
    gt32 = gt.detach().float().contiguous()
    nuc8 = is_nuc.detach().to(torch.int8).contiguous()
    mask8 = mask.detach().to(torch.int8).contiguous()
    tile, cnt = smooth_lddt_prep(gt32, nuc8, mask8)
    denom = cnt.sum(dim=1).clamp(min=1).to(torch.float32)               # [B] pairs of the ground truth
    total = SmoothLddtFunction.apply(x_pred, gt32, nuc8, mask8, tile)   # [A, B] sum of the pair scores
    return (1.0 - total / denom).mean()


__all__ = ["refusal", "smooth_lddt"]
