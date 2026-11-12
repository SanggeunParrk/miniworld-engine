"""Smooth lDDT of the diffusion loss (kernel family ``smooth_lddt``): AlphaFold 3 Algorithm 27,

    L = mean_{augment a, batch b} (1 - lddt_ab),   lddt_ab = sum_{pairs (i, j)} keep_ij * mean_t sigmoid(t - |d_pred_ij - d_gt_ij|) / sum keep,

t in (0.5, 1, 2, 4) A, ``keep`` = both atoms resolved, i != j, ``d_gt(i, j) < 30 A`` for a nucleotide atom i and ``15 A`` otherwise.

The reference (and the old PyTorch loss) builds an [N, N] tensor per augment (48 x 4096^2 pairs at the diffusion training shape: seconds
of elementwise passes and 3 GB intermediates). The kernels build none: the ground-truth pair set is kept as bit masks (N x N bits, built once per call), distances come from the
coordinates, the four sigmoids share one exponential, and only the pairs inside the radius are visited (tiles with none are skipped). The augments are rigid motions of one structure,
so the ground-truth pair set, its normaliser and the tile flags are computed once per call. The backward is analytic, deterministic
(every atom collects the gradient of both of its roles in a pair; no atomics) and CUDA-graph capturable.

Served: B200 (sm_100a) and H100 (sm_90a, same source) by the CUDA kernels; every other card by the portable kernels (``triton/main.py``,
dense 64 x 64 tiles with per-tile skip flags, fixed-order reductions, no atomics) once the engine holds a record for that exact card
(``autotune.records``, op ``smooth_lddt``), else by the PyTorch reference with an ``UnrecordedCardWarning``. ``settings.force_portable_kernels=True``
(the A/B switch) runs the portable kernels on any card, H100 and B200 included. The kernels compute in fp32; ``x_pred`` may be fp32 or bf16 (cast
outside autograd, the gradient goes back in its dtype). :func:`refusal` says why a call cannot be served at all; it never raises.
"""
from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.smooth_lddt.reference import smooth_lddt_reference

BLK = 128
PORTABLE_BLK = 128   # N is padded to the largest tile edge of the portable kernels (``triton/main.py``: BLOCK_I, BLOCK_J, BLOCK_P up to 128)


def refusal(x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> str | None:
    """None if a fused path (CUDA, or the portable kernels on a card the engine has a record for) can take this call's tensors, else why not.
    A card with no record is not a refusal: :func:`smooth_lddt` warns once and runs the PyTorch reference there. Only metadata is read."""
    if not x_pred.is_cuda:
        return "the input is not on a CUDA device"
    if x_pred.dtype not in (torch.float32, torch.bfloat16):
        return f"x_pred must be fp32 or bf16, got {x_pred.dtype}"
    if x_pred.dim() != 4 or x_pred.shape[-1] != 3:
        return f"x_pred must be [A, B, N, 3], got {tuple(x_pred.shape)}"
    _a, b, n, _ = x_pred.shape
    if tuple(gt.shape) != (b, n, 3):
        return f"gt must be [B, N, 3] = {(b, n, 3)}, got {tuple(gt.shape)}"
    if tuple(is_nuc.shape) != (b, n) or tuple(mask.shape) != (b, n):
        return f"is_nuc / mask must be [B, N] = {(b, n)}, got {tuple(is_nuc.shape)} / {tuple(mask.shape)}"
    return None


def _backend(x_pred: torch.Tensor) -> str:
    """``"cuda"`` (H100 / B200), ``"portable"`` (A/B switch, or a recorded card) or ``"reference"`` (an unrecorded card: one warning)."""
    if settings.current().force_portable_kernels:
        return "portable"
    from miniworld_engine.modules import dispatch

    card = dispatch._card_of(x_pred.device)
    if card is None:
        return "reference"
    if card[1] in ((9, 0), (10, 0)):
        return "cuda"
    training = torch.is_grad_enabled() and x_pred.requires_grad
    mode = "training" if training else "inference"
    # prep reads the (fp32) ground truth whatever x_pred is, so the fp32 records are always needed; a bf16 x_pred needs the bf16 fwd / bwd rows too
    routed = dispatch.route_for_card("smooth_lddt", card[0], card[1], dtype="fp32", mode=mode)
    if routed is dispatch.KernelBackend.PORTABLE and x_pred.dtype == torch.bfloat16:
        routed = dispatch.route_for_card("smooth_lddt", card[0], card[1], dtype="bf16", mode=mode)
    return "portable" if routed is dispatch.KernelBackend.PORTABLE else "reference"


def _smooth_lddt_prep_fake(gt, nuc, mask):
    """[tile flags [B, NB, NB] int8, per-row pair counts [B, N] int32, pair bit masks FM and UM [B, N, N // 32] int32]."""
    b, n = mask.shape
    nb = n // BLK
    return [torch.empty((b, nb, nb), dtype=torch.int8, device=gt.device), torch.empty((b, n), dtype=torch.int32, device=gt.device),
            torch.empty((b, n, n // 32), dtype=torch.int32, device=gt.device), torch.empty((b, n, n // 32), dtype=torch.int32, device=gt.device)]


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
def smooth_lddt_bwd(x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, tile: torch.Tensor, um: torch.Tensor,
                    go: torch.Tensor) -> torch.Tensor:
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


class PortableSmoothLddtFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gt, nuc, mask):
        """``(sum of the pair scores [A, B], number of ground-truth pairs [B])``; the second output carries no gradient."""
        from miniworld_engine.kernels.smooth_lddt.triton.main import (
            smooth_lddt_fwd_portable,
            smooth_lddt_prep_portable,
        )

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
        from miniworld_engine.kernels.smooth_lddt.triton.main import (
            smooth_lddt_bwd_portable,
        )

        x32, gt, nuc, mask, tile = ctx.saved_tensors
        gx = smooth_lddt_bwd_portable(x32, gt, nuc, mask, tile, grad_sum.contiguous().view(-1).float())
        return gx.to(ctx.in_dtype), None, None, None


def smooth_lddt(x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``mean_{a, b} (1 - smooth lDDT)``: differentiable in ``x_pred`` [A, B, N, 3]; ``gt`` [B, N, 3] is the (un-noised, un-augmented) structure,
    ``is_nuc`` / ``mask`` [B, N] bool (nucleotide atoms, resolved atoms). Check :func:`refusal` first. Which kernels run: see the module docstring."""
    backend = _backend(x_pred)
    if backend == "reference":
        return smooth_lddt_reference(x_pred.float(), gt.float(), is_nuc, mask)
    portable = backend == "portable"
    blk = PORTABLE_BLK if portable else BLK
    _a, _b, n, _ = x_pred.shape
    pad = (-n) % blk
    if pad:
        x_pred = torch.nn.functional.pad(x_pred, (0, 0, 0, pad))
        gt = torch.nn.functional.pad(gt, (0, 0, 0, pad))
        is_nuc = torch.nn.functional.pad(is_nuc, (0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
    gt32 = gt.detach().float().contiguous()
    nuc8 = is_nuc.detach().to(torch.int8).contiguous()
    mask8 = mask.detach().to(torch.int8).contiguous()
    if portable:
        total, denom = PortableSmoothLddtFunction.apply(x_pred, gt32, nuc8, mask8)   # [A, B] sums of the pair scores, [B] pairs of the ground truth
        return (1.0 - total / denom).mean()
    tile, cnt, fm, um = smooth_lddt_prep(gt32, nuc8, mask8)
    denom = cnt.sum(dim=1).clamp(min=1).to(torch.float32)               # [B] pairs of the ground truth
    total = SmoothLddtFunction.apply(x_pred, gt32, nuc8, tile, fm, um)   # [A, B] sum of the pair scores
    return (1.0 - total / denom).mean()


__all__ = ["refusal", "smooth_lddt"]
