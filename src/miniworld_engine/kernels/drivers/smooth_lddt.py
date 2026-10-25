"""Drivers for the ``smooth_lddt`` family (AF3 Algorithm 27, the portable kernels of ``smooth_lddt/triton/main.py``).

The predicted coordinates are the activation (bf16 or fp32, ``drivers.BF16``); the ground truth ``gt`` is always fp32 (the interface casts it:
it only fixes the pair set and the target distances), which is why ``smooth_lddt_prep_triton`` declares ``fp32`` alone. N is the atom count of one structure; the
tile edges go up to 128 and the caller pads N to it, so N stays aligned (``aligned_only``). The structure is a random chain (neighbours inside the
15 A radius, a few farther than 30 A apart), 10 % of the atoms unresolved, 20 % nucleotide, the augments the noised copies of it.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, aligned_only, dev, driver_length

_N = aligned_only("smooth_lddt N", driver_length(1024), "the tile edges go up to 128 atoms and the caller pads N to it")
_A, _B = 4, 2


def inputs(n: int = _N, a: int = _A, b: int = _B):
    """``(x [A, B, N, 3] in the activation dtype, gt [B, N, 3] fp32, nuc [B, N] int8, mask [B, N] int8)``."""
    gt = torch.cumsum(torch.randn(b, n, 3, device=dev()), dim=1) * 1.5
    nuc = (torch.rand(b, n, device=dev()) < 0.2).to(torch.int8)
    mask = (torch.rand(b, n, device=dev()) < 0.9).to(torch.int8)
    x = gt.unsqueeze(0) + torch.randn(a, b, n, 3, device=dev()) * 1.5
    return x.to(BF16).contiguous(), gt.contiguous(), nuc, mask


def smooth_lddt_prep_triton() -> None:
    """smooth_lddt_prep_kernel: tile flags and pair counts of the ground truth."""
    from miniworld_engine.kernels.smooth_lddt.triton.main import (
        smooth_lddt_prep_portable,
    )

    _x, gt, nuc, mask = inputs()
    smooth_lddt_prep_portable(gt, nuc, mask)


def smooth_lddt_fwd_triton() -> None:
    """smooth_lddt_fwd_kernel: the partial sums of the pair scores."""
    from miniworld_engine.kernels.smooth_lddt.triton.main import (
        smooth_lddt_fwd_portable,
        smooth_lddt_prep_portable,
    )

    x, gt, nuc, mask = inputs()
    tile, _cnt = smooth_lddt_prep_portable(gt, nuc, mask)
    smooth_lddt_fwd_portable(x, gt, nuc, mask, tile)


def smooth_lddt_bwd_triton() -> None:
    """smooth_lddt_bwd_kernel: the coordinate gradient."""
    from miniworld_engine.kernels.smooth_lddt.triton.main import (
        smooth_lddt_bwd_portable,
        smooth_lddt_prep_portable,
    )

    x, gt, nuc, mask = inputs()
    tile, _cnt = smooth_lddt_prep_portable(gt, nuc, mask)
    smooth_lddt_bwd_portable(x, gt, nuc, mask, tile, torch.ones(x.shape[0] * x.shape[1], device=dev()))
