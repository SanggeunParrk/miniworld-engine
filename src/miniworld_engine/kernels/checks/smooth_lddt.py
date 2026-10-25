"""Numerical checks for the ``smooth_lddt`` family, against the definition written out densely in float64.

The reference is AF3 Algorithm 27 as ``smooth_lddt/reference.py`` states it (an [N, N] tensor per augment, exact distances), evaluated per
(augment, batch) sample in float64 so the kernels' fp32 error is what is measured. Inputs come from ``drivers/smooth_lddt.py``.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers.smooth_lddt import inputs

_BINS = (0.5, 1.0, 2.0, 4.0)


def _pairs(gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor):
    """(d_gt [B, N, N], keep [B, N, N]) in float64: the ground-truth pair set of the loss."""
    g = gt.double()
    d = torch.cdist(g, g, compute_mode="donot_use_mm_for_euclid_dist")
    cut = torch.where(nuc.bool(), 30.0, 15.0).double().unsqueeze(-1)
    m = mask.bool()
    n = gt.shape[1]
    keep = (d < cut) & m.unsqueeze(-1) & m.unsqueeze(-2) & ~torch.eye(n, dtype=torch.bool, device=gt.device)
    return d, keep


def _sums(x: torch.Tensor, d_gt: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Sum of the pair scores per sample, [A, B] float64."""
    xd = x.double()
    d = torch.cdist(xd, xd, compute_mode="donot_use_mm_for_euclid_dist")
    diff = (d - d_gt).abs()
    score = sum(torch.sigmoid(t - diff) for t in _BINS) / len(_BINS)
    return (score * keep).sum(dim=(-1, -2))


def _tiles(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from miniworld_engine.kernels.smooth_lddt.triton.main import BLK

    b, n, _ = keep.shape
    nb = n // BLK
    um = keep | keep.transpose(-1, -2)
    f = keep.reshape(b, nb, BLK, nb, BLK)
    u = um.reshape(b, nb, BLK, nb, BLK)
    flag = f.any(dim=(2, 4)).to(torch.int8) | (u.any(dim=(2, 4)).to(torch.int8) << 1)
    return flag, f.sum(dim=(2, 4)).to(torch.int32)


def smooth_lddt_prep_triton():
    """Tile flags and per-tile pair counts, exactly (integers)."""
    from miniworld_engine.kernels.smooth_lddt.triton.main import (
        smooth_lddt_prep_portable,
    )

    _x, gt, nuc, mask = inputs()
    flag, cnt = _tiles(_pairs(gt, nuc, mask)[1])
    got_flag, got_cnt = smooth_lddt_prep_portable(gt, nuc, mask)
    return {"flags": (got_flag, flag), "counts": (got_cnt, cnt)}


def smooth_lddt_fwd_triton():
    """Per-sample sum of the pair scores (the 64-row partials added up)."""
    from miniworld_engine.kernels.smooth_lddt.triton.main import (
        smooth_lddt_fwd_portable,
        smooth_lddt_prep_portable,
    )

    x, gt, nuc, mask = inputs()
    d_gt, keep = _pairs(gt, nuc, mask)
    tile, _cnt = smooth_lddt_prep_portable(gt, nuc, mask)
    got = smooth_lddt_fwd_portable(x, gt, nuc, mask, tile).sum(dim=1).view(x.shape[0], x.shape[1])
    return {"sums": (got, _sums(x, d_gt, keep))}


def smooth_lddt_bwd_triton():
    """d(sum of the pair scores)/dx for an upstream gradient that differs per sample, against autograd over the float64 definition."""
    from miniworld_engine.kernels.smooth_lddt.triton.main import (
        smooth_lddt_bwd_portable,
        smooth_lddt_prep_portable,
    )

    x, gt, nuc, mask = inputs()
    d_gt, keep = _pairs(gt, nuc, mask)
    tile, _cnt = smooth_lddt_prep_portable(gt, nuc, mask)
    go = torch.rand(x.shape[0], x.shape[1], device=x.device) + 0.5
    got = smooth_lddt_bwd_portable(x, gt, nuc, mask, tile, go.reshape(-1))
    xr = x.double().requires_grad_()
    (_sums(xr, d_gt, keep) * go.double()).sum().backward()
    return {"dx": (got, xr.grad)}
