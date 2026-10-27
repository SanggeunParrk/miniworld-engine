"""Numerical checks for the ``sampler_ops`` family (portable kernels) against the PyTorch reference (`sampler_ops.reference`)."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _fixed
from miniworld_engine.kernels.drivers.sampler_ops import CHURN, align_scene, pool_scene, scene


def sampler_ops_stats_triton():
    """The chunk partials summed: the mask weight and the mask-weighted coordinates of both tensors."""
    from miniworld_engine.kernels.sampler_ops.triton import main

    _fixed()
    noisy, denoised, mask, _sc, _sn = align_scene()
    parts = main.portable_stats(noisy, denoised, mask).sum(1)
    w = mask[None, :, None]
    want = torch.cat([w.sum(1).expand(noisy.shape[0], 1), (noisy * w).sum(1), (denoised * w).sum(1)], dim=1)
    return {"moments": (parts[:, :7], want)}


def sampler_ops_augment_triton():
    from miniworld_engine.kernels.sampler_ops import reference
    from miniworld_engine.kernels.sampler_ops.triton import portable_augment_noise

    _fixed()
    args = scene()
    got, sc = portable_augment_noise(*args, **CHURN)
    want, want_sc = reference.augment_noise(*args, **CHURN)
    return {"noisy": (got, want), "scalars": (sc[:2], want_sc)}


def sampler_ops_cov_triton():
    from miniworld_engine.kernels.sampler_ops.triton import main

    _fixed()
    noisy, denoised, mask, _sc, _sn = align_scene()
    cov = main.portable_cov(noisy, denoised, mask).sum(1)[:, :9].view(-1, 3, 3)
    w = mask[None, :, None]
    cm = (noisy * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)
    cr = (denoised * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)
    return {"cov": (cov, torch.einsum("bnc,bnd->bcd", (noisy - cm) * w, denoised - cr))}


def sampler_ops_align_triton():
    from miniworld_engine.kernels.sampler_ops import reference
    from miniworld_engine.kernels.sampler_ops.triton import portable_align_euler

    _fixed()
    args = align_scene()
    got = portable_align_euler(*args[:4], args[4], 1.5)
    return {"out": (got, reference.align_euler(*args[:4], args[4], 1.5))}


def sampler_ops_pool_triton():
    from miniworld_engine.kernels.sampler_ops import reference
    from miniworld_engine.kernels.sampler_ops.triton import portable_pool_atoms

    _fixed()
    x, idx, gm, am = pool_scene()
    got = portable_pool_atoms(x, idx, gm.float(), am.float())
    return {"out": (got.float(), reference.pool_atoms(x, idx, gm, am).float())}
