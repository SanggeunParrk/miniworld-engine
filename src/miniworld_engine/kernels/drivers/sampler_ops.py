"""Drivers for the ``sampler_ops`` family (portable kernels of the diffusion sampler's rigid-motion glue and the atom encoder's pooling).

The coordinates are fp32 ``[B, N, 3]`` (B samples of one structure of N atoms, the last quarter padding); the random draws are inputs, as for the CUDA op.
``pool_atoms`` reads bf16 atom features ``[S, Q, C]`` and the token-atoms layout ``[T, A]``.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import dev, driver_length, ragged

_B = 3
_N = ragged(driver_length(4096))
CHURN = dict(window=0, gamma=0.8, gamma_min=1.0, sigma_min=0.0, sigma_max=float("inf"), noise_scale=1.003, variance_floor=0.0, clamp=False)
_T, _A, _C = ragged(driver_length(256), by=5), 24, 128


def scene(b: int = _B, n: int = _N):
    """``(coords, mask, rv, tr, noise, sigma, sigma_next)``."""
    d = dev()
    coords = 10 * torch.randn(b, n, 3, device=d) + 3
    mask = (torch.arange(n, device=d) < (n * 3) // 4).float()
    return (coords, mask, torch.randn(b, 2, 3, device=d), torch.randn(b, 3, device=d), torch.randn(b, n, 3, device=d),
            torch.tensor(2.0, device=d), torch.tensor(1.5, device=d))


def align_scene(b: int = _B, n: int = _N):
    """``(noisy, denoised, mask, sc, sigma_next)``: a rotated, shifted, noised copy of the structure and the denoiser's answer."""
    coords, mask, *_ = scene(b, n)
    angle = torch.rand(b, device=dev()) * 6.0
    rot = torch.zeros(b, 3, 3, device=dev())
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = angle.cos(), -angle.sin(), angle.sin(), angle.cos(), 1.0
    noisy = coords @ rot + 5 + 0.5 * torch.randn_like(coords)
    denoised = coords + 0.1 * torch.randn_like(coords)
    return noisy.contiguous(), denoised.contiguous(), mask, torch.tensor([4.0, 0.3, 0.0, 0.0], device=dev()), torch.tensor(2.5, device=dev())


def pool_scene(s: int = _B, t: int = _T, a: int = _A, c: int = _C):
    """``(x, idx, gather_mask, atom_mask)`` with a token without atoms."""
    d = dev()
    rows = t * a + 64
    x = torch.randn(s, rows, c, device=d, dtype=torch.bfloat16)
    idx = torch.randint(0, rows, (t, a), device=d)
    atom_mask = torch.rand(t, a, device=d) > 0.3
    gather_mask = atom_mask | (torch.rand(t, a, device=d) > 0.9)
    atom_mask[5] = False
    return x, idx, gather_mask, atom_mask


def sampler_ops_stats_triton() -> None:
    from miniworld_engine.kernels.sampler_ops.triton import main

    noisy, denoised, mask, _sc, _sn = align_scene()
    main.portable_stats(noisy, denoised, mask)


def sampler_ops_augment_triton() -> None:
    from miniworld_engine.kernels.sampler_ops.triton import portable_augment_noise

    portable_augment_noise(*scene(), **CHURN)


def sampler_ops_cov_triton() -> None:
    _align()


def sampler_ops_align_triton() -> None:
    _align()


def _align() -> None:
    from miniworld_engine.kernels.sampler_ops.triton import portable_align_euler

    noisy, denoised, mask, sc, sn = align_scene()
    portable_align_euler(noisy, denoised, mask, sc, sn, 1.5)


def sampler_ops_pool_triton() -> None:
    from miniworld_engine.kernels.sampler_ops.triton import portable_pool_atoms

    x, idx, gm, am = pool_scene()
    portable_pool_atoms(x, idx, gm, am)
