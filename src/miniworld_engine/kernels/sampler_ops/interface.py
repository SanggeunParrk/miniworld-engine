"""H100 path of the sampler glue (``cuda/sampler_ops.cu``): a step's pre-denoiser part (random rigid motion + re-noising) and its
post-denoiser part (Kabsch alignment + Euler update) as three kernels each instead of ~100 small torch ops, with the churn arithmetic on
device scalars. fp32 coordinates ``[B, N, 3]``, one rigid motion per sample; ``mask`` ``[N]`` fp32."""

from __future__ import annotations

import torch


def serves(coords: torch.Tensor, mask: torch.Tensor) -> bool:
    """fp32 coordinates on an H100, ``[B, N, 3]`` with an ``[N]`` mask."""
    if not coords.is_cuda or coords.dtype is not torch.float32 or coords.ndim != 3 or coords.shape[-1] != 3:
        return False
    if mask.dtype is not torch.float32 or mask.shape != coords.shape[1:2]:
        return False
    return torch.cuda.get_device_capability(coords.device) == (9, 0)


def augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, *, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale,
                  variance_floor, clamp):
    """``(noisy, sc)``: ``coords`` centred (mask-weighted), rotated by the Gram-Schmidt frame of ``rv`` ``[B, 2, 3]``, translated by
    ``tr`` ``[B, 3]``, masked, plus ``magnitude * noise``; ``sc[0]`` is ``sigma_hat`` (a device scalar, ``sc[:1].squeeze()`` is a view)."""
    from miniworld_engine.kernels.sampler_ops.cuda import ext

    noisy, sc = ext().augment_noise(
        coords.contiguous(), mask, rv.contiguous(), tr.contiguous(), noise.contiguous(), sigma, sigma_next,
        int(window), float(gamma), float(gamma_min), float(sigma_min), float(sigma_max), float(noise_scale), float(variance_floor),
        bool(clamp))
    return noisy, sc


def align_euler(noisy, denoised, mask, sc, sigma_next, step_scale):
    """The Euler update from ``noisy`` aligned onto ``denoised`` (weights ``mask``), with ``sigma_hat = sc[0]``."""
    from miniworld_engine.kernels.sampler_ops.cuda import ext

    return ext().align_euler(noisy.contiguous(), denoised.contiguous(), mask, sc, sigma_next, float(step_scale))


def pool_atoms(x, idx, gather_mask, atom_mask):
    """The atom encoder's pooling in one pass: ``out[s, t] = sum_a m[t, a] relu(g[t, a] x[s, idx[t, a]]) / max(sum_a m[t, a], 1e-10)`` in bf16.

    ``x`` bf16 ``[S, Q, C]`` (atom features in the queries layout, flattened), ``idx`` / ``gather_mask`` the gather of the token-atoms layout
    and ``atom_mask`` its atom mask, all ``[T, A]``."""
    from miniworld_engine.kernels.sampler_ops.cuda import ext

    return ext().pool_atoms(x.contiguous(), idx.contiguous(), gather_mask.contiguous(), atom_mask.contiguous())


def pool_atoms_serves(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype is torch.bfloat16 and x.shape[-1] % 4 == 0 and torch.cuda.get_device_capability(x.device) == (9, 0)
