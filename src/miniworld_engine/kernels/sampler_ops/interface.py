"""The sampler glue (``cuda/sampler_ops.cu``): a step's pre-denoiser part (random rigid motion + re-noising), its post-denoiser part (Kabsch alignment + Euler
update) and the atom encoder's pooling, as a few kernels instead of ~100 small torch ops, with the churn arithmetic on device scalars. fp32 coordinates
``[B, N, 3]``, one rigid motion per sample; ``mask`` ``[N]`` fp32. The random numbers are drawn by the caller.

Where a call runs (the card rule of ``docs/dispatch.md``, ``kernels/_pair_route.py``):

* H100: the CUDA kernels;
* any other card the engine holds a record for (every portable kernel of ``sampler_ops`` has an ``ok`` row in that card's manifest, at fp32 for the coordinate ops, bf16 for
  the pooling): the portable kernels (``triton/main.py``, same contract and algorithm, bitwise reproducible);
* any other card: one ``UnrecordedCardWarning`` and the PyTorch reference (``reference.py``); B200 has no CUDA kernel of this op and runs the reference silently;
* the internal A/B switch ``settings.force_portable_kernels`` runs the portable kernels on every card, H100 included; a CPU tensor is the reference.

:func:`serves` / :func:`pool_atoms_serves` say whether a caller should hand the call to this module at all (the dtype and layout are the kernels'; the card is any one but B200).
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._pair_route import CUDA, PORTABLE, restricted_reason, route

_CUDA_CAPS = ((9, 0),)


def _where(device: torch.device, dtype: str) -> str:
    return route("sampler_ops", device, cuda_caps=_CUDA_CAPS, dtype=dtype)


def _card_ok(device: torch.device) -> bool:
    """Is this a card the glue can be handed to: H100 (CUDA), the A/B switch, or a card that routes by its record (not B200, which runs the reference anyway)."""
    return device.type == "cuda" and restricted_reason(device, _CUDA_CAPS) is None


def serves(coords: torch.Tensor, mask: torch.Tensor) -> bool:
    """fp32 ``[B, N, 3]`` coordinates with an fp32 ``[N]`` mask on a card the glue routes (:func:`_card_ok`)."""
    if not coords.is_cuda or coords.dtype is not torch.float32 or coords.ndim != 3 or coords.shape[-1] != 3:
        return False
    if mask.dtype is not torch.float32 or mask.shape != coords.shape[1:2]:
        return False
    return _card_ok(coords.device)


def _augment_noise_portable_fake(coords, mask, rv, tr, noise, sigma, sigma_next, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp):
    """[noisy (the shape of ``coords``), sc [4]], both fp32."""
    return [torch.empty_like(coords), torch.empty(4, dtype=coords.dtype, device=coords.device)]


@opaque(fake=_augment_noise_portable_fake, name="sampler_ops_augment_noise")
def _augment_noise_portable(coords: torch.Tensor, mask: torch.Tensor, rv: torch.Tensor, tr: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor,
                            sigma_next: torch.Tensor, window: int, gamma: float, gamma_min: float, sigma_min: float, sigma_max: float, noise_scale: float,
                            variance_floor: float, clamp: bool) -> list[torch.Tensor]:
    """``[noisy, sc]`` of the portable kernels (``triton/main.py``)."""
    from miniworld_engine.kernels.sampler_ops.triton import portable_augment_noise

    noisy, sc = portable_augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, window=window, gamma=gamma, gamma_min=gamma_min, sigma_min=sigma_min,
                                       sigma_max=sigma_max, noise_scale=noise_scale, variance_floor=variance_floor, clamp=clamp)
    return [noisy, sc]


def augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, *, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale,
                  variance_floor, clamp):
    """``(noisy, sc)``: ``coords`` centred (mask-weighted), rotated by the Gram-Schmidt frame of ``rv`` ``[B, 2, 3]``, translated by
    ``tr`` ``[B, 3]``, masked, plus ``magnitude * noise``; ``sc[0]`` is ``sigma_hat`` (a device scalar, ``sc[:1].squeeze()`` is a view)."""
    where = _where(coords.device, "fp32")
    if where == CUDA:
        from miniworld_engine.kernels.sampler_ops.cuda import ext

        noisy, sc = ext().augment_noise(
            coords.contiguous(), mask, rv.contiguous(), tr.contiguous(), noise.contiguous(), sigma, sigma_next,
            int(window), float(gamma), float(gamma_min), float(sigma_min), float(sigma_max), float(noise_scale), float(variance_floor),
            bool(clamp))
        return noisy, sc
    cfg = dict(window=window, gamma=gamma, gamma_min=gamma_min, sigma_min=sigma_min, sigma_max=sigma_max, noise_scale=noise_scale,
               variance_floor=variance_floor, clamp=clamp)
    if where == PORTABLE:
        noisy, sc = _augment_noise_portable(coords.contiguous(), mask.contiguous(), rv.contiguous(), tr.contiguous(), noise.contiguous(), sigma, sigma_next,
                                            int(window), float(gamma), float(gamma_min), float(sigma_min), float(sigma_max), float(noise_scale),
                                            float(variance_floor), bool(clamp))
        return noisy, sc
    from miniworld_engine.kernels.sampler_ops import reference

    return reference.augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, **cfg)


def _align_euler_portable_fake(noisy, denoised, mask, sc, sigma_next, step_scale):
    """The updated coordinates, the shape of ``noisy``."""
    return torch.empty_like(noisy)


@opaque(fake=_align_euler_portable_fake, name="sampler_ops_align_euler")
def _align_euler_portable(noisy: torch.Tensor, denoised: torch.Tensor, mask: torch.Tensor, sc: torch.Tensor, sigma_next: torch.Tensor,
                          step_scale: float) -> torch.Tensor:
    """The Euler update of the portable kernels (``triton/main.py``)."""
    from miniworld_engine.kernels.sampler_ops.triton import portable_align_euler

    return portable_align_euler(noisy, denoised, mask, sc, sigma_next, step_scale)


def align_euler(noisy, denoised, mask, sc, sigma_next, step_scale):
    """The Euler update from ``noisy`` aligned onto ``denoised`` (weights ``mask``), with ``sigma_hat = sc[0]``."""
    where = _where(noisy.device, "fp32")
    if where == CUDA:
        from miniworld_engine.kernels.sampler_ops.cuda import ext

        return ext().align_euler(noisy.contiguous(), denoised.contiguous(), mask, sc, sigma_next, float(step_scale))
    if where == PORTABLE:
        return _align_euler_portable(noisy.contiguous(), denoised.contiguous(), mask.contiguous(), sc, sigma_next, float(step_scale))
    from miniworld_engine.kernels.sampler_ops import reference

    return reference.align_euler(noisy, denoised, mask, sc, sigma_next, step_scale)


def _pool_atoms_portable_fake(x, idx, gather_mask, atom_mask):
    """[S, T, C] in the dtype of ``x`` (bf16)."""
    return torch.empty(x.shape[0], idx.shape[0], x.shape[2], dtype=x.dtype, device=x.device)


@opaque(fake=_pool_atoms_portable_fake, name="sampler_ops_pool_atoms")
def _pool_atoms_portable(x: torch.Tensor, idx: torch.Tensor, gather_mask: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """The encoder pooling of the portable kernels (``triton/main.py``)."""
    from miniworld_engine.kernels.sampler_ops.triton import portable_pool_atoms

    return portable_pool_atoms(x, idx, gather_mask, atom_mask)


def pool_atoms(x, idx, gather_mask, atom_mask):
    """The atom encoder's pooling in one pass: ``out[s, t] = sum_a m[t, a] relu(g[t, a] x[s, idx[t, a]]) / max(sum_a m[t, a], 1e-10)`` in bf16.

    ``x`` bf16 ``[S, Q, C]`` (atom features in the queries layout, flattened), ``idx`` / ``gather_mask`` the gather of the token-atoms layout
    and ``atom_mask`` its atom mask, all ``[T, A]``."""
    where = _where(x.device, "bf16")
    if where == CUDA:
        from miniworld_engine.kernels.sampler_ops.cuda import ext

        return ext().pool_atoms(x.contiguous(), idx.contiguous(), gather_mask.contiguous(), atom_mask.contiguous())
    if where == PORTABLE:
        return _pool_atoms_portable(x.contiguous(), idx.contiguous(), gather_mask.contiguous(), atom_mask.contiguous())
    from miniworld_engine.kernels.sampler_ops import reference

    return reference.pool_atoms(x, idx, gather_mask, atom_mask)


def pool_atoms_serves(x: torch.Tensor) -> bool:
    """bf16 atom features ``[S, Q, C]`` on a card the pooling routes; the CUDA kernel (H100) needs ``C % 4 == 0``."""
    if not x.is_cuda or x.dtype is not torch.bfloat16 or not _card_ok(x.device):
        return False
    return x.shape[-1] % 4 == 0 or _where(x.device, "bf16") != CUDA
