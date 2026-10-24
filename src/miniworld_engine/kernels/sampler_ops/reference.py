"""PyTorch reference of the sampler glue: ``interface`` is this, fused (fp32 coordinates ``[B, N, 3]``, one rigid motion per sample)."""

from __future__ import annotations

import torch


def churn(sigma, sigma_next, *, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp):
    """``(sigma_hat, noise magnitude)`` of one churn step; device scalars in, device scalars out (no host read)."""
    one = torch.ones((), dtype=sigma.dtype, device=sigma.device)
    on = (sigma >= sigma_min) & (sigma <= sigma_max) if window else sigma_next > gamma_min
    sigma_hat = sigma * torch.where(on, one * (1 + gamma), one)
    variance = sigma_hat**2 - sigma**2
    if clamp:
        variance = variance.clamp(min=variance_floor)
    return sigma_hat, noise_scale * torch.sqrt(variance)


def augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, **cfg):
    """Centre (mask-weighted), rotate by the Gram-Schmidt frame of ``rv`` ``[B, 2, 3]``, translate by ``tr`` ``[B, 3]``, mask, add
    ``magnitude * noise``. Returns ``(noisy, sc)`` with ``sc[0] = sigma_hat``."""
    sigma_hat, magnitude = churn(sigma, sigma_next, **cfg)
    w = mask[None, :, None]
    centre = (w * coords).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1e-6)
    v0, v1 = rv.unbind(-2)
    e0 = v0 / torch.clamp(torch.linalg.norm(v0, dim=-1, keepdim=True), min=1e-10)
    v1 = v1 - e0 * (v1 * e0).sum(dim=-1, keepdim=True)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1, dim=-1, keepdim=True), min=1e-10)
    rot = torch.stack([e0, e1, torch.cross(e0, e1, dim=-1)], dim=-2)
    aug = (torch.einsum("bni,bij->bnj", coords - centre, rot) + tr[:, None, :]) * w
    return aug + magnitude * noise, torch.stack([sigma_hat, magnitude])


def align_euler(noisy, denoised, mask, sc, sigma_next, step_scale):
    """Kabsch-align ``noisy`` onto ``denoised`` (weights ``mask``), then the Euler update with ``sigma_hat = sc[0]``."""
    from miniworld_engine.kernels.sampler_ops.kabsch import kabsch_rotation

    w = mask[None, :, None]
    total = w.sum(1, keepdim=True)
    cm = (noisy * w).sum(1, keepdim=True) / total
    cr = (denoised * w).sum(1, keepdim=True) / total
    a, b = noisy - cm, denoised - cr
    cov = torch.einsum("bnc,bnd->bcd", a * w, b)
    aligned = a @ kabsch_rotation(cov) + cr
    sigma_hat = sc[0]
    direction = (aligned - denoised) / sigma_hat
    return aligned + step_scale * (sigma_next - sigma_hat) * direction


def pool_atoms(x, idx, gather_mask, atom_mask):
    """The gather (``x[:, idx]`` masked by ``gather_mask``), relu and masked mean over the atoms, as the encoder spells them."""
    g = x[:, idx.long()] * gather_mask.to(x.dtype)[None, :, :, None]
    m = atom_mask.to(torch.float32)[None, :, :, None]
    return ((m * torch.relu(g)).sum(-2) / torch.clamp(m.sum(-2), min=1e-10)).to(x.dtype)
