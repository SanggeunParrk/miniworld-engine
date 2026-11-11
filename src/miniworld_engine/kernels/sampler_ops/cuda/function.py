"""The CUDA backend of ``AugmentNoise`` / ``AlignEuler`` / ``PoolAtoms``: the launches of the ``sampler_ops_sm90a`` extension (``sampler_ops.cu``) as
opaque ops, so a compiled sampler launches them instead of breaking the graph. Inputs are the modules' (``modules/sampler_ops.py``): fp32 coordinates
``[B, N, 3]``, an fp32 ``mask`` ``[N]`` (one rigid motion per sample), bf16 atom features ``[S, Q, C]`` with ``C % 4 == 0`` for the pooling. Eval only
(no autograd) but for the pooling, which has a backward in ``x`` (``_PoolAtomsCuda``)."""

from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque


def _augment_noise_fake(
    coords, mask, rv, tr, noise, sigma, sigma_next, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp
):
    """[noisy (the shape of ``coords``), sc [4]], both fp32."""
    return [torch.empty_like(coords), torch.empty(4, dtype=coords.dtype, device=coords.device)]


@opaque(fake=_augment_noise_fake, name="sampler_ops_augment_noise_sm90")
def _augment_noise_op(
    coords: torch.Tensor,
    mask: torch.Tensor,
    rv: torch.Tensor,
    tr: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    window: int,
    gamma: float,
    gamma_min: float,
    sigma_min: float,
    sigma_max: float,
    noise_scale: float,
    variance_floor: float,
    clamp: bool,
) -> list[torch.Tensor]:
    """``[noisy, sc]`` of the sm_90a kernels (``sc[2:]`` is not written)."""
    from miniworld_engine.kernels.sampler_ops.cuda.sm80 import device_ext as ext

    noisy, sc = ext().augment_noise(
        coords, mask, rv, tr, noise, sigma, sigma_next, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp
    )
    return [noisy, sc]


def augment_noise_cuda(
    coords, mask, rv, tr, noise, sigma, sigma_next, *, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp
):
    """``(noisy [B, N, 3], sc [2])``: ``sc = (sigma_hat, noise magnitude)``, a view of the kernel's scratch (no host read)."""
    noisy, sc = _augment_noise_op(
        coords.contiguous(),
        mask.contiguous(),
        rv.contiguous(),
        tr.contiguous(),
        noise.contiguous(),
        sigma,
        sigma_next,
        int(window),
        float(gamma),
        float(gamma_min),
        float(sigma_min),
        float(sigma_max),
        float(noise_scale),
        float(variance_floor),
        bool(clamp),
    )
    return noisy, sc[:2]


def _align_euler_fake(noisy, denoised, mask, sc, sigma_next, step_scale):
    """The updated coordinates, the shape of ``noisy``."""
    return torch.empty_like(noisy)


@opaque(fake=_align_euler_fake, name="sampler_ops_align_euler_sm90")
def _align_euler_op(
    noisy: torch.Tensor, denoised: torch.Tensor, mask: torch.Tensor, sc: torch.Tensor, sigma_next: torch.Tensor, step_scale: float
) -> torch.Tensor:
    """The Euler update of the sm_90a kernels (``cuda/sampler_ops.cu``)."""
    from miniworld_engine.kernels.sampler_ops.cuda.sm80 import device_ext as ext

    return ext().align_euler(noisy, denoised, mask, sc, sigma_next, step_scale)


def align_euler_cuda(noisy, denoised, mask, sc, sigma_next, step_scale):
    """The Euler update from ``noisy`` Kabsch-aligned onto ``denoised`` (weights ``mask``), with ``sigma_hat = sc[0]``."""
    return _align_euler_op(noisy.contiguous(), denoised.contiguous(), mask.contiguous(), sc.contiguous(), sigma_next, float(step_scale))


def _pool_atoms_fake(x, idx, gather_mask, atom_mask):
    """[S, T, C] in the dtype of ``x`` (bf16)."""
    return torch.empty(x.shape[0], idx.shape[0], x.shape[2], dtype=x.dtype, device=x.device)


@opaque(fake=_pool_atoms_fake, name="sampler_ops_pool_atoms_sm90")
def _pool_atoms_op(x: torch.Tensor, idx: torch.Tensor, gather_mask: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """The encoder pooling of the sm_90a kernels (``cuda/sampler_ops.cu``)."""
    from miniworld_engine.kernels.sampler_ops.cuda.sm80 import device_ext as ext

    return ext().pool_atoms(x, idx, gather_mask, atom_mask)



def _pool_atoms_bwd_fake(gout, x, idx, gather_mask, atom_mask):
    """The gradient of ``x``: its shape and dtype (bf16)."""
    return torch.empty_like(x)


@opaque(fake=_pool_atoms_bwd_fake, name="sampler_ops_pool_atoms_bwd_sm90")
def _pool_atoms_bwd_op(gout: torch.Tensor, x: torch.Tensor, idx: torch.Tensor, gather_mask: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """d out / d x of the pooling (``cuda/sampler_ops.cu``): the slots' rows and weights (``pool_atoms_prep``), a stable sort of the slots by row,
    then one warp per (row, sample) adds its slots in slot order (fp32, no atomics)."""
    from miniworld_engine.kernels.sampler_ops.cuda.sm80 import device_ext as ext

    key, w = ext().pool_atoms_prep(idx, gather_mask, atom_mask, x.shape[1])
    skey, perm = torch.sort(key, stable=True)
    return ext().pool_atoms_bwd(gout, x, skey, perm, w, idx.shape[1])


class _PoolAtomsCuda(torch.autograd.Function):
    """The pooling with its backward (``x`` only: the index and the masks carry no gradient)."""

    @staticmethod
    def forward(ctx, x, idx, gather_mask, atom_mask):
        ctx.save_for_backward(x, idx, gather_mask, atom_mask)
        return _pool_atoms_op(x, idx, gather_mask, atom_mask)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gout):
        x, idx, gather_mask, atom_mask = ctx.saved_tensors
        return _pool_atoms_bwd_op(gout.contiguous(), x, idx, gather_mask, atom_mask), None, None, None


def pool_atoms_cuda(x, idx, gather_mask, atom_mask):
    """``out[s, t] = sum_a m[t, a] relu(g[t, a] x[s, idx[t, a]]) / max(sum_a m[t, a], 1e-10)`` in bf16 (``x`` ``[S, Q, C]``, the rest ``[T, A]``);
    differentiable in ``x`` when a gradient is recorded."""
    args = (x.contiguous(), idx.contiguous(), gather_mask.contiguous(), atom_mask.contiguous())
    if torch.is_grad_enabled() and x.requires_grad:
        return _PoolAtomsCuda.apply(*args)
    return _pool_atoms_op(*args)
