"""Shared rigid augmentation with explicit random-number sources.

Quaternion, orthogonalized-vector and SciPy rotations retain their numerical
contracts. The sampling loop selects an augmentation callable without importing
any model package.
"""

from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Bool, Float
from scipy.spatial.transform import Rotation

from miniworld_engine.architecture.tensor_layout import expand_at_dim, rot_vec_mul


def random_rigid_motion(
    coords: Float[torch.Tensor, "*batch N 3"],
    mask: Bool[torch.Tensor, "*batch N"] | None = None,
    trans_scale: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "*batch N 3"]:
    """Center, randomly rotate and randomly translate each structure.

    Unlike :func:`miniworld_engine.architecture.utils.transform.random_augmentation` or
    :meth:`miniworld_engine.architecture.diffusion.Diffuser.random_rotation_and_translation`: both draw
    rotations through ``scipy.spatial.transform.Rotation.random``, which reads
    NumPy's global RNG and so cannot be pinned by a :class:`torch.Generator`.
    Rotations here come from normalised quaternions sampled with torch, so a
    seeded generator reproduces the whole sampling trajectory.

    Parameters
    ----------
    coords : Tensor
        Coordinates to re-pose.
    mask : Tensor or None
        Validity mask used for the center of mass.
    trans_scale : float
        Standard deviation of the random translation.
    generator : torch.Generator or None
        Source of randomness.

    Returns
    -------
    Tensor
        Re-posed coordinates.

    """
    batch = coords.shape[:-2]
    weights = (
        torch.ones_like(coords[..., :1])
        if mask is None
        else mask.to(coords.dtype).unsqueeze(-1)
    )
    total = weights.sum(dim=-2, keepdim=True).clamp(min=1.0)
    centered = coords - (coords * weights).sum(dim=-2, keepdim=True) / total

    quat = torch.randn(
        (*batch, 4), device=coords.device, dtype=coords.dtype, generator=generator
    )
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = quat.unbind(-1)
    rotation = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*batch, 3, 3)

    rotated = torch.einsum("...ij,...nj->...ni", rotation, centered)
    shift = trans_scale * torch.randn(
        (*batch, 1, 3), device=coords.device, dtype=coords.dtype, generator=generator
    )
    return rotated + shift


# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2024 ByteDance and/or its affiliates.
def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
    torch_generator: torch.Generator | None = None,
    numpy_rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Implements Algorithm 19 in AF3

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
    Returns:
        torch.Tensor:  the Augmentation version of input coords
            [..., N_sample, N_atom, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    if mask is None:
        x_input_coords = x_input_coords - torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1, keepdim=True) + eps
        )
        x_input_coords = x_input_coords - center.unsqueeze(dim=-2)

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        return x_input_coords

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment, numpy_rng=numpy_rng)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(
        size=(*batch_size_shape, N_sample, 3),
        device=device,
        generator=torch_generator,
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]

    if mask is not None:
        x_augment_coords = x_augment_coords * mask[..., None, :, None]
    return x_augment_coords


def uniform_random_rotation(
    N_sample: int = 1,
    numpy_rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation

    Args:
        N_sample (int, optional): the total number of augmentation. Defaults to 1.

    Returns:
        torch.Tensor: N_sample rot matrics
            [N_sample, 3, 3]
    """
    rotation = Rotation.random(num=N_sample, random_state=numpy_rng)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()  # [N_sample, 3, 3]
    return rot_matrix


# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
# This dense-layout augmentation is derived from AlphaFold 3, CC BY-NC-SA 4.0.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
def gram_schmidt_rotation(device, dtype, batch_shape=()):
    # Create a random rotation (Gram-Schmidt orthogonalization of two
    # random normal vectors)
    v0, v1 = torch.randn(size=(*batch_shape, 2, 3), dtype=dtype, device=device).unbind(
        -2
    )
    e0 = v0 / torch.clamp(torch.linalg.norm(v0, dim=-1, keepdim=True), min=1e-10)
    v1 = v1 - e0 * (v1 * e0).sum(dim=-1, keepdim=True)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1, dim=-1, keepdim=True), min=1e-10)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-2)


def masked_dense_rigid_motion(
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply random rigid augmentation.

    Args:
      positions: atom positions of shape (<common_axes>, 3)
      mask: per-atom mask of shape (<common_axes>,)

    Returns:
      Transformed positions with the same shape as input positions.
    """

    weights = mask[..., None]
    center = (weights * positions).sum(dim=(-2, -3), keepdim=True) / weights.sum(
        dim=(-2, -3), keepdim=True
    ).clamp(min=1e-6)
    batch_shape = positions.shape[:-3]
    rot = gram_schmidt_rotation(
        device=positions.device, dtype=positions.dtype, batch_shape=batch_shape
    )
    translation = torch.randn(
        size=(*batch_shape, 1, 1, 3), dtype=positions.dtype, device=positions.device
    )

    augmented_positions = (
        torch.einsum(
            "...nmi,...ij->...nmj",
            positions - center,
            rot,
        )
        + translation
    )
    return augmented_positions * mask[..., None]


@torch.no_grad()
def scipy_rigid_motion(
    x: Float[torch.Tensor, "... L 3"],
    mask: Float[torch.Tensor, "... L"] | None = None,
    translation_scale: float = 1.0,
) -> Float[torch.Tensor, "... L 3"]:
    """AF3 CentreRandomAugmentation (Alg. 19), applied every step (Alg. 18 l.3).

    Centre the coordinates to zero COM, then apply an independent uniform
    random rotation and a small random translation (s_trans = 1 Å) per sample.
    The centering removes the unconstrained global-translation mode (the
    rigid-aligned loss gives it no gradient, so without this the COM can drift
    at the low-noise end); the random rotation/translation averages over the
    model's non-equivariance, as in AF3.

    ``mask`` marks the atoms that actually exist. It matters: the atom axis is
    padded (to a bucket at training, to a stride multiple at inference) and the
    padding slots carry pure noise at sigma_0 = sigma_data * sigma_max = 2560 A.
    An unmasked mean therefore drags the centre away from the real structure in
    proportion to how much of the axis is padding -- a rounding error for the
    <=7 stride-padding atoms of FoldBench inference, but a large error for a
    crop that fills half its bucket. AF3 does the same thing
    (``utils.mask_mean`` in ``random_augmentation``) and likewise zeroes the
    padding slots afterwards.
    """
    if mask is not None:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        centre = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp(
            min=1e-6
        )
        x = x - centre
    else:
        x = x - x.mean(dim=-2, keepdim=True)
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-2], x_shape[-1])  # (n, L, 3)
    n = x.shape[0]
    rot = torch.from_numpy(Rotation.random(n).as_matrix()).to(x.device, x.dtype)
    trans = torch.randn(n, 1, 3, device=x.device, dtype=x.dtype) * translation_scale
    x = torch.bmm(x, rot.transpose(-1, -2)) + trans
    x = x.reshape(*x_shape)
    if mask is not None:
        x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
    return x
