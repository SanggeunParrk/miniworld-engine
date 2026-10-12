import torch
from jaxtyping import Bool, Float
from scipy.spatial.transform import Rotation

from miniworld_engine.architecture.typing import typecheck


@typecheck
@torch.no_grad()
def random_augmentation(
    coord: Float[torch.Tensor, "*batch N 3"],
    mask: Bool[torch.Tensor, "... N"] | None = None,
    trans_scale: float = 1.0,
) -> Float[torch.Tensor, "*batch N 3"]:
    """Apply random rotation and translation to the input coordinates.

    Parameters
    ----------
    coord : Float[torch.Tensor, "*batch N 3"]
        Coordinates to be augmented.
    mask : Bool[torch.Tensor, "... N"], optional
        Mask indicating valid coordinates, by default None.
    trans_scale : float, optional
        Translation scale factor, by default 1.0.

    Returns
    -------
    Float[torch.Tensor, "*batch N 3"]
        Augmented coordinates.

    Notes
    -----
    Coordinates are first centered to their center of mass before applying
    augmentation.

    Exceptions
    ----------
    RuntimeError
        If broadcasting between coord and mask shapes fails.

    """
    batch_dims = coord.shape[:-2]
    if mask is None:
        mask = torch.ones(coord.shape[:-1], dtype=torch.bool, device=coord.device)
    else:
        try:
            torch.broadcast_shapes(coord.shape[:-1], mask.shape)
        except RuntimeError as e:
            msg = (
                f"Cannot broadcast coord shape {coord.shape} and mask shape "
                f"{mask.shape}: {e}"
            )
            raise RuntimeError(msg) from e
    mask_expanded = mask[..., None]

    rots = Rotation.random(batch_dims.numel()).as_matrix()
    rots = torch.from_numpy(rots).reshape(*batch_dims, 3, 3)
    rots = rots.to(device=coord.device, dtype=coord.dtype)
    trans = trans_scale * torch.randn(*batch_dims, 1, 3, device=coord.device)

    coord = coord.masked_fill(~mask_expanded, 0.0)
    center = coord.sum(dim=-2) / mask_expanded.sum(dim=-2).clamp(min=1)

    augmented_coord = coord - center.unsqueeze(-2)
    augmented_coord = torch.einsum("...ij,...nj->...ni", rots, augmented_coord)
    augmented_coord = augmented_coord + trans
    return augmented_coord * mask_expanded


@typecheck
@torch.no_grad()
def kabsch_align(
    mobile: Float[torch.Tensor, "*batch N 3"],
    target: Float[torch.Tensor, "... N 3"],
    mask: Bool[torch.Tensor, "... N"] | None = None,
) -> Float[torch.Tensor, "*batch N 3"]:
    """Align mobile structure to target using Kabsch algorithm.

    Parameters
    ----------
    mobile : Float[torch.Tensor, "*batch N 3"]
        Coordinates of the mobile structure.
    target : Float[torch.Tensor, "... N 3"]
        Coordinates of the target structure.
    mask : Bool[torch.Tensor, "... N"], optional
        Mask indicating valid points, by default None.

    Returns
    -------
    Float[torch.Tensor, "*batch N 3"]
        Aligned coordinates of the mobile structure.

    Exceptions
    ----------
    RuntimeError
        If broadcasting between mobile and target shapes fails.
        If mobile shape is not compatible with broadcasted shape.
        If broadcasting between mobile and mask shapes fails.

    """
    if mask is None:
        mask = torch.ones(mobile.shape[:-1], dtype=torch.bool, device=mobile.device)
    return weighted_kabsch_align(mobile, target, mask.float(), mask)


@typecheck
@torch.no_grad()
def weighted_kabsch_align(
    mobile: Float[torch.Tensor, "*batch N 3"],
    target: Float[torch.Tensor, "... N 3"],
    weights: Float[torch.Tensor, "... N"],
    mask: Bool[torch.Tensor, "... N"] | None = None,
) -> Float[torch.Tensor, "*batch N 3"]:
    """Align mobile structure to target using weighted Kabsch algorithm.

    Each point contributes to the alignment proportionally to its weight.

    Parameters
    ----------
    mobile : Float[torch.Tensor, "*batch N 3"]
        Coordinates of the mobile structure.
    target : Float[torch.Tensor, "... N 3"]
        Coordinates of the target structure.
    weights : Float[torch.Tensor, "... N"]
        Per-point weights for alignment.
    mask : Bool[torch.Tensor, "... N"], optional
        Mask indicating valid points, by default None.

    Returns
    -------
    Float[torch.Tensor, "*batch N 3"]
        Aligned coordinates of the mobile structure.

    Exceptions
    ----------
    RuntimeError
        If broadcasting between mobile and target shapes fails.
        If mobile shape is not compatible with broadcasted shape.
        If broadcasting between mobile and mask shapes fails.

    References
    ----------
    AlphaFold 3 Supplementary Information, Algorithm 28.

    """
    try:
        batch_dims = torch.broadcast_shapes(mobile.shape[:-2], target.shape[:-2])
    except RuntimeError as e:
        msg = (
            f"Cannot broadcast mobile shape {mobile.shape} and target shape "
            f"{target.shape}: {e}"
        )
        raise RuntimeError(msg) from e

    if mobile.shape[:-2] != batch_dims:
        msg = (
            f"Mobile shape {mobile.shape} must be equal to broadcasted shape "
            f"{batch_dims} + (N, 3) to align with target shape {target.shape}."
        )
        raise RuntimeError(msg)

    if mask is None:
        mask = torch.ones(mobile.shape[:-1], dtype=torch.bool, device=mobile.device)
    else:
        try:
            torch.broadcast_shapes(mobile.shape[:-1], mask.shape)
        except RuntimeError as e:
            msg = (
                f"Cannot broadcast mobile shape {mobile.shape} and mask shape "
                f"{mask.shape}: {e}"
            )
            raise RuntimeError(msg) from e

    weights = (weights * mask.float()).unsqueeze(-1)
    mask_expanded = mask[..., None]

    mobile = mobile.masked_fill(~mask_expanded, 0.0)
    mobile_center = (mobile * weights).sum(-2) / weights.sum(-2).clamp(min=1)
    mobile_centered = (mobile - mobile_center.unsqueeze(-2)) * mask_expanded

    target = target.masked_fill(~mask_expanded, 0.0)
    target_center = (target * weights).sum(-2) / weights.sum(-2).clamp(min=1)
    target_centered = (target - target_center.unsqueeze(-2)) * mask_expanded

    cov = torch.einsum("...ni,...nj->...ij", target_centered * weights, mobile_centered)
    u, _, vh = torch.linalg.svd(cov)
    rots = torch.einsum("...ij,...jk->...ik", u, vh)
    det = torch.linalg.det(rots)

    vh_corrected = vh.clone()
    vh_corrected[..., -1, :] = vh[..., -1, :] * det.unsqueeze(-1)
    rots = torch.einsum("...ij,...jk->...ik", u, vh_corrected)

    mobile_aligned = torch.einsum("...ij,...nj->...ni", rots, mobile_centered)
    mobile_aligned = mobile_aligned + target_center.unsqueeze(-2)
    return mobile_aligned * mask_expanded
