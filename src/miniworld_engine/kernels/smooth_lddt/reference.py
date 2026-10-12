"""The definition: AlphaFold 3 Algorithm 27 written out densely (an [N, N] tensor per augment), the mean over augments and batch of
``1 - smooth_lddt``. ``gt`` is the same rigid structure for every augment (the augments are random rotations / translations of it)."""
from __future__ import annotations

import torch

BINS = (0.5, 1.0, 2.0, 4.0)
NUC_CUTOFF, OTHER_CUTOFF = 30.0, 15.0


def smooth_lddt_reference(x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x_pred [A, B, N, 3], gt [B, N, 3], is_nuc / mask [B, N] bool -> scalar ``mean_{a, b} (1 - lddt)``."""
    exact = "donot_use_mm_for_euclid_dist"   # the matmul form cancels |a|^2 + |b|^2 - 2ab: errors ~1e-4 A at structure scale
    d_pred = torch.cdist(x_pred, x_pred, compute_mode=exact)               # [A, B, N, N]
    d_gt = torch.cdist(gt, gt, compute_mode=exact)                         # [B, N, N]
    diff = (d_pred - d_gt).abs()
    score = sum(torch.sigmoid(t - diff) for t in BINS) / len(BINS)
    cutoff = torch.where(is_nuc, NUC_CUTOFF, OTHER_CUTOFF).unsqueeze(-1)   # radius of atom i (the row)
    n = gt.shape[-2]
    keep = (d_gt < cutoff) & mask.unsqueeze(-1) & mask.unsqueeze(-2) & ~torch.eye(n, dtype=torch.bool, device=gt.device)
    lddt = (score * keep).sum(dim=(-1, -2)) / keep.sum(dim=(-1, -2)).clamp(min=1)
    return (1.0 - lddt).mean()
