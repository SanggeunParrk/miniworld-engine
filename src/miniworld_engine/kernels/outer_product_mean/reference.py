"""The torch reference the outer-product-mean kernels are checked against: `OuterProductMean.forward`'s statements, written
out over explicit weights so a kernel can be compared in fp32 on the same inputs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def outer_product_mean_reference(msa: torch.Tensor, mask: torch.Tensor | None, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                                 w_left: torch.Tensor, w_right: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor,
                                 residual: torch.Tensor | None = None, *, eps: float = 1e-5,
                                 normalize_before_proj: bool = True) -> torch.Tensor:
    """msa [B, S, L, CM], mask [B, S, L] bool -> [B, L, L, CZ] (+ residual). Same order of operations as the module."""
    y = F.layer_norm(msa, (msa.shape[-1],), ln_weight, ln_bias, eps)
    left, right = y @ w_left.t(), y @ w_right.t()
    if mask is None:
        mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
    left = left * mask[..., None]
    right = right * mask[..., None]
    out = torch.einsum("bmid,bmje->bijde", left, right).flatten(-2)
    norm = torch.einsum("bmi,bmj->bij", mask.float(), mask.float()).clamp(min=1)[..., None]
    if normalize_before_proj:
        pair = (out / norm).to(left.dtype) @ w_out.t() + b_out
    else:
        pair = ((out @ w_out.t() + b_out) / norm).to(left.dtype)
    return residual + pair if residual is not None else pair
