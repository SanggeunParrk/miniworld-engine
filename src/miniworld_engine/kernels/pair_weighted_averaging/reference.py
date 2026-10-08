"""The torch reference the pair-weighted-averaging kernels are checked against: `MSAPairWeightedAveraging.forward`'s statements,
written out over explicit weights and an explicit dropout keep-mask so a kernel can be compared in fp32 on the same inputs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pair_weighted_averaging_reference(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, ln_msa_weight: torch.Tensor,
                                      ln_msa_bias: torch.Tensor, w_value: torch.Tensor, w_gate: torch.Tensor,
                                      ln_pair_weight: torch.Tensor, w_bias: torch.Tensor,
                                      w_out: torch.Tensor, *, eps_msa: float = 1e-5, eps_pair: float = 1e-5,
                                      keep: torch.Tensor | None = None, p_drop: float = 0.0) -> torch.Tensor:
    """msa [B, S, L, D], pair [B, L, L, DZ], mask [B, L] bool, keep [B, L, D] -> msa + dropout(PWA). Same order as the module."""
    n_head = w_bias.shape[0]
    y = F.layer_norm(msa, (msa.shape[-1],), ln_msa_weight, ln_msa_bias, eps_msa)
    value = (y @ w_value.t()).unflatten(-1, (n_head, -1))
    bias = (F.layer_norm(pair, (pair.shape[-1],), ln_pair_weight, None, eps_pair) @ w_bias.t()).permute(0, 3, 1, 2)
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
    out = torch.einsum("bhij,bmjhd->bmihd", F.softmax(bias, dim=-1), value).flatten(-2)
    out = (torch.sigmoid(y @ w_gate.t()) * out) @ w_out.t()
    if keep is not None:
        out = out * keep[:, None].to(out.dtype) / (1.0 - p_drop)
    return msa + out
