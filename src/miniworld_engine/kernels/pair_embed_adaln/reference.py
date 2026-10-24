"""PyTorch reference of the pair builder (``PairReprBuilder`` of La-Proteina's denoiser): three binned one-hots, a bias-free Linear
over their concatenation, an affine LayerNorm, then the adaptive LayerNorm with the per-sample ``gamma`` / ``beta`` that the
(pair-constant) time conditioning produces.  The ground truth of ``interface.pair_embed_adaln``."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bin_limits(lo: float, hi: float, dim: int, device) -> torch.Tensor:
    return torch.linspace(lo, hi, dim - 1, device=device)


def one_hot_bins(v: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.bucketize(v, limits), len(limits) + 1).float()


def pair_embed_adaln_reference(w, ln_weight, ln_bias, gamma, beta, xt, sc, residue_idx, mask, *, n_sep=127, n_xt=30, n_sc=30,
                               xt_range=(0.1, 3.0), sc_range=(0.1, 3.0), eps1=1e-5, eps2=1e-5):
    """w [256, >= n_sep + n_xt + n_sc]; gamma / beta [B, 256] (gamma already through the sigmoid); xt / sc [B, n, 3] (sc may be None:
    the self-conditioning one-hot is then all zeros); residue_idx [B, n]; mask [B, n] bool. Returns [B, n, n, 256] fp32."""
    dev = xt.device
    pm = (mask[:, :, None] & mask[:, None, :]).float()
    sep = residue_idx[:, :, None].float() - residue_idx[:, None, :].float()
    half = n_sep / 2.0 - 1
    feats = [one_hot_bins(sep, torch.linspace(-half, half, n_sep - 1, device=dev))]
    dist = lambda x: torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
    feats.append(one_hot_bins(dist(xt), bin_limits(*xt_range, n_xt, dev)))
    feats.append(one_hot_bins(dist(sc), bin_limits(*sc_range, n_sc, dev)) if sc is not None else torch.zeros_like(feats[1][..., :n_sc]))
    f = torch.cat(feats, -1) * pm[..., None]
    h = F.linear(f, w[:, : n_sep + n_xt + n_sc].float())
    h = F.layer_norm(h, (h.shape[-1],), ln_weight, ln_bias, eps1) * pm[..., None]
    y = F.layer_norm(h, (h.shape[-1],), None, None, eps2)
    return (y * gamma[:, None, None, :] + beta[:, None, None, :]) * pm[..., None]
