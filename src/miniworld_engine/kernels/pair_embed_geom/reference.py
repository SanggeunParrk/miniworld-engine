"""PyTorch reference of the backbone pair embedding (La-Proteina's autoencoder-encoder ``FeatureFactory`` in pair mode with
``rel_seq_sep, x1_bb_pair_dists_nm, x1_bb_pair_orientation, chain_idx_pair`` and no output LayerNorm): binned one-hots, a bias-free Linear
over their concatenation, masks. The ground truth of ``interface.pair_embed_geom``."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _one_hot(v, limits):
    return F.one_hot(torch.bucketize(v, limits), len(limits) + 1).float()


def _unit(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)


def _dihedral(a, b, c, d):
    b0, b1, b2 = b - a, c - b, d - c
    n1, n2 = _unit(torch.linalg.cross(b0, b1)), _unit(torch.linalg.cross(b1, b2))
    x = torch.linalg.cross(n1, n2, dim=-1)
    return torch.atan2(torch.sign(torch.linalg.vecdot(x, b1)) * torch.linalg.norm(x, dim=-1), torch.linalg.vecdot(n1, n2, dim=-1))


def _bond(a, b, c):
    b0, b1 = _unit(b - a), _unit(c - a)
    return torch.atan2(torch.linalg.norm(torch.linalg.cross(b0, b1, dim=-1), dim=-1), torch.linalg.vecdot(b0, b1, dim=-1))


def pair_embed_geom_features(bb, ca_mask, has_cb, mask, residue_idx, chains):
    """The 317 binned channels per pair, [B, n, n, 317] fp32, masked by the batch mask: sequence separation 127 | four distances 4 x 21 |
    five angles 5 x 21 | chain product 1."""
    dev = bb.device
    N, CA, Cc, CB = (bb[:, :, k] for k in range(4))
    pm = (ca_mask[:, :, None] & ca_mask[:, None, :]).float()
    sep = (residue_idx[:, :, None] - residue_idx[:, None, :]).float()
    feats = [_one_hot(sep, torch.linspace(-62.5, 62.5, 126, device=dev))]
    dl = torch.linspace(0.1, 2, 20, device=dev)
    dists = [torch.linalg.norm(CA[:, :, None] - X[:, None, :], dim=-1) for X in (N, CA, Cc, CB)]
    dists[3] = dists[3] * has_cb[:, None, :].float()
    feats += [_one_hot(d * pm, dl) for d in dists]
    p1 = lambda v: v[:, :, None]
    p2 = lambda v: v[:, None, :]
    ang = torch.stack([_dihedral(p1(N), p1(CA), p1(CB), p2(CB)), _dihedral(p2(N), p2(CA), p2(CB), p1(CB)), _bond(p1(CA), p1(CB), p2(CB)),
                       _bond(p2(CA), p2(CB), p1(CB)), _dihedral(p1(CA), p1(CB), p2(CB), p2(CA))], -1)
    feats_ang = _one_hot(ang, torch.linspace(-torch.pi, torch.pi, 20, device=dev)).flatten(-2)
    feats_dist = torch.cat(feats[1:], -1) * pm[..., None]
    feats_ang = feats_ang * (pm * has_cb[:, :, None].float())[..., None]
    chain = (chains[:, :, None] * chains[:, None, :]).float()[..., None]
    f = torch.cat([feats[0], feats_dist, feats_ang, chain], -1)
    return f * (mask[:, :, None] & mask[:, None, :]).float()[..., None]


def pair_embed_geom_reference(w, bb, ca_mask, has_cb, mask, residue_idx, chains):
    """w [256, >= 317]; bb [B, n, 4, 3] (N, CA, C, CB, nm); ca_mask / has_cb / mask [B, n] bool; residue_idx, chains [B, n] ints.
    Returns [B, n, n, 256] fp32 from the weights rounded to bf16 (what a bf16 autocast Linear multiplies by)."""
    f = pair_embed_geom_features(bb, ca_mask, has_cb, mask, residue_idx, chains)
    bm = (mask[:, :, None] & mask[:, None, :]).float()[..., None]
    return F.linear(f, w[:, :317].to(torch.bfloat16).float()) * bm


def pair_embed_ca_features(ca, mask, residue_idx=None):
    """The 157 binned channels of the decoder pair: sequence separation 127 | CA distance 30 (0.1 .. 3.0 nm, 29 limits), masked."""
    B, n = mask.shape
    dev = ca.device
    idx = residue_idx if residue_idx is not None else torch.arange(1, n + 1, device=dev).expand(B, n)
    sep = (idx[:, :, None] - idx[:, None, :]).float()
    d = torch.norm(ca[:, :, None, :] - ca[:, None, :, :], dim=-1)
    f = torch.cat([_one_hot(sep, torch.linspace(-62.5, 62.5, 126, device=dev)), _one_hot(d, torch.linspace(0.1, 3.0, 29, device=dev))], -1)
    return f * (mask[:, :, None] & mask[:, None, :]).float()[..., None]


def pair_embed_ca_reference(w, ca, mask, residue_idx=None):
    """w [256, 157]; ca [B, n, 3] nm; mask [B, n] bool. [B, n, n, 256] fp32 from the weights rounded to bf16."""
    f = pair_embed_ca_features(ca, mask, residue_idx)
    bm = (mask[:, :, None] & mask[:, None, :]).float()[..., None]
    return F.linear(f, w.to(torch.bfloat16).float()) * bm
