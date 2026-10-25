"""Numerical checks for the ``pair_embed_adaln`` family (portable kernels).

The reference is the dense builder `reference.pair_embed_adaln_reference` (three binned one-hots, a Linear, both LayerNorms, gamma / beta), run on the kernel's own
bins where a bin flipped by a last-bit difference of a distance would otherwise move a whole pair: the forward is compared with a torch evaluation of the same
formula on the table rows the kernel's `bins` word names, and the number of bins that differ from the dense builder's is reported separately. The gradients are
compared with autograd through that same formula (the table gradient with the bf16 rounding of d(table sum) the kernels apply).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels.checks import _fixed
from miniworld_engine.kernels.drivers import dev
from miniworld_engine.kernels.drivers.pair_embed_adaln import _EPS, _NS, _NX, adaln_inputs


def _own(bins: torch.Tensor, nt: int, has_sc: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(one-hot [B, n, n, NT] fp32 of the kernel's bins, live [B, n, n])."""
    w = bins.long() & 0xFFFFFFFF
    live = (w & (1 << 31)) == 0
    f = torch.zeros(*bins.shape, nt, device=bins.device)
    f.scatter_add_(-1, (w & 255)[..., None], live[..., None].float())
    f.scatter_add_(-1, (_NS + ((w >> 8) & 255))[..., None], live[..., None].float())
    if has_sc:
        f.scatter_add_(-1, (_NS + _NX + ((w >> 16) & 255))[..., None], live[..., None].float())
    return f, live


def _formula(f, live, tbl, lnw, lnb, gamma, beta):
    h = F.layer_norm(f @ tbl, (256,), lnw, lnb, _EPS) * live[..., None]
    return (F.layer_norm(h, (256,), None, None, _EPS) * gamma[:, None, None] + beta[:, None, None]) * live[..., None]


def _flips(bins: torch.Tensor, args, has_sc: int) -> torch.Tensor:
    """Fraction of the live pairs whose distance bins differ from torch's ``bucketize`` of the same distances (0-d tensor)."""
    xt, sc, xlim, clim = args[5], args[6], args[10], args[11]
    dist = lambda x: torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
    w = bins.long() & 0xFFFFFFFF
    live = (w & (1 << 31)) == 0
    bad = ((w >> 8) & 255) != torch.bucketize(dist(xt), xlim)
    if has_sc:
        bad = bad | (((w >> 16) & 255) != torch.bucketize(dist(sc), clim))
    return (bad & live).float().sum() / live.float().sum().clamp_min(1)


def pair_embed_adaln_fwd_triton():
    from miniworld_engine.kernels.pair_embed_adaln.triton import portable_forward

    _fixed()
    out = {}
    for has_sc in (1, 0):
        args = adaln_inputs(has_sc)
        tbl, lnw, lnb, gamma, beta = args[:5]
        pair, bins = portable_forward(*args)
        f, live = _own(bins, tbl.shape[0], has_sc)
        out[f"pair_sc{has_sc}"] = (pair.float(), _formula(f, live, tbl, lnw, lnb, gamma, beta))
        out[f"bin_flips_sc{has_sc}"] = (_flips(bins, args, has_sc), torch.zeros(()))
    return out


def _backward_pairs(has_sc: int):
    from miniworld_engine.kernels.pair_embed_adaln.triton import portable_backward, portable_forward

    args = adaln_inputs(has_sc)
    tbl, lnw, lnb, gamma, beta = args[:5]
    pair, bins = portable_forward(*args)
    b, n = bins.shape[:2]
    g = torch.randn(b, n, n, 256, device=dev()).to(torch.bfloat16)
    got = portable_backward(tbl, lnw, lnb, gamma, bins, g, _NS, _NX, _EPS, _EPS)
    f, live = _own(bins, tbl.shape[0], has_sc)
    leaves = [t.clone().requires_grad_() for t in (tbl, lnw, lnb, gamma, beta)]
    _formula(f, live, *leaves).backward(g.float())
    return got, [t.grad for t in leaves]


def pair_embed_adaln_bwd_triton():
    """d(table sum) feeds everything: checked through the gradients of the LayerNorm affine and of gamma / beta, which the row kernel produces."""
    _fixed()
    got, want = _backward_pairs(1)
    return {name: (a, e) for name, a, e in zip(("dtbl", "dln_w", "dln_b", "dgamma", "dbeta"), got, want, strict=True) if name != "dtbl"}


def pair_embed_adaln_dw_triton():
    """The table gradient (tensor-core one-hot product of the bf16 d(table sum)), with and without self-conditioning."""
    _fixed()
    out = {}
    for has_sc in (1, 0):
        got, want = _backward_pairs(has_sc)
        out[f"dtbl_sc{has_sc}"] = (got[0], want[0])
    return out


def pair_embed_adaln_bwd_reduce_triton():
    """The fixed-order partial reductions on their own: random partials against torch's sums."""
    from miniworld_engine.kernels.pair_embed_adaln.triton import main

    _fixed()
    b, nrb = 3, 5
    part = torch.randn(b * nrb * 1024, device=dev())
    key = main.token_key(128)
    per_sample = main._reduce(part, b, nrb, 512, 1024, 0, key)
    ln = main._reduce(part, 1, b * nrb, 512, 1024, 512, key)
    p = part.view(b, nrb, 4, 256)
    return {"per_sample": (per_sample.view(b, 2, 256), p[:, :, :2].sum(1)), "ln": (ln.view(2, 256), p[:, :, 2:].sum((0, 1)))}
