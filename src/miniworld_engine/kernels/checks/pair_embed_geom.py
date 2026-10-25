"""Numerical checks for the ``pair_embed_geom`` family (portable kernels).

Two things are checked and kept apart. The BINS (distances, dihedrals and bond angles computed in registers, bucketized) are compared with the dense
reference builder's features (`reference.pair_embed_geom_features`): a bin flipped by a last-bit difference of a distance or an angle is allowed for a tiny
fraction of the pairs, reported as one number. Everything downstream of the bins (the gather of table rows, the bf16 rounding, the LayerNorm, the table gradient)
is compared with the dense product of the kernel's OWN bins, which is exact up to fp32 summation order and one bf16 rounding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels.checks import _fixed
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.pair_embed_geom import _B, _EPS, _L, ca_inputs, geom_inputs


def rows_from_codes(codes: torch.Tensor, kind: str) -> torch.Tensor:
    """Dense [B, n, n, 317 (geom) or 157 (ca)] fp32 one-hot (x the chain product) rebuilt from the kernel's own codes."""
    b, n = codes.shape[:2]
    w0, w1 = codes[..., 0].long() & 0xFFFFFFFF, codes[..., 1].long() & 0xFFFFFFFF
    cvf = codes[..., 2].contiguous().view(torch.float32)
    f = torch.zeros(b, n, n, 317 if kind == "geom" else 157, device=codes.device)
    live = (w0 & (1 << 31)) != 0
    f.scatter_add_(-1, (w0 & 127)[..., None], live[..., None].float())
    if kind == "geom":
        for base, sh in ((127, 8), (148, 13), (169, 18), (190, 23)):
            sel = live & ((w0 & (1 << 28)) != 0)
            f.scatter_add_(-1, (base + ((w0 >> sh) & 31))[..., None], sel[..., None].float())
        for k in range(5):
            sel = live & ((w0 & (1 << 29)) != 0)
            f.scatter_add_(-1, (211 + 21 * k + ((w1 >> (5 * k)) & 31))[..., None], sel[..., None].float())
        f[..., 316] = cvf * live.float()
    else:
        f.scatter_add_(-1, (127 + ((w0 >> 8) & 31))[..., None], live[..., None].float())
    return f


def _dense(codes: torch.Tensor, tbl: torch.Tensor, kind: str) -> torch.Tensor:
    """The pair the kernel should have written, from its own bins: bf16-rounded fp32 sum of the table rows."""
    return (rows_from_codes(codes, kind) @ tbl.float()).to(torch.bfloat16).float()


def _flips(codes: torch.Tensor, args, kind: str) -> torch.Tensor:
    """Fraction of the pairs whose bins differ from the dense reference builder's, as a 0-d tensor."""
    from miniworld_engine.kernels.pair_embed_geom.reference import pair_embed_ca_features, pair_embed_geom_features

    if kind == "geom":
        _, bb, ca, cb, mask, idx, chains = args[:7]
        ref = pair_embed_geom_features(bb, ca.bool(), cb.bool(), mask.bool(), idx, chains)
    else:
        _, cax, mask, idx = args[:4]
        ref = pair_embed_ca_features(cax, mask.bool(), idx)
    own = rows_from_codes(codes, kind)
    return (own != ref).any(-1).float().mean()


def _bins_checks(codes: torch.Tensor, args, kind: str) -> dict:
    live = (codes[..., 0] < 0)
    mask = args[4] if kind == "geom" else args[2]
    pm = (mask.bool()[:, :, None] & mask.bool()[:, None, :])
    out = {"live": (live.float(), pm.float()), "bin_flips": (_flips(codes, args, kind), torch.zeros(()))}
    if kind == "geom":
        chains = args[6].long()
        out["chain_product"] = (codes[..., 2].contiguous().view(torch.float32), (chains[:, :, None] * chains[:, None, :]).float() * pm)
    return out


def pair_embed_geom_bins_triton():
    """The 16-byte codes against the dense builder: the live pairs, the chain product, and the fraction of pairs with a bin that differs."""
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward

    _fixed()
    args = geom_inputs()
    return _bins_checks(portable_forward(*args, with_codes=True)[1], args, "geom")


def pair_embed_ca_bins_triton():
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward_ca

    _fixed()
    args = ca_inputs()
    return _bins_checks(portable_forward_ca(*args, with_codes=True)[1], args, "ca")


def pair_embed_geom_fwd_gather_triton():
    """The gather (both tables, with and without the LayerNorm) against the dense product of the kernel's own bins."""
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward, portable_forward_ca

    _fixed()
    out = {}
    for kind, args, fn in (("geom", geom_inputs(), portable_forward), ("ca", ca_inputs(), portable_forward_ca)):
        tbl = args[0]
        pair, codes = fn(*args, with_codes=True)
        ln = fn(*args, ln_eps=_EPS)
        hat = F.layer_norm(_dense(codes, tbl, kind), (256,), None, None, _EPS)
        live = (codes[..., 0] < 0)[..., None]
        out[f"pair_{kind}"] = (pair.float(), _dense(codes, tbl, kind))
        out[f"layernorm_{kind}"] = (ln.float(), torch.where(live, hat, torch.zeros_like(hat)))
    return out


def _grad(mode: int):
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward, portable_forward_ca

    args = geom_inputs() if mode == 0 else ca_inputs()
    _, codes = portable_forward(*args, with_codes=True) if mode == 0 else portable_forward_ca(*args, with_codes=True)
    g = torch.randn(_B, _L, _L, 256, device=dev(), dtype=BF16).to(torch.bfloat16)
    return args[0], codes, g


def pair_embed_geom_dw_triton():
    """The table gradient of both tables against the dense product ``onehot^T @ d pair`` of the kernel's own bins (the bf16 result of the whole backward)."""
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_backward

    _fixed()
    out = {}
    for mode, kind in ((0, "geom"), (1, "ca")):
        _, codes, g = _grad(mode)
        exact = torch.einsum("bijr,bijc->rc", rows_from_codes(codes, kind), g.float())
        out[f"dtbl_{kind}"] = (portable_backward(g, codes, mode).float(), exact)
    return out


def pair_embed_geom_bwd_reduce_triton():
    """The slice reduction on its own: random partials summed in slice order, rounded to bf16 once."""
    import triton

    from miniworld_engine.kernels.pair_embed_geom.triton import main

    _fixed()
    part = torch.randn(main.NBLK, main.NT, 256, device=dev())
    out = torch.empty(main.NT, 256, device=dev(), dtype=torch.bfloat16)
    total = main.NT * 256
    main._geom_dw_reduce[lambda meta: (triton.cdiv(total, meta["BLOCK"]),)](part, out, total, NBLK=main.NBLK, shape_key=main.token_key(_L))
    return {"dtbl": (out.float(), part.sum(0).to(torch.bfloat16).float())}


def pair_embed_geom_bwd_ln_triton():
    """d pair from d LayerNorm(pair), both tables, against torch's layer_norm backward on the dense pair of the kernel's own bins."""
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_ln_backward

    _fixed()
    out = {}
    for mode, kind in ((0, "geom"), (1, "ca")):
        tbl, codes, g = _grad(mode)
        dense = rows_from_codes(codes, kind) @ tbl.float()
        live = (codes[..., 0] < 0)[..., None]
        x = dense.to(torch.bfloat16).float().requires_grad_()
        F.layer_norm(x, (256,), None, None, _EPS).backward(g.float() * live)
        out[f"dx_{kind}"] = (portable_ln_backward(tbl, codes, g, mode, _EPS).float(), torch.where(live, x.grad, torch.zeros_like(x.grad)))
    return out
