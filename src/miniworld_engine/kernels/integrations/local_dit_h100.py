"""The AF3 atom transformer block on H100 (sm_90a): tensor-core row kernels on wgmma (the A operand from registers; bf16 or TF32) and
the mma.sync window attention, no tcgen05, cuBLAS only for the weight gradients.

The gate is ``AtomDiT._supports_cuda`` (modules/atom_dit.py). fp32 activations and parameters (no autocast)
run the TF32 kernels; operands are rounded to nearest tf32, the residual stream, LayerNorms, softmax and every stored activation stay fp32.

    inference  cond90 (hoisted) -> pre90 -> window attention -> post90
"""

from __future__ import annotations

import contextlib

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._tf32 import tf32_matmul
from miniworld_engine.kernels.integrations import local_dit as _ld

DS, DC, DP, NH = 128, 128, 16, 4
F32 = torch.float32
BF = torch.bfloat16
_HOIST_COND: dict = {}
_HOIST_BIAS: dict = {}
_HOIST_EDGES: dict = {}  # (pair tensor, LN / bias weights) -> the slid edge windows' indices, key mask and pair bias


def _edge_tables(z, gamma, wb, n, kmask, n0, qblock=False):
    """What the slid edge windows read that does not change between the steps of a fold: the five window indices, their key positions and
    validity, and their pair bias [5, h, 32, 128] (the pair LayerNorm and projection of those windows only). It depends on the real atom
    count, the pair and the block's LN / bias weights, so it is built once per pair (``_HOIST_EDGES``) and read at every step."""
    import torch.nn.functional as F

    nwin = n // 32
    n_real = (
        kmask.sum()
        if kmask is not None
        else torch.full((), n0, device=z.device, dtype=torch.int64)
    )
    # IntelliFold2's "slide_qblock": the slid rule against the real count rounded up to a whole query window
    # (``atom_dit._window_positions``); the keys past the real atoms stay masked (``valid``)
    n_rule = (n_real + 31) // 32 * 32 if qblock else n_real
    last = (n_real - 1) // 32
    idx = torch.stack([last * 0, last * 0 + 1, last - 2, last - 1, last]).clamp(0, nwin - 1)  # [5]
    starts = torch.minimum((32 * idx - 48).clamp(min=0), (n_rule - DS).clamp(min=0))
    keys = starts[:, None] + torch.arange(DS, device=z.device)  # [5, 128]
    valid = keys < n_real  # a system under 128 real atoms has keys past them
    zz = F.layer_norm(z[idx].float(), (DP,), gamma.float(), None, _ld.EPS)
    bias = F.linear(zz, wb.float()).permute(0, 3, 1, 2)  # [5, h, 32, 128]
    mask = ~valid[None, :, None, None, :]
    return idx, keys, mask, bias[None].contiguous()


def _slid_edges(o, q, k, v, tables, a, groups, n):
    """Overwrite the attention output ``o`` [A, N, 128] of the windows whose slid key window is not the clipped one with the attention
    over their slid key windows (torch, fp32). Those are windows 0 and 1 and the last three real windows (a window ends at
    ``32 w + 80``, so at most three end past the last real atom): five windows chosen on the device from the real atom count, so there
    is no host read and the call can be captured. A window chosen twice is computed twice with the same result.

    ``a`` samples laid out [a / G][G] over G pairs (G = 1: one pair): ONE set of torch ops over [a / G, G, ...]. The five windows, their key
    positions and validity depend on the real atom count only, which the G pairs of one input share (``tables[g][:3]`` are equal); only the
    pair bias differs, ``tables[g][3]`` stacked to [G, 5, h, 32, 128]. ``tables``: one ``_edge_tables`` per pair."""
    idx, keys, mask, _ = tables[0]
    bias = torch.cat([t[3] for t in tables], 0)
    ag = a // groups
    h, d = NH, DS // NH
    nwin = n // 32
    qs = q.view(ag, groups, nwin, 32, h, d)[:, :, idx].float()
    ks = k.view(ag, groups, n, h, d)[:, :, keys].float()
    vs = v.view(ag, groups, n, h, d)[:, :, keys].float()
    logits = torch.einsum("agwqhd,agwkhd->agwhqk", qs, ks) * d**-0.5 + bias[None]
    logits = logits.masked_fill(mask[None], float("-inf"))
    probs = torch.nan_to_num(torch.softmax(logits, dim=-1))
    out = torch.einsum("agwhqk,agwkhd->agwqhd", probs, vs).reshape(ag, groups, 5, 32, DS)
    o.view(ag, groups, nwin, 32, DS).index_copy_(2, idx, out.to(o.dtype))


# --------------------------------------------------------------------------------------------------------------------- the ops
def _shared(cond) -> bool:
    """One conditioning for every sample: an expanded view (stride 0 over the samples) of [1, 1, N, 128]. The modulation tables are made
    once per atom instead of once per atom and sample, and read modulo N."""
    return (
        cond.shape[0] > 1
        and cond.stride(0) == 0
        and cond.stride(2) == DC
        and cond.stride(3) == 1
    )


def _dims(single):
    """(samples, real atoms, padded atoms, rows): the samples are the A x G rows of ``single`` [A, G, N, d] (G distinct pairs, one per
    trunk_seed; G = 1 for every training call), laid out [A][G]."""
    a, g, n0, _ = single.shape
    n = _ld._padded_length(n0)
    return a * g, n0, n, a * g * n


def _fwd_fake(single, cond, pair, kmask, weights, save, parallel_kv=False, slide=False, qblock=False):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _n0, n, m = _dims(single)
    dt, dev = single.dtype, single.device
    t = lambda *shape, d=dt: torch.empty(*shape, device=dev, dtype=d)
    return [
        out,
        t(NH, n // 32, 32, 128, d=F32),
        t(n if _shared(cond) else m, 768),
        t(n if _shared(cond) else m, 256),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(a, NH, n, d=F32),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(m, DS),
    ]


@opaque(fake=_fwd_fake, name="local_dit_h100_fwd")
def _fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    kmask: torch.Tensor | None,
    weights: list[torch.Tensor],
    save: bool,
    parallel_kv: bool = False,
    slide: bool = False,
    qblock: bool = False,
) -> list[torch.Tensor]:
    """``parallel_kv``: AlphaFold 3's K/V AdaLN of the raw atoms; ``slide``: its slid key window (inference only).
    The block output [A, 1, N, 128] and, when ``save``, the tensors the backward reads (bias2, mod, mkv, q, k, v, sg, x1, xkv, o, lse, u, a2,
    x2, t): fresh tensors."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm90_atom as k90
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    tf32 = single.dtype == F32
    a, _n0, n, m = _dims(single)
    groups = single.shape[1]
    if groups > 1 and (tf32 or save):
        raise ValueError("G distinct pairs: bf16 inference only (the TF32 twin and the backward keep one bias)")
    attn = local.attn_fwd_mma32 if tf32 else local.attn_fwd_mma
    with torch.cuda.device(single.device):
        p = k90.pack(weights, tf32)
        s2 = _ld._padded(single, n).reshape(m, DS).contiguous()
        shared = _shared(cond)
        c2 = (
            _ld._padded(cond[:1], n).reshape(groups * n, DC)
            if shared
            else _ld._padded(cond, n).reshape(m, DC)
        ).contiguous()
        nmod = groups * n if shared else m   # shared: one conditioning per (group, atom), read modulo G N (rows are [A][G][N])
        hoist = (not save)
        wkey = (
            (tuple((w.data_ptr(), w._version) for w in weights), parallel_kv)
            if hoist
            else None
        )

        def tables():
            return k90.cond_fwd(c2, p, tf32)

        mod, mkv = (
            _ld._hoisted(_HOIST_COND, cond, (wkey, n), tables) if hoist else tables()
        )
        zs = [_ld._trunk(pair[g : g + 1], n).contiguous() for g in range(groups)]

        def pair_bias():
            biases = [local.pair_bias_mma(zg, p["gamma"], p["wbias"], kmask, n) for zg in zs]
            return biases[0] if groups == 1 else torch.stack(biases).contiguous()   # [G, 4, nwin, 32, 128]

        if hoist:
            mk = None if kmask is None else _ld._tkey(kmask)
            bias2 = _ld._hoisted(_HOIST_BIAS, pair, (wkey, n, mk), pair_bias)
        else:
            bias2 = pair_bias()
        if save:
            q, sg, k, v, x1, xkv = k90.pre_fwd(
                s2, mod, mkv, p, tf32, save=True, nmod=nmod
            )  # the training path is the chained K/V norm only (``_supports_cuda``)
        else:
            q, sg, k, v = k90.pre_fwd(
                s2, mod, mkv, p, tf32, nmod=nmod, parallel_kv=parallel_kv
            )
        o, lse = attn(q.view(a, n, DS), k.view(a, n, DS), v.view(a, n, DS), bias2, **({"groups": groups} if groups > 1 else {}))
        if slide:
            def edges():
                return [_edge_tables(zg, weights[_ld._GAMMA], weights[_ld._WBIAS], n, kmask, _n0, qblock) for zg in zs]

            tables = _ld._hoisted(_HOIST_EDGES, pair, (wkey, n, mk, qblock), edges) if hoist else edges()
            _slid_edges(o, q, k, v, tables, a, groups, n)
        o = o.view(m, DS)
        if not save:
            return [
                _ld._unpadded(
                    k90.post_fwd(s2, o, sg, mod, p, tf32, nmod=nmod).view(a // groups, groups, n, DS),
                    single.shape,
                )
            ]
        out, u, a2, x2, t = k90.post_fwd(s2, o, sg, mod, p, tf32, save=True, nmod=nmod)
    return [
        _ld._unpadded(out.view(a // groups, groups, n, DS), single.shape),
        bias2,
        mod,
        mkv,
        q,
        k,
        v,
        sg,
        x1,
        xkv,
        o,
        lse,
        u,
        a2,
        x2,
        t,
    ]


def _bwd_fake(dy, single, cond, pair, kmask, weights, saved):
    return [
        torch.empty_like(single),
        torch.empty_like(cond),
        torch.empty_like(pair),
        *(torch.empty_like(w) for w in weights),
    ]


@opaque(fake=_bwd_fake, name="local_dit_h100_bwd")
def _bwd(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    kmask: torch.Tensor | None,
    weights: list[torch.Tensor],
    saved: list[torch.Tensor],
) -> list[torch.Tensor]:
    """d single, d cond, d pair, then one gradient per parameter (its dtype)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm90_atom as k90
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    bias2, mod, mkv, q, k, v, sg, x1, xkv, o, lse, u, a2, x2, t = saved
    tf32 = single.dtype == F32
    a, _n0, n, m = _dims(single)
    dev = single.device
    with torch.cuda.device(dev):
        s2 = _ld._padded(single, n).reshape(m, DS).contiguous()
        shared = _shared(cond)
        c2 = (
            _ld._padded(cond[:1], n).reshape(n, DC)
            if shared
            else _ld._padded(cond, n).reshape(m, DC)
        ).contiguous()
        nmod = n if shared else m
        z = _ld._trunk(pair, n).contiguous()
        p, pb = k90.pack(weights, tf32), k90.pack_bwd(weights, tf32)
        dyp = _ld._padded(dy.to(s2.dtype).contiguous(), n).reshape(m, DS).contiguous()
        sh = (
            (4, n, 1) if (shared and a % 4 == 0 and n % 32 == 0) else None
        )  # grouped-sample reduction (4 samples per CTA group)
        dt, hh, dab, dmod, cs_st = k90.tr_bwd(
            dyp, t, mod, x2, p, pb, tf32, nmod=nmod, shared=sh
        )
        dp = torch.empty(m, 512, device=dev, dtype=s2.dtype)  # [dg | dq | dk | dv]
        da2, du, gated, do, dd, cs_so, cs_sc2 = k90.post_bwd(
            dab, dyp, a2, mod, u, sg, o, pb, dmod, dp, n, tf32, nmod=nmod, shared=sh
        )
        dbias = local.attn_bwd_mma(
            q.view(a, n, DS),
            k.view(a, n, DS),
            v.view(a, n, DS),
            do.view(a, n, DS),
            bias2,
            lse,
            dd,
            dp[:, DS:],
            tf32,
        )
        dz, dgamma_pair, dwb = local.pair_bias_bwd_mma(z, p["gamma"], p["wbias"], dbias)
        dx, dmkv, cs_sc1, cs_bs, cs_bq = k90.pre_bwd(
            dp, x1, s2, mod, mkv, da2, pb, p, dmod, tf32, nmod=nmod, shared=sh
        )
        if shared:
            if (
                n % k90.ROWS == 0
            ):  # one kernel: partial sums, the six products, LN backward
                dc, nc, dmod, dmkv = k90.cond_bwd_shared(
                    dmod.view(-1, n, 768), dmkv.view(-1, n, 256), c2, pb, tf32, a
                )
                c2g = c2.float()
            else:
                dc, nc, dmod, dmkv, c2g = _cond_bwd_shared(
                    dmod, dmkv, c2, weights, a, n
                )
        else:
            dc, nc = k90.cond_bwd(dmod, dmkv, c2, pb, tf32)
            c2g = c2
        mm = lambda x, y: _mmT(x, y, tf32)
        dwgq = mm(dp[:, 0:256], x1)  # [dg | dq]^T x1 -> Wg, Wq
        dwkv = mm(dp[:, 256:512], xkv)  # [dk | dv]^T xkv -> Wk, Wv
        dwo = mm(du, gated)
        dwab = mm(dab, x2)  # [Wa; Wb]
        dwd = mm(dt, hh)
        dc1 = mm(dmod[:, 0:256], nc)  # d W' of (Ws1', Wb1')
        dc2 = mm(dmod[:, 384:640], nc)  # (Ws2', Wb2')
        dckv = mm(dmkv, nc)  # (Wkp', Wks')
        dwso = mm(dmod[:, 256:384], c2g)
        dwst = mm(dmod[:, 640:768], c2g)
        outs = [torch.empty(w.shape, device=dev, dtype=w.dtype) for w in weights]
        k90.finalize(
            {
                "dwgq": dwgq,
                "dwkv": dwkv,
                "dwo": dwo,
                "dwab": dwab,
                "dwd": dwd,
                "dc1": dc1,
                "dc2": dc2,
                "dckv": dckv,
                "dwso": dwso,
                "dwst": dwst,
                "cs_sc1": cs_sc1,
                "cs_bq": cs_bq,
                "cs_so": cs_so,
                "cs_sc2": cs_sc2,
                "cs_st": cs_st,
                "cs_bs": cs_bs,
                "dgamma_pair": dgamma_pair,
                "dwb": dwb,
            },
            weights,
            outs,
            dev.index if dev.index is not None else torch.cuda.current_device(),
        )
    dsingle = _ld._unpadded(dx.view(a, 1, n, DS), single.shape)
    if shared:  # the gradient of the one shared row set, split evenly over the expanded view's A copies (autograd sums them back)
        dcond = _ld._unpadded(
            dc.view(1, 1, n, DC).to(cond.dtype).expand(a, 1, n, DC).contiguous(),
            cond.shape,
        )
    else:
        dcond = _ld._unpadded(dc.view(a, 1, n, DC), cond.shape)
    dpair = dz[: pair.shape[1]].reshape(pair.shape).to(pair.dtype)
    return [dsingle, dcond, dpair, *outs]


def _cond_bwd_shared(dmod, dmkv, c2, weights, a, n):
    """The conditioning backward on the N shared rows: the per-sample modulation gradients are summed over the samples (fp32), then one
    fp32 pass on [N, 128] rows (small: torch ops). -> dc / A (the A copies of the expanded input each get 1 / A of it), LN(c), the summed
    dmod / dmkv (fp32, the weight-gradient GEMM operands) and c in fp32."""
    f32 = torch.float32
    w = {nm: x.detach().float() for nm, x in zip(_NAMES, weights, strict=True)}
    ds = dmod.view(-1, n, 768).sum(0, dtype=f32)
    dk = dmkv.view(-1, n, 256).sum(0, dtype=f32)
    c = c2.float()
    mean = c.mean(-1, keepdim=True)
    var = ((c - mean) ** 2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + _ld.EPS)
    nrm = (c - mean) * rstd
    with _fp32_exact():
        d1 = ds[:, 0:128] @ w["sc1_w"] + ds[:, 128:256] @ w["bi1_w"]
        d2 = ds[:, 384:512] @ w["sc2_w"] + ds[:, 512:640] @ w["bi2_w"]
        dkv = dk[:, 0:128] @ w["kv_sc_w"] + dk[:, 128:256] @ w["kv_bi_w"]
        raw = ds[:, 256:384] @ w["so_w"] + ds[:, 640:768] @ w["st_w"]
    dn = d1 * w["g1"] + d2 * w["g2"] + dkv * w["gkv"]
    dc = (
        rstd
        * (dn - dn.mean(-1, keepdim=True) - nrm * (dn * nrm).mean(-1, keepdim=True))
        + raw
    )
    return dc / a, nrm, ds, dk, c


@contextlib.contextmanager
def _fp32_exact():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _mmT(a, b, tf32):
    """a^T b -> fp32 (a, b [M, *] views; TF32 tensor cores for the fp32 path, bf16 products with fp32 accumulation otherwise)."""
    if (
        a.dtype == F32 and not tf32
    ):  # the shared-conditioning path's fp32 operands under a bf16 block: exact fp32
        with _fp32_exact():
            return torch.mm(a.t(), b)
    if tf32:
        with tf32_matmul():
            return torch.mm(a.t(), b)
    return torch.mm(a.t(), b, out_dtype=torch.float32)


_NAMES = (
    "g1",
    "sc1_w",
    "sc1_b",
    "bi1_w",
    "wq",
    "bq",
    "wk",
    "wv",
    "ln_pair",
    "wb",
    "wg",
    "wo",
    "so_w",
    "so_b",
    "g2",
    "sc2_w",
    "sc2_b",
    "bi2_w",
    "wa",
    "wb2",
    "wd",
    "st_w",
    "st_b",
    "gkv",
    "kv_sc_w",
    "kv_sc_b",
    "kv_bi_w",
)
_IDX = {n: i for i, n in enumerate(_NAMES)}


class _H100Block(torch.autograd.Function):
    """The whole block: forward saving what the backward kernels read, backward on the same warp-level kernels."""

    @staticmethod
    def forward(ctx, kmask, single, cond, pair, *weights):
        out, *saved = _fwd(single, cond, pair, kmask, list(weights), True)
        ctx.save_for_backward(kmask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        kmask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _bwd(dy, single, cond, pair, kmask, list(weights), list(saved))
        return (None, *grads)


def _weights_of(module):
    names = (*_ld.WEIGHTS, *_ld.CROSS_WEIGHTS)
    return [module.get_parameter(n) for n in names]


def run_block(module, single, cond, pair, mask=None):
    """The block's output stream on the H100 warp-level kernels; ``AtomDiT._supports_cuda`` must have accepted the call."""
    weights = _weights_of(module)
    n0 = single.shape[2]
    kmask = _ld._keys(mask, n0, _ld._padded_length(n0))
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (single, cond, pair, *weights)
    ):
        return _H100Block.apply(kmask, single, cond, pair, *weights)
    parallel = module.kv_norm == "parallel"
    return _fwd(
        single, cond, pair, kmask, weights, False, parallel, module.key_window in ("slide", "slide_qblock"),
        module.key_window == "slide_qblock",
    )[0]
