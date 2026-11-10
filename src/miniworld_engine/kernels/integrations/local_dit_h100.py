"""The AF3 atom transformer block on H100 (sm_90a): tensor-core row kernels on wgmma (the A operand from registers; bf16 or TF32) and
the mma.sync window attention, no tcgen05, cuBLAS only for the weight gradients.

The gate is ``AtomDiT._supports_cuda`` (modules/atom_dit.py). fp32 activations and parameters (no autocast)
run the TF32 kernels; operands are rounded to nearest tf32, the residual stream, LayerNorms, softmax and every stored activation stay fp32.

    inference  cond90 (hoisted) -> pre90 -> window attention -> post90
    training   the same forward saving its activations; tr_bwd90 -> post_bwd90 -> window attention backward -> pre_bwd90 (pre_bwd90_par:
               the parallel K / V AdaLN) -> cond_bwd90 -> finalize90. The slid window (AF3 "slide", IntelliFold2 "slide_qblock"): the clipped
               kernels for every window, the five edge windows recomputed in torch (forward) and differentiated in torch (backward).
"""

from __future__ import annotations

import contextlib

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._tf32 import tf32_matmul
from miniworld_engine.kernels.integrations import local_dit as _ld

DS, DC, DP, NH = 128, 128, 16, 4
_NW = 27   # the canonical leaves (local_dit.WEIGHTS + CROSS_WEIGHTS); a model's option leaves follow them (KQLN: wq bq wk bk)
F32 = torch.float32
BF = torch.bfloat16
_HOIST_COND: dict = {}
_HOIST_BIAS: dict = {}
_HOIST_EDGES: dict = {}  # (pair tensor, LN / bias weights) -> the slid edge windows' indices, key mask and pair bias


def _edge_tables(z, gamma, wb, n, kmask, n0, qblock=False, uid=None):
    """What the slid edge windows read that does not change between the steps of a fold: the five window indices, their key positions and
    validity, and their pair bias [5, h, 32, 128] (the pair LayerNorm and projection of those windows only). It depends on the real atom
    count, the pair and the block's LN / bias weights, so it is built once per pair (``_HOIST_EDGES``) and read at every step."""
    idx, keys, valid = _edge_index(z, n, kmask, n0, qblock)
    valid = _edge_valid(valid, idx, keys, kmask, uid)
    bias = _edge_bias(z[idx].float(), gamma.float(), wb.float())
    mask = ~valid[None, :, None, :, :]
    return idx, keys, mask, bias[None].contiguous()


def _edge_index(z, n, kmask, n0, qblock=False):
    """The five slid edge windows of ``_edge_tables``: their indices [5], key positions [5, 128] and key validity [5, 128]."""
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
    return idx, keys, keys < n_real  # a system under 128 real atoms has keys past them


def _edge_valid(valid, idx, keys, kmask, uid):
    """The edge windows' key validity [5, 1, 128] (``valid`` [5, 128]), or with the conformer mask [5, 32, 128] (``uid`` [n]: a real query
    sees only its conformer's keys, a padding query every valid key)."""
    if uid is None:
        return valid[:, None, :]
    rows = 32 * idx[:, None] + torch.arange(32, device=idx.device)                     # [5, 32]
    same = uid[rows][:, :, None] == uid[keys][:, None, :]
    if kmask is not None:
        same = same | ~kmask[rows][:, :, None]
    return valid[:, None, :] & same


def _edge_bias(ze, gamma, wb):
    """The pair bias [5, h, 32, 128] of the edge windows' pair rows ``ze`` [5, 32, 128, 16] (fp32): LN(z) gamma . Wb."""
    import torch.nn.functional as F

    zz = F.layer_norm(ze, (DP,), gamma, None, _ld.EPS)
    return F.linear(zz, wb).permute(0, 3, 1, 2)


def _slid_edges_bwd(do, dd, q, k, v, z, gamma, wb, n, kmask, n0, qblock, a, uid=None):
    """The backward of ``_slid_edges`` (training, one pair), before the window kernels. Those must not see the five edge windows (their
    forward output was replaced): their rows of ``do`` / ``dd`` are zeroed in place, so the kernels give them no dq / dk / dv / dbias.
    Their gradients are taken here in fp32 torch, the backward of the forward's math (a window chosen twice, small systems, counts once:
    the first choice) -> what ``_slid_edges_add`` adds once the kernels have written dp and dz."""
    h, dh = NH, DS // NH
    nwin = n // 32
    idx, keys, valid = _edge_index(z, n, kmask, n0, qblock)
    valid = _edge_valid(valid, idx, keys, kmask, uid)
    first = torch.ones(5, device=z.device, dtype=torch.bool)
    for j in range(1, 5):
        first[j] = (idx[j] != idx[:j]).all()
    g_out = do.view(a, nwin, 32, DS)[:, idx].float() * first[None, :, None, None]   # [a, 5, 32, 128]
    do.view(a, nwin, 32, DS).index_fill_(1, idx, 0)
    dd.view(a, NH, nwin, 32).index_fill_(2, idx, 0)
    kf = keys.reshape(-1)                                                        # [640]
    # written out (the opaque op runs below autograd): the window softmax backward, then the pair LayerNorm / projection backward
    qs = q.view(a, nwin, 32, h, dh)[:, idx].float()                              # [a, 5, 32, h, d]
    ks = k.view(a, n, h, dh)[:, kf].float().view(a, 5, DS, h, dh)                # [a, 5, 128, h, d]
    vs = v.view(a, n, h, dh)[:, kf].float().view(a, 5, DS, h, dh)
    gf, wf = gamma.float(), wb.float()
    ze = z[idx].float()                                                          # [5, 32, 128, 16]
    mu = ze.mean(-1, keepdim=True)
    rstd = torch.rsqrt(((ze - mu) ** 2).mean(-1, keepdim=True) + _ld.EPS)
    zn = (ze - mu) * rstd
    zg = zn * gf
    bias = torch.einsum("wqkc,hc->whqk", zg, wf)                                 # [5, h, 32, 128]
    scale = dh**-0.5
    logits = torch.einsum("awqhd,awkhd->awhqk", qs, ks) * scale + bias[None]
    logits = logits.masked_fill(~valid[None, :, None, :, :], float("-inf"))
    probs = torch.nan_to_num(torch.softmax(logits, dim=-1))                      # [a, 5, h, 32, 128]
    g = g_out.view(a, 5, 32, h, dh)
    dprob = torch.einsum("awqhd,awkhd->awhqk", g, vs)
    dvs = torch.einsum("awhqk,awqhd->awkhd", probs, g)
    ds = probs * (dprob - (probs * dprob).sum(-1, keepdim=True))                 # d logits (0 at masked keys)
    dqs = torch.einsum("awhqk,awkhd->awqhd", ds, ks) * scale
    dks = torch.einsum("awhqk,awqhd->awkhd", ds, qs) * scale
    dbias = ds.sum(0)                                                            # [5, h, 32, 128]
    dwb = torch.einsum("whqk,wqkc->hc", dbias, zg)
    dzg = torch.einsum("whqk,hc->wqkc", dbias, wf)
    dgamma = (dzg * zn).sum((0, 1, 2))
    dzn = dzg * gf
    dze = rstd * (dzn - dzn.mean(-1, keepdim=True) - zn * (dzn * zn).mean(-1, keepdim=True))
    return idx, kf, (dqs, dks, dvs, dze, dgamma, dwb)


def _slid_edges_add(edges, dp, dz):
    """Add ``_slid_edges_bwd``'s gradients: dq into ``dp[:, 128:256]`` (the edge windows' rows, zero from the kernels), dk / dv into
    ``dp[:, 256:512]`` at the slid key positions (each touched row summed once in fp32 and written back: no atomics, deterministic),
    d z into ``dz`` at the edge windows (zero from the kernels). -> dz, the extra d gamma_pair and d Wb."""
    idx, kf, (dq, dk, dv, dze, dg, dwb) = edges
    a, n = dq.shape[0], dp.shape[0] // dq.shape[0]
    nwin = n // 32
    dp[:, DS : 2 * DS].view(a, nwin, 32, DS).index_add_(1, idx, dq.reshape(a, 5, 32, DS).to(dp.dtype))
    same = (kf[:, None] == kf[None, :]).float()                                  # [640, 640]: rows of one key position
    for col, g in ((2 * DS, dk), (3 * DS, dv)):
        tgt = dp[:, col : col + DS].view(a, n, DS)
        with _fp32_exact():   # 0 / 1 weights: an exact fp32 sum
            tot = tgt[:, kf].float() + torch.einsum("ij,ajc->aic", same, g.reshape(a, 5 * DS, DS))
        tgt.index_copy_(1, kf, tot.to(dp.dtype))   # a repeated position writes the same total
    dz.index_add_(0, idx, dze.to(dz.dtype))
    return dz, dg, dwb


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


def _fwd_fake(single, cond, pair, kmask, weights, save, parallel_kv=0, slide=False, qblock=False, ident=False, pars=False, noout=False,
              kqln=False, kq_eps=1e-5, eps=1e-5,
              uid=None):
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
        t(0 if parallel_kv == 2 else m, DS),   # xkv (one AdaLN: x1, an empty slot)
        t(m, DS),
        t(a, NH, n, d=F32),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(m, DS),
        t(m if kqln else 0, 2 * DS),   # qkr: the raw q | k (KQLN)
    ]


@opaque(fake=_fwd_fake, name="local_dit_h100_fwd")
def _fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    kmask: torch.Tensor | None,
    weights: list[torch.Tensor],
    save: bool,
    parallel_kv: int = 0,
    slide: bool = False,
    qblock: bool = False,
    ident: bool = False,
    pars: bool = False,
    noout: bool = False,
    kqln: bool = False,
    kq_eps: float = 1e-5,
    eps: float = 1e-5,
    uid: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """``parallel_kv``: the K / V source (0 chained, 1 AF3 parallel, 2 one AdaLN); ``ident`` / ``pars`` / ``eps`` / ``uid``: the model options
    (``options``, ``run_block``); ``slide``: its slid key window (``qblock``: IntelliFold2's rule); the
    saved ``o`` holds the slid edge windows' output, ``lse`` the clipped kernel's (``_bwd`` takes those windows from torch).
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
    opts = (("IDENT",) if ident else ()) + (("PARS",) if pars else ()) + (("NOOUT",) if noout else ()) + (("KQLN",) if kqln else ())
    kq = torch.stack([w.float() for w in weights[_NW:]]).contiguous() if kqln else None   # [wq | bq | wk | bk] of the flat q / k LN
    weights = weights[:_NW]
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
            (tuple((w.data_ptr(), w._version) for w in weights), parallel_kv, ident, pars, eps)
            if hoist
            else None
        )

        def tables():
            return k90.cond_fwd(c2, p, tf32, options=opts)

        mod, mkv = (
            _ld._hoisted(_HOIST_COND, cond, (wkey, n), tables) if hoist else tables()
        )
        zs = [_ld._trunk(pair[g : g + 1], n).contiguous() for g in range(groups)]

        def pair_bias():
            biases = [local.pair_bias_mma(zg, p["gamma"], p["wbias"], kmask, n) for zg in zs]
            if uid is not None:   # Chai-1: a real query sees only the keys of its own conformer
                same = _same_conformer(uid, kmask, n)
                for b in biases:
                    b.masked_fill_(~same[None], float("-inf"))
            return biases[0] if groups == 1 else torch.stack(biases).contiguous()   # [G, 4, nwin, 32, 128]

        if hoist:
            mk = None if kmask is None else _ld._tkey(kmask)
            bias2 = _ld._hoisted(_HOIST_BIAS, pair, (wkey, n, mk, None if uid is None else _ld._tkey(uid)), pair_bias)
        else:
            bias2 = pair_bias()
        if save:
            q, sg, k, v, x1, xkv, *rest = k90.pre_fwd(
                s2, mod, mkv, p, tf32, save=True, eps=eps, nmod=nmod, parallel_kv=parallel_kv, options=opts, kq=kq, kq_eps=kq_eps
            )
            qkr = rest[0] if kqln else s2.new_empty(0, 2 * DS)
        else:
            q, sg, k, v = k90.pre_fwd(
                s2, mod, mkv, p, tf32, eps=eps, nmod=nmod, parallel_kv=parallel_kv, options=opts, kq=kq, kq_eps=kq_eps
            )
        o, lse = attn(q.view(a, n, DS), k.view(a, n, DS), v.view(a, n, DS), bias2, **({"groups": groups} if groups > 1 else {}))
        if slide:
            def edges():
                return [_edge_tables(zg, weights[_ld._GAMMA], weights[_ld._WBIAS], n, kmask, _n0, qblock, uid) for zg in zs]

            tables = _ld._hoisted(_HOIST_EDGES, pair, (wkey, n, mk, qblock, None if uid is None else _ld._tkey(uid)), edges) if hoist else edges()
            _slid_edges(o, q, k, v, tables, a, groups, n)
        o = o.view(m, DS)
        if not save:
            return [
                _ld._unpadded(
                    k90.post_fwd(s2, o, sg, mod, p, tf32, eps=eps, nmod=nmod, options=opts).view(a // groups, groups, n, DS),
                    single.shape,
                )
            ]
        out, u, a2, x2, t = k90.post_fwd(s2, o, sg, mod, p, tf32, save=True, eps=eps, nmod=nmod, options=opts)
        if parallel_kv == 2:   # one AdaLN: xkv is x1 (custom-op outputs may not alias: an empty slot, the backward reads x1)
            xkv = x1.new_empty(0, DS)
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
        qkr,
    ]


def _bwd_fake(dy, single, cond, pair, kmask, weights, saved, parallel_kv=0, slide=False, qblock=False, ident=False, pars=False, noout=False,
              kqln=False, kq_eps=1e-5, eps=1e-5,
              uid=None):
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
    parallel_kv: int = 0,
    slide: bool = False,
    qblock: bool = False,
    ident: bool = False,
    pars: bool = False,
    noout: bool = False,
    kqln: bool = False,
    kq_eps: float = 1e-5,
    eps: float = 1e-5,
    uid: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """d single, d cond, d pair, then one gradient per parameter (its dtype). The flags: the forward's (``_fwd``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm90_atom as k90
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    bias2, mod, mkv, q, k, v, sg, x1, xkv, o, lse, u, a2, x2, t, qkr = saved
    if parallel_kv == 2:   # one AdaLN for q and k / v: xkv is x1
        xkv = x1
    opts = (("IDENT",) if ident else ()) + (("PARS",) if pars else ()) + (("NOOUT",) if noout else ()) + (("KQLN",) if kqln else ())
    kq = torch.stack([w.float() for w in weights[_NW:]]).contiguous() if kqln else None   # [wq | bq | wk | bk] of the flat q / k LN
    weights, kq_leaves = weights[:_NW], weights[_NW:]
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
            dab, dyp, s2 if pars else a2, mod, u, sg, o, pb, dmod, dp, n, tf32, eps=eps, nmod=nmod, shared=sh, options=opts
        )
        if slide:  # the edge windows: their do / dd rows zeroed for the kernels, their gradients from torch
            edges = _slid_edges_bwd(do, dd, q, k, v, z, weights[_ld._GAMMA], weights[_ld._WBIAS], n, kmask, _n0, qblock, a, uid)
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
        dzk, dgamma_pair, dwb = local.pair_bias_bwd_mma(z, p["gamma"], p["wbias"], dbias)
        if slide:  # after the kernels wrote dp and dz: add the edge windows' dq / dk / dv / d z
            dz, dg_e, dwb_e = _slid_edges_add(edges, dp, dzk)
            dgamma_pair, dwb = dgamma_pair + dg_e, dwb + dwb_e
        else:
            dz = dzk
        if kqln:   # the flat q / k LayerNorm backward on dp's dq / dk (after the attention backward and the edge windows)
            cs_kq = k90.kqln_bwd(dp, qkr, kq, tf32, kq_eps)
        dx, dmkv, cs_sc1, cs_bs, cs_bq = k90.pre_bwd(
            dp, x1, s2, mod, mkv, da2, pb, p, dmod, tf32, eps=eps, nmod=nmod, shared=sh, parallel_kv=parallel_kv, options=opts
        )
        if shared:
            if (
                n % k90.ROWS == 0
            ):  # one kernel: partial sums, the six products, LN backward
                dc, nc, dmod, dmkv = k90.cond_bwd_shared(
                    dmod.view(-1, n, 768), dmkv.view(-1, n, 256), c2, pb, tf32, a, options=opts
                )
                c2g = c2.float()
            else:   # (unreached: n is a multiple of 128; the default block's conditioning only)
                dc, nc, dmod, dmkv, c2g = _cond_bwd_shared(
                    dmod, dmkv, c2, weights, a, n
                )
        else:
            dc, nc = k90.cond_bwd(dmod, dmkv, c2, pb, tf32, options=opts)
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
    kq_grads = [cs_kq[128 * i:128 * (i + 1)].to(w.dtype).clone() for i, w in enumerate(kq_leaves)] if kqln else []
    return [dsingle, dcond, dpair, *outs, *kq_grads]


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
    def forward(ctx, kmask, uid, flags, single, cond, pair, *weights):
        out, *saved = _fwd(single, cond, pair, kmask, list(weights), True, *flags, uid=uid)
        ctx.flags = flags
        ctx.save_for_backward(kmask, uid, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        kmask, uid, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _bwd(dy, single, cond, pair, kmask, list(weights), list(saved), *ctx.flags, uid=uid)
        return (None, None, None, *grads)


# ------------------------------------------------------------------------------------------------ the models' options
# ONE atom DiT implementation for every block-local atom model (AtomDiTProtenix / OpenFold3 and the variants moved onto it): a model's
# differences are options of these kernels, read off the block's modules (``options``), and its parameters are mapped onto the 27 canonical
# leaves (``_leaf``; a part the model lacks is a cached constant leaf):
#   K / V source       kvm 0 chained (Protenix), 1 parallel (AF3), 2 one AdaLN for q and k / v (Boltz-2, RF3, Chai-1: pre90 par 2,
#                      pre_bwd90_sh; the K / V AdaLN leaves are zeros)
#   identity AdaLN     IDENT (Chai-1): tables of the raw c, scale 1 + W c (ln_cond leaf 1, scale bias leaf 1)
#   parallel block     PARS (RF3, Chai-1): the transition's AdaLN normalises the block input
#   no gating query    gate weight leaf 0: sigmoid(0) = 0.5 exactly ...
#   no out projection  ... and out-projection leaf 2 I (Chai-1): u = o exactly (bf16 and TF32: products by 0.5 and 2), du -> do exactly
#   conformer mask     uid (Chai-1): the other-conformer keys -inf in the pair bias (and in the slid edge windows)
#   AdaLN eps          the activations' AdaLN epsilon (Chai-1: 0.1)
_CONST: dict = {}


def _const(kind: str, shape, device) -> torch.Tensor:
    key = (kind, tuple(shape), str(device))
    t = _CONST.get(key)
    if t is None:
        if kind == "two_eye":
            t = 2.0 * torch.eye(shape[0], dtype=F32, device=device)
        else:
            t = torch.full(tuple(shape), 1.0 if kind == "one" else 0.0, dtype=F32, device=device)
        _CONST[key] = t
    return t


def _identity_adaln(ada) -> bool:
    return not hasattr(ada, "ln_cond")   # IdentityAdaptiveLayerNorm


def options(module) -> tuple | None:
    """(kvm, IDENT, PARS, NOOUT, KQLN, its eps, AdaLN eps) of a block-local atom block, or None when the block is outside the kernels (mixed AdaLN forms or
    epsilons, a q / k norm, the up-gate transition, a gating query with no out projection)."""
    a, t = module.attention, module.transition
    ident = _identity_adaln(a.ada_ln_in)
    adalns = [a.ada_ln_in, t.ada_ln_in] + ([a.ada_ln_kv] if hasattr(a, "ada_ln_kv") else [])
    if any(_identity_adaln(x) != ident for x in adalns) or len({x.ln_in.eps for x in adalns}) != 1:
        return None
    if not ident and any(x.ln_cond.eps != _ld.EPS for x in adalns):
        return None
    if a.use_qk_norm or hasattr(t, "up"):
        return None   # the per-head q / k RMSNorm (no atom model), Boltz-2's up-gate (stage 3): still on kernels/dit_variants
    kqln = hasattr(a, "norm_query")   # RoseTTAFold3's affine LayerNorm of the flat q / k
    if kqln and (a.norm_query.eps != a.norm_key.eps or a.norm_query.bias is None or a.norm_key.bias is None):
        return None
    if hasattr(a, "to_gate") and isinstance(a.to_out, torch.nn.Identity):
        return None
    kvm = {"chained": 0, "parallel": 1, "shared": 2}[module.kv_norm]
    noout = not hasattr(a, "to_gate") and isinstance(a.to_out, torch.nn.Identity)
    return (kvm, ident, bool(module.parallel), noout, kqln, float(a.norm_query.eps) if kqln else 1e-5,
            float(a.ada_ln_in.ln_in.eps))


def _leaf(module, name):
    """The parameter ``name`` of WEIGHTS / CROSS_WEIGHTS, or the constant leaf of a part the block does not have."""
    a = module.attention
    dev = a.to_query.weight.device
    part, _, rest = name.partition(".")
    owner = getattr(module, part)
    if rest.startswith("ada_ln_kv."):
        if not hasattr(a, "ada_ln_kv"):   # one AdaLN for q and k / v: zero K / V AdaLN leaves (never read by the kernels)
            return _const("one" if rest == "ada_ln_kv.ln_cond.weight" else "zero",
                          (DC,) if rest.endswith("ln_cond.weight") else (DS,) if rest.endswith(".bias") else (DS, DC), dev)
        return module.get_parameter(name)
    if rest.startswith("ada_ln_in.") and _identity_adaln(owner.ada_ln_in):
        if rest == "ada_ln_in.ln_cond.weight":
            return _const("one", (DC,), dev)
        if rest == "ada_ln_in.to_scale.bias":
            return _const("one", (DS,), dev)
    if rest == "to_query.bias" and a.to_query.bias is None:
        return _const("zero", (DS,), dev)
    if rest == "to_gate.weight" and not hasattr(a, "to_gate"):
        return _const("zero", (DS, DS), dev)
    if rest == "to_out.weight" and isinstance(a.to_out, torch.nn.Identity):
        return _const("two_eye", (DS, DS), dev)
    return module.get_parameter(name)


def _weights_of(module):
    """The 27 canonical leaves, then the option leaves (the flat q / k LayerNorm's wq, bq, wk, bk)."""
    names = (*_ld.WEIGHTS, *_ld.CROSS_WEIGHTS)
    out = [_leaf(module, n) for n in names]
    a = module.attention
    if hasattr(a, "norm_query") and not a.use_qk_norm:
        out += [a.norm_query.weight, a.norm_query.bias, a.norm_key.weight, a.norm_key.bias]
    return out


def _same_conformer(uid, kmask, n):
    """[nwin, 32, 128] bool: key j of query window w's clipped window (position 32 w - 48 + j) is of the query's conformer, or the query is
    a padding atom (it keeps every valid key). ``uid`` [n] (padded with -1)."""
    nwin = n // 32
    idx = (32 * torch.arange(nwin, device=uid.device) - 48)[:, None] + torch.arange(128, device=uid.device)
    same = uid.view(nwin, 32, 1) == uid[idx.clamp(0, n - 1)][:, None, :]
    if kmask is not None:
        same = same | ~kmask.view(nwin, 32, 1)
    return same


def run_block(module, single, cond, pair, mask=None, uid=None):
    """The block's output stream on the H100 warp-level kernels; ``AtomDiT._supports_cuda`` must have accepted the call. ``uid`` [1, N]:
    the conformer ids (Chai-1)."""
    weights = _weights_of(module)
    n0 = single.shape[2]
    n = _ld._padded_length(n0)
    kmask = _ld._keys(mask, n0, n)
    kvm, ident, pars, noout, kqln, kq_eps, eps = options(module)
    u = None
    if uid is not None:
        u = uid[0] if n == n0 else torch.nn.functional.pad(uid[0], (0, n - n0), value=-1)
        u = u.contiguous()
    flags = (kvm, module.key_window in ("slide", "slide_qblock"), module.key_window == "slide_qblock", ident, pars, noout, kqln, kq_eps,
             eps)
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (single, cond, pair, *weights)
    ):
        return _H100Block.apply(kmask, u, flags, single, cond, pair, *weights)
    return _fwd(single, cond, pair, kmask, weights, False, *flags, uid=u)[0]
