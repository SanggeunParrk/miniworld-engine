"""The AF3 atom transformer block on H100 (sm_90a): warp-level tensor-core row kernels (mma.sync, bf16 or TF32) and the mma.sync window
attention, no tcgen05 / wgmma, cuBLAS only for the weight gradients.

``matches`` is the whole gate; ``integrations.local_dit.serves`` / ``block`` ask it first. fp32 activations and parameters (no autocast)
run the TF32 kernels; operands are rounded to nearest tf32, the residual stream, LayerNorms, softmax and every stored activation stay fp32.

    inference  cond90 (hoisted) -> pre90 -> window attention -> post90
"""

from __future__ import annotations

import contextlib

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.integrations import local_dit as _ld
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._tf32 import tf32_matmul

DS, DC, DP, NH = 128, 128, 16, 4
F32 = torch.float32
BF = torch.bfloat16
_HOIST_COND: dict = {}
_HOIST_BIAS: dict = {}
_HOIST_EDGES: dict = {}  # (pair tensor, LN / bias weights) -> the slid edge windows' indices, key mask and pair bias


def _hopper(t) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (9, 0)


def matches(module, single, cond, pair, mask, rope=None) -> bool:
    """H100, cross-attention AF3 atom block at the atom widths, fp32 inference (no gradient recorded)."""
    if _ld._esm_variant(module):
        return _esm_matches(module, single, cond, pair, mask, rope)
    if not (single.is_cuda and _hopper(single)):
        return False
    if not settings.current().local_dit_h100_mma:
        return False
    if not module.cross_attention or (
        torch.is_autocast_enabled("cuda") and single.dtype != BF
    ):
        return False
    if _ld._af3_variant(module) and not _af3_servable(module, single, cond, pair, mask):
        return False
    a, tr = module.attention, module.transition
    if settings.current().force_portable_kernels:
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    if a._backend != KernelBackend.PORTABLE:
        return False
    params = _weights_of(module)
    if any(p.device != single.device or not p.is_contiguous() for p in params):
        return False
    if a.use_qk_norm and (a.norm_query.eps is not None or a.norm_key.eps is not None):
        return False   # the kernels take torch's default eps (the float32 one for bf16 and fp32 alike)
    dt = (single.dtype, cond.dtype, pair.dtype)
    if dt == (F32, F32, F32):
        if (
            any(p.dtype != F32 for p in params)
            or not settings.current().local_dit_h100_mma_fp32
        ):
            return False
    elif dt == (BF, BF, BF):
        if (
            any(p.dtype not in (F32, BF) for p in params)
            or not settings.current().local_dit_h100_mma_bf16
        ):
            return False
    else:
        return False
    if (
        single.ndim != 4
        or single.shape[1] != 1
        or single.shape[3] != DS
        or single.shape[2] == 0
    ):
        return False
    n = single.shape[2]
    nwin = (n + 31) // 32
    if tuple(cond.shape) != (*single.shape[:3], DC) or tuple(pair.shape) != (
        1,
        nwin,
        32,
        128,
        DP,
    ):
        return False
    if not (
        single.is_contiguous()
        and pair.is_contiguous()
        and (cond.is_contiguous() or _shared(cond))
    ):
        return False
    if mask is not None and (
        mask.dtype != torch.bool
        or tuple(mask.shape) != (1, n)
        or mask.device != single.device
    ):
        return False
    if (
        a.n_head,
        a.to_query.weight.shape[0],
        a.to_bias.weight.shape[1],
        tr.expand_a.weight.shape[0],
    ) != (NH, DS, DP, 2 * DS):
        return False
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        a.ln_pair,
        tr.ada_ln_in.ln_in,
        tr.ada_ln_in.ln_cond,
        a.ada_ln_kv.ln_in,
        a.ada_ln_kv.ln_cond,
    )
    return all(norm.eps == _ld.EPS for norm in norms)


def _af3_servable(module, single, cond, pair, mask) -> bool:
    """AlphaFold 3's K/V norm and slid key window are inference-only here (the backward kernels know neither). The slid window needs no
    host read of the real atom count: it is a device tensor, so the block also runs inside a CUDA graph capture."""
    params = [module.get_parameter(n) for n in (*_ld.WEIGHTS, *_ld.CROSS_WEIGHTS)]
    return not (
        torch.is_grad_enabled()
        and any(t.requires_grad for t in (single, cond, pair, *params))
    )


def _edge_tables(z, gamma, wb, n, kmask, n0):
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
    last = (n_real - 1) // 32
    idx = torch.stack([last * 0, last * 0 + 1, last - 2, last - 1, last]).clamp(0, nwin - 1)  # [5]
    starts = torch.minimum((32 * idx - 48).clamp(min=0), (n_real - DS).clamp(min=0))
    keys = starts[:, None] + torch.arange(DS, device=z.device)  # [5, 128]
    valid = keys < n_real  # a system under 128 real atoms has keys past them
    zz = F.layer_norm(z[idx].float(), (DP,), gamma.float(), None, _ld.EPS)
    bias = F.linear(zz, wb.float()).permute(0, 3, 1, 2)  # [5, h, 32, 128]
    mask = ~valid[None, :, None, None, :]
    return idx, keys, mask, bias[None].contiguous()


def _slid_edges(o, q, k, v, tables, a, n):
    """Overwrite the attention output ``o`` [A, N, 128] of the windows whose slid key window is not the clipped one with the attention
    over their slid key windows (torch, fp32). Those are windows 0 and 1 and the last three real windows (a window ends at
    ``32 w + 80``, so at most three end past the last real atom): five windows chosen on the device from the real atom count, so there
    is no host read and the call can be captured. A window chosen twice is computed twice with the same result. ``tables``: ``_edge_tables``."""
    idx, keys, mask, bias = tables
    h, d = NH, DS // NH
    nwin = n // 32
    qs = q.view(a, nwin, 32, h, d)[:, idx].float()
    ks = k.view(a, n, h, d)[:, keys].float()
    vs = v.view(a, n, h, d)[:, keys].float()
    logits = torch.einsum("awqhd,awkhd->awhqk", qs, ks) * d**-0.5 + bias
    logits = logits.masked_fill(mask, float("-inf"))
    probs = torch.nan_to_num(torch.softmax(logits, dim=-1))
    out = torch.einsum("awhqk,awkhd->awqhd", probs, vs).reshape(a, 5, 32, DS)
    o.view(a, nwin, 32, DS).index_copy_(1, idx, out.to(o.dtype))


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
    a, _, n0, _ = single.shape
    n = _ld._padded_length(n0)
    return a, n0, n, a * n


_NBASE = len(_ld.WEIGHTS) + len(_ld.CROSS_WEIGHTS)
_QK_EPS = 1.1920928955078125e-07   # torch.finfo(torch.float32).eps: what the engine's RMSNorm uses for bf16 and fp32 inputs alike


def _qk(weights) -> bool:
    return len(weights) > _NBASE


def _fwd_fake(single, cond, pair, kmask, weights, save, parallel_kv=False, slide=False):
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
        *((t(m, DS), t(m, DS)) if _qk(weights) else ()),
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
    attn = local.attn_fwd_mma32 if tf32 else local.attn_fwd_mma
    qkw = [w.float().contiguous() for w in weights[_NBASE:]]
    weights = weights[:_NBASE]
    with torch.cuda.device(single.device):
        p = k90.pack(weights, tf32)
        s2 = _ld._padded(single, n).reshape(m, DS).contiguous()
        shared = _shared(cond)
        c2 = (
            _ld._padded(cond[:1], n).reshape(n, DC)
            if shared
            else _ld._padded(cond, n).reshape(m, DC)
        ).contiguous()
        nmod = n if shared else m
        hoist = (not save) and settings.current().local_dit_hoist
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
        z = _ld._trunk(pair, n).contiguous()

        def pair_bias():
            return local.pair_bias_mma(z, p["gamma"], p["wbias"], kmask, n)

        if hoist:
            mk = None if kmask is None else _ld._tkey(kmask)
            bias2 = _ld._hoisted(_HOIST_BIAS, pair, (wkey, n, mk), pair_bias)
        else:
            bias2 = pair_bias()
        if save:
            q, sg, k, v, x1, xkv = k90.pre_fwd(
                s2, mod, mkv, p, tf32, save=True, nmod=nmod
            )  # the training path is the chained K/V norm only (``matches``)
        else:
            q, sg, k, v = k90.pre_fwd(
                s2, mod, mkv, p, tf32, nmod=nmod, parallel_kv=parallel_kv
            )
        qraw = kraw = None
        if qkw:   # the attention reads the normalised q / k; the backward keeps the raw ones
            qraw, kraw = q, k
            q, k = torch.empty_like(qraw), torch.empty_like(kraw)
            _qk_kernels().qknorm_fwd(qraw, kraw, qkw[0], qkw[1], _QK_EPS, _QK_EPS, q, k)
            if not save:
                del qraw, kraw
        o, lse = attn(q.view(a, n, DS), k.view(a, n, DS), v.view(a, n, DS), bias2)
        if slide:
            def edges():
                return _edge_tables(z, weights[_ld._GAMMA], weights[_ld._WBIAS], n, kmask, _n0)

            _slid_edges(
                o,
                q,
                k,
                v,
                _ld._hoisted(_HOIST_EDGES, pair, (wkey, n, mk), edges) if hoist else edges(),
                a,
                n,
            )
        o = o.view(m, DS)
        if not save:
            return [
                _ld._unpadded(
                    k90.post_fwd(s2, o, sg, mod, p, tf32, nmod=nmod).view(a, 1, n, DS),
                    single.shape,
                )
            ]
        out, u, a2, x2, t = k90.post_fwd(s2, o, sg, mod, p, tf32, save=True, nmod=nmod)
    return [
        _ld._unpadded(out.view(a, 1, n, DS), single.shape),
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
        *((qraw, kraw) if qkw else ()),
    ]


def _qk_kernels():
    from miniworld_engine.kernels.augmented_attention import cuda as sm90

    return sm90._ext("qknorm_local")


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

    bias2, mod, mkv, q, k, v, sg, x1, xkv, o, lse, u, a2, x2, t, *raw = saved
    wq_dtype = weights[_NBASE].dtype if len(weights) > _NBASE else None
    qkw = [w.float().contiguous() for w in weights[_NBASE:]]
    weights = weights[:_NBASE]
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
        if qkw:   # dq / dk leave the attention as gradients of the normalised tensors: back to the raw projections, in place
            part = _qk_kernels().qknorm_bwd(dp[:, DS : 2 * DS], dp[:, 2 * DS : 3 * DS], raw[0], raw[1], qkw[0], qkw[1], _QK_EPS, _QK_EPS)
            dqkw = part.sum(0)   # [2, 32]: the per-block partials in order
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
    out = outs
    if qkw:
        out = [*outs, dqkw[0].to(wq_dtype).clone(), dqkw[1].to(wq_dtype).clone()]   # outputs of an opaque op may not alias
    dsingle = _ld._unpadded(dx.view(a, 1, n, DS), single.shape)
    if shared:  # the gradient of the one shared row set, split evenly over the expanded view's A copies (autograd sums them back)
        dcond = _ld._unpadded(
            dc.view(1, 1, n, DC).to(cond.dtype).expand(a, 1, n, DC).contiguous(),
            cond.shape,
        )
    else:
        dcond = _ld._unpadded(dc.view(a, 1, n, DC), cond.shape)
    dpair = dz[: pair.shape[1]].reshape(pair.shape).to(pair.dtype)
    return [dsingle, dcond, dpair, *out]


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
    if module.attention.use_qk_norm:
        names = (*names, "attention.norm_query.weight", "attention.norm_key.weight")
    return [module.get_parameter(n) for n in names]


def run_block(module, single, cond, pair, mask=None, rope=None):
    """The block's output stream on the H100 warp-level kernels; ``matches`` must have accepted the call."""
    if _ld._esm_variant(module):
        return _run_esm(module, single, cond, pair, mask, rope)
    weights = _weights_of(module)
    n0 = single.shape[2]
    kmask = _ld._keys(mask, n0, _ld._padded_length(n0))
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (single, cond, pair, *weights)
    ):
        return _H100Block.apply(kmask, single, cond, pair, *weights)
    parallel = module.kv_norm == "parallel"
    return _fwd(
        single, cond, pair, kmask, weights, False, parallel, module.key_window == "slide"
    )[0]


# --------------------------------------------------------------------------------------------------------------------- ESMFold2
# ``LocalDiTBlock(**ESMFOLD2)`` for inference, on the same row kernels built with -DRMS=1 (RMSNorm AdaLNs, tables of silu(c), (1 + s) scales,
# raw gates), the q / k RMSNorm + rotary of qknorm_local.cu, and the 192-key rank-window attention of lattn90_esm.cu:
#
#     cond90 (hoisted) -> pre90 -> qknorm_rope -> pair_bias_esm (hoisted) -> attn_esm -> post90
#
# The row kernels take the AF3 block's 27 parameters; ESMFold2 has no conditioning LayerNorm and no projection biases, so those slots hold
# constants: the LN weights 1, the scale biases 1 (the identity-centred scale), the gate biases 0.
ESM_KEYS = 192
_ESM_NAMES = {
    "sc1_w": "attention.ada_ln_in.to_scale.weight",
    "bi1_w": "attention.ada_ln_in.to_bias.weight",
    "wq": "attention.to_query.weight",
    "bq": "attention.to_query.bias",
    "wk": "attention.to_key.weight",
    "wv": "attention.to_value.weight",
    "ln_pair": "attention.ln_pair.weight",
    "wb": "attention.to_bias.weight",
    "wg": "attention.to_gate.weight",
    "wo": "attention.to_out.weight",
    "so_w": "attention.to_scale.weight",
    "sc2_w": "transition.ada_ln_in.to_scale.weight",
    "bi2_w": "transition.ada_ln_in.to_bias.weight",
    "wa": "transition.expand_a.weight",
    "wb2": "transition.expand_b.weight",
    "wd": "transition.squeeze.weight",
    "st_w": "transition.to_scale.weight",
    "kv_sc_w": "attention.ada_ln_kv.to_scale.weight",
    "kv_bi_w": "attention.ada_ln_kv.to_bias.weight",
}
_ESM_ONES = ("g1", "sc1_b", "g2", "sc2_b", "gkv", "kv_sc_b")
_ESM_ZEROS = ("so_b", "st_b")
_ESM_CONST: dict = {}


def _esm_const(device, value: float):
    """A persistent fp32 constant vector [128] per device (made eagerly and kept; inside a capture with none kept yet, a fresh one that is not kept)."""
    key = (device, value)
    c = _ESM_CONST.get(key)
    if c is None:
        c = torch.full((DS,), value, device=device, dtype=F32)
        if not torch.cuda.is_current_stream_capturing():
            _ESM_CONST[key] = c
    return c


def _esm_weights(module):
    """The module's parameters in the row kernels' 27 slots (constants where ESMFold2 has none), then the q / k RMSNorm weights."""
    dev = module.attention.to_query.weight.device
    out = []
    for name in _NAMES:
        if name in _ESM_NAMES:
            out.append(module.get_parameter(_ESM_NAMES[name]))
        else:
            out.append(_esm_const(dev, 1.0 if name in _ESM_ONES else 0.0))
    return out, [module.attention.norm_query.weight, module.attention.norm_key.weight]


def _esm_matches(module, single, cond, pair, mask, rope) -> bool:
    """H100, exactly ESMFold2's option set (``modules.local_dit.ESMFOLD2``) at the atom widths, inference (no gradient recorded), bf16 or fp32
    single / cond / pair, the rotary tables [1, N, 16]."""
    from miniworld_engine.modules.adaptive_layernorm.module import RMS_EPS
    from miniworld_engine.modules.dispatch import KernelBackend

    if not (single.is_cuda and _hopper(single)) or not settings.current().local_dit_h100_mma:
        return False
    if settings.current().force_portable_kernels or module.attention._backend != KernelBackend.PORTABLE:
        return False
    a, tr = module.attention, module.transition
    options = (module.cross_attention, module.kv_norm, module.key_window, module.keys, module.window_half, module.conditioning,
               module.rope, a.use_qk_norm)
    if options != (True, "parallel", "slide", ESM_KEYS, 64, "rms_silu", True, True):
        return False
    if a.norm_query.eps != RMS_EPS or a.norm_key.eps != RMS_EPS:
        return False
    params = [*(module.get_parameter(n) for n in _ESM_NAMES.values()), a.norm_query.weight, a.norm_key.weight]
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *params))
    if grad and not _esm_train_serves(single, cond, pair):
        return False   # a gradient: only the bf16 training step below (fp32, or a conditioning of its own per sample, run the composed path)
    if any(p.device != single.device or not p.is_contiguous() for p in params):
        return False
    if torch.is_autocast_enabled("cuda") and single.dtype != BF:
        return False
    dt = (single.dtype, cond.dtype, pair.dtype)
    if dt == (F32, F32, F32):
        if any(p.dtype != F32 for p in params) or not settings.current().local_dit_h100_mma_fp32:
            return False
    elif dt == (BF, BF, BF):
        if any(p.dtype not in (F32, BF) for p in params) or not settings.current().local_dit_h100_mma_bf16:
            return False
    else:
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[3] != DS or single.shape[2] == 0:
        return False
    n = single.shape[2]
    if tuple(cond.shape) != (*single.shape[:3], DC) or tuple(pair.shape) != (1, (n + 31) // 32, 32, ESM_KEYS, DP):
        return False
    if not (single.is_contiguous() and pair.is_contiguous() and (cond.is_contiguous() or _shared(cond))):
        return False
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (1, n) or mask.device != single.device):
        return False
    if rope is None or len(rope) != 2 or any(
        t.device != single.device or tuple(t.shape) != (1, n, DS // NH // 2) or not t.is_floating_point() for t in rope
    ):
        return False
    if (a.n_head, a.to_query.weight.shape[0], a.to_bias.weight.shape[1], tr.expand_a.weight.shape[0]) != (NH, DS, DP, 2 * DS):
        return False
    return a.ln_pair.eps == _ld.EPS


def _esm_fwd_fake(single, cond, pair, kmask, weights, qkw, cos, sin):
    return torch.empty_like(single)


@opaque(fake=_esm_fwd_fake, name="local_dit_h100_esm_fwd")
def _esm_fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    kmask: torch.Tensor | None,
    weights: list[torch.Tensor],
    qkw: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """ESMFold2's block output [A, 1, N, 128] (inference): the RMS row kernels, the q / k RMSNorm + rotary and the rank-window attention.
    ``cos`` / ``sin``: the rotary tables, fp32 [n, 16] at the padded length."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm90_atom as k90
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )
    from miniworld_engine.modules.adaptive_layernorm.module import RMS_EPS

    tf32 = single.dtype == F32
    a, n0, n, m = _dims(single)
    with torch.cuda.device(single.device):
        p = k90.pack(weights, tf32)
        s2 = _ld._padded(single, n).reshape(m, DS).contiguous()
        shared = _shared(cond)
        c2 = (_ld._padded(cond[:1], n).reshape(n, DC) if shared else _ld._padded(cond, n).reshape(m, DC)).contiguous()
        nmod = n if shared else m
        hoist = settings.current().local_dit_hoist
        wkey = (tuple((w.data_ptr(), w._version) for w in weights), "esm") if hoist else None

        def tables():
            return k90.cond_fwd(c2, p, tf32, rms=True)

        mod, mkv = _ld._hoisted(_HOIST_COND, cond, (wkey, n), tables) if hoist else tables()
        z = _ld._trunk(pair, n).contiguous()

        def pair_bias():
            nreal = (kmask.sum(dtype=torch.int32) if kmask is not None
                     else torch.full((), n0, device=single.device, dtype=torch.int32))
            return local.pair_bias_esm(z, p["gamma"], p["wbias"], kmask, nreal, n)

        if hoist:
            mk = None if kmask is None else _ld._tkey(kmask)
            bias2 = _ld._hoisted(_HOIST_BIAS, pair, (wkey, n, mk), pair_bias)
        else:
            bias2 = pair_bias()
        qraw, sg, kraw, v = k90.pre_fwd(s2, mod, mkv, p, tf32, eps=RMS_EPS, nmod=nmod, parallel_kv=True, rms=True)
        q, k = torch.empty_like(qraw), torch.empty_like(kraw)
        _qk_kernels().qknorm_rope_fwd(qraw, kraw, qkw[0].float().contiguous(), qkw[1].float().contiguous(), RMS_EPS, RMS_EPS, cos, sin, q, k)
        del qraw, kraw
        o = local.attn_esm(q.view(a, n, DS), k.view(a, n, DS), v.view(a, n, DS), bias2, tf32).view(m, DS)
        out = k90.post_fwd(s2, o, sg, mod, p, tf32, eps=RMS_EPS, nmod=nmod, rms=True)
    return _ld._unpadded(out.view(a, 1, n, DS), single.shape)


def _run_esm(module, single, cond, pair, mask, rope):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *module.parameters())):
        return _esm_train(module, single, cond, pair, mask, rope)
    weights, qkw = _esm_weights(module)
    n0 = single.shape[2]
    n = _ld._padded_length(n0)
    kmask = _ld._keys(mask, n0, n)
    # fp32 [n, 16] at the padded length (no copy when the caller's tables already are)
    cos, sin = (
        (t[0].float() if n == n0 else torch.nn.functional.pad(t[0].float(), (0, 0, 0, n - n0))).contiguous() for t in rope
    )
    return _esm_fwd(single, cond, pair, kmask, weights, qkw, cos, sin)


# --------------------------------------------------------------------------------------------------------------------- ESMFold2 training
# bf16 forward + backward of ``LocalDiTBlock(**ESMFOLD2)`` with the conditioning shared by the samples. The block is the DiT variants' atom
# training block (``kernels/dit_variants/cuda/train.py``: row kernels of ``train_rows.cu`` with their ESMFold2 switch -- RMSNorm statistics and raw
# gates --, cuBLAS for every GEMM) around ESMFold2's own attention: the q / k RMSNorm + rotary (``qknorm_local.cu``) and the rank-window
# attention (``lattn90_esm.cu``: forward with the row LSE, dQ + bias gradient, dK + dV). What lives on the N atom rows (the conditioning
# tables) and on the pair (the windowed bias) is made by ordinary torch ops / autograd Functions outside the block, so their gradients chain
# through the module's own layers:
#
#     tables [N, 8 x 128]   silu(cond) . [Ws1; Wb1; Wskv; Wbkv; Wso; Ws2; Wb2; Wst]^T, the scales + 1 (identity-centred), the gates raw
#     bias2 [4, nwin, 32, 160]   pair_bias_esm (hoist-free: its gradient is needed)
#     block                 x [A N, 128] bf16 in, tables / bias / weights in; dx, dtables, dbias, dweights out
A_SC_Q, A_SH_Q, A_SC_K, A_SH_K, A_G_A, A_SC_T, A_SH_T, A_G_T = range(8)
_RMS_EPS = _QK_EPS   # the RMSNorm epsilon of the engine's RMS layers (``adaptive_layernorm.module.RMS_EPS``: float32's)
ESM_SPAN_ = 160       # the bias columns per query row (``sm100_atom_local.ESM_SPAN``)
_RMS_MODE = 2   # train_rows.cu: bit 1 = ESMFold2 (RMSNorm statistics, raw gates), bit 0 = sigmoid scale (not here)
_ESM_SPLITS = 2
_ESM_TRAIN_NAMES = ("out", "xq", "xk", "xt", "q", "k", "v", "qn", "kn", "g", "og", "lse", "ya", "y2", "xp", "st1", "st2", "ab", "h",
                    "w_gq", "w_kv", "w_o", "w_exp", "w_sq")
_ESM_TRAIN_SAVED = _ESM_TRAIN_NAMES[1:]


def _esm_train_serves(single, cond, pair) -> bool:
    """The training step takes bf16 activations whose conditioning is shared by the samples (one row set: an expanded view, or one sample)."""
    if not settings.current().local_dit_esm_train_cuda:
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (BF, BF, BF):
        return False
    return single.shape[0] == 1 or _shared(cond)


def _esm_rows():
    from miniworld_engine.kernels.dit_variants import cuda as dv

    return dv.train_ext()


def _esm_train_fwd_fake(x, tab, bias2, cos, sin, wq, bq, wk, wv, wg, wo, wexp, wsq, qkw, length, eps):
    m = x.shape[0]
    a = m // length
    t = lambda *shape, d=BF: x.new_empty(shape, dtype=d)  # noqa: E731
    spec = {"out": (m, DS), "xq": (m, DS), "xk": (m, DS), "xt": (m, DS), "q": (m, DS), "k": (m, DS), "v": (m, DS), "qn": (m, DS), "kn": (m, DS),
            "g": (m, DS), "og": (m, DS), "ya": (m, DS), "y2": (m, DS), "ab": (m, 4 * DS), "h": (m, 2 * DS), "w_gq": (2 * DS, DS), "w_kv": (2 * DS, DS),
            "w_o": (DS, DS), "w_exp": (4 * DS, DS), "w_sq": (DS, 2 * DS)}
    out = []
    for n in _ESM_TRAIN_NAMES:
        if n == "lse":
            out.append(t(a, NH, length, d=F32))
        elif n == "xp":
            out.append(t(m, DS, d=F32))
        elif n in ("st1", "st2"):
            out.append(t(m, 2, d=F32))
        else:
            out.append(t(*spec[n]))
    return out


@opaque(fake=_esm_train_fwd_fake, name="local_dit_h100_esm_train_fwd")
def _esm_train_fwd(x: torch.Tensor, tab: torch.Tensor, bias2: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, wq: torch.Tensor, bq: torch.Tensor,
                   wk: torch.Tensor, wv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, wexp: torch.Tensor, wsq: torch.Tensor,
                   qkw: list[torch.Tensor], length: int, eps: float) -> list[torch.Tensor]:
    """One block's forward: [out, saved ...] (``_ESM_TRAIN_NAMES``). x [A N, 128] bf16, tab [N, 8 x 128] bf16, bias2 [4, nwin, 32, 160] fp32 (log2,
    -inf off the band), cos / sin fp32 [N, 16], fp32 or bf16 weights (cast here; ``wexp`` = [Wa; Wb]), qkw = the q / k RMSNorm weights [32]."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    ext, qk = _esm_rows(), _qk_kernels()
    m = x.shape[0]
    n, a = length, m // length
    cols = lambda c: c * DS  # noqa: E731
    w_q, w_k, w_v, w_g, w_o, w_exp, w_sq = (w.to(BF, copy=True) for w in (wq, wk, wv, wg, wo, wexp, wsq))
    w_gq, w_kv = torch.cat([w_g, w_q]), torch.cat([w_k, w_v])
    new = lambda *shape, d=BF: torch.empty(*shape, device=x.device, dtype=d)  # noqa: E731
    xq, xk, xt, g, k, v = new(m, DS), new(m, DS), new(m, DS), new(m, DS), new(m, DS), new(m, DS)
    st1, st2, xp = new(m, 2, d=F32), new(m, 2, d=F32), new(m, DS, d=F32)
    ext.ln_mod_fwd(x, tab, [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SH_Q), cols(A_SH_K)], [xq, xk], st1, n, _RMS_EPS, _RMS_MODE)
    q = torch.addmm(bq.to(BF), xq, w_q.t())
    torch.mm(xk, w_k.t(), out=k)
    torch.mm(xk, w_v.t(), out=v)
    torch.mm(xq, w_g.t(), out=g)
    qn, kn = torch.empty_like(q), torch.empty_like(k)
    qk.qknorm_rope_fwd(q, k, qkw[0].float().contiguous(), qkw[1].float().contiguous(), _RMS_EPS, _RMS_EPS, cos, sin, qn, kn)
    o, lse = local.attn_esm_fwd_lse(qn.view(a, n, DS), kn.view(a, n, DS), v.view(a, n, DS), bias2)
    o = o.view(m, DS)
    og = torch.empty_like(o)
    ext.gate_fwd(o, g, og)
    ya = torch.mm(og, w_o.t())
    ext.res_ln_fwd(x, ya, tab, cols(A_G_A), cols(A_SC_T), cols(A_SH_T), xp, xt, st2, n, _RMS_EPS, _RMS_MODE)
    ab = torch.mm(xt, w_exp.t())
    h = new(m, 2 * DS)
    ext.glu_fwd(ab, h, False)
    y2 = torch.mm(h, w_sq.t())
    out = torch.empty_like(x)
    ext.res_out_fwd(xp, y2, tab, cols(A_G_T), None, 0, out, n, True)
    res = {"out": out, "xq": xq, "xk": xk, "xt": xt, "q": q, "k": k, "v": v, "qn": qn, "kn": kn, "g": g, "og": og, "lse": lse, "ya": ya, "y2": y2,
           "xp": xp, "st1": st1, "st2": st2, "ab": ab, "h": h, "w_gq": w_gq, "w_kv": w_kv, "w_o": w_o, "w_exp": w_exp, "w_sq": w_sq}
    return [res[nm] for nm in _ESM_TRAIN_NAMES]


def _esm_train_bwd_fake(dout, x, tab, bias2, cos, sin, saved, qkw, length, eps):
    f = F32
    return [torch.empty_like(x), tab.new_empty(tab.shape, dtype=f), bias2.new_empty(bias2.shape, dtype=f), x.new_empty((DS, DS), dtype=f),
            x.new_empty((DS,), dtype=f), x.new_empty((DS, DS), dtype=f), x.new_empty((DS, DS), dtype=f), x.new_empty((DS, DS), dtype=f),
            x.new_empty((DS, DS), dtype=f), x.new_empty((4 * DS, DS), dtype=f), x.new_empty((DS, 2 * DS), dtype=f), x.new_empty((32,), dtype=f),
            x.new_empty((32,), dtype=f)]


@opaque(fake=_esm_train_bwd_fake, name="local_dit_h100_esm_train_bwd")
def _esm_train_bwd(dout: torch.Tensor, x: torch.Tensor, tab: torch.Tensor, bias2: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   saved: list[torch.Tensor], qkw: list[torch.Tensor], length: int, eps: float) -> list[torch.Tensor]:
    """[dx, dtab, dbias2, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq, dwql, dwkl] (fp32 but dx)."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    ext, qk = _esm_rows(), _qk_kernels()
    s = dict(zip(_ESM_TRAIN_SAVED, saved, strict=True))
    m = x.shape[0]
    n, a = length, m // length
    dev = x.device
    sp = min(a, _ESM_SPLITS)
    cols = lambda c: c * DS  # noqa: E731
    dout = dout.to(x.dtype).contiguous()
    part = torch.empty(sp, n, 8 * DS, device=dev, dtype=F32)
    w_gq, w_kv, w_o, w_exp, w_sq = s["w_gq"], s["w_kv"], s["w_o"], s["w_exp"], s["w_sq"]
    mmt = lambda p, r: torch.mm(p.t(), r, out_dtype=F32)  # noqa: E731
    dy2 = torch.empty(m, DS, device=dev, dtype=BF)
    ext.res_bwd(dout, s["y2"], tab, cols(A_G_T), dy2, None, 0, None, part, cols(A_G_T), 0, n, sp, True)
    dh = torch.mm(dy2, w_sq)
    dwsq = mmt(dy2, s["h"])
    dab = torch.empty_like(s["ab"])
    ext.glu_bwd(s["ab"], dh, dab, False)
    dt = torch.mm(dab, w_exp)
    dwexp = mmt(dab, s["xt"])
    dxp = torch.empty(m, DS, device=dev, dtype=F32)
    dya = torch.empty(m, DS, device=dev, dtype=BF)
    ext.res_adaln_bwd(dout, dt, s["xp"], s["st2"], tab, cols(A_SC_T), cols(A_G_A), s["ya"], dxp, dya, part, cols(A_SC_T), cols(A_SH_T), cols(A_G_A), n, sp,
                      _RMS_MODE)
    dog = torch.mm(dya, w_o)
    dwo = mmt(dya, s["og"])
    d_o = torch.empty(m, DS, device=dev, dtype=BF)
    dp = torch.empty(m, 4 * DS, device=dev, dtype=BF)               # [dg | dq | dk | dv]
    dd = torch.empty(a, NH, n, device=dev, dtype=F32)
    ext.gate_o_bwd(dog, s["og"], s["g"], d_o, dp[:, :DS], dd, n, DS // NH, False)
    dbias2 = local.attn_esm_bwd(s["qn"].view(a, n, DS), s["kn"].view(a, n, DS), s["v"].view(a, n, DS), d_o.view(a, n, DS), bias2, s["lse"], dd, dp[:, DS:])
    qw = [w.float().contiguous() for w in qkw]
    qpart = qk.qknorm_rope_bwd(dp[:, DS:2 * DS], dp[:, 2 * DS:3 * DS], s["q"], s["k"], qw[0], qw[1], _RMS_EPS, _RMS_EPS, cos, sin)
    dqkw = qpart.sum(0)
    dbq = dp[:, DS:2 * DS].sum(0, dtype=F32)
    dxq = torch.mm(dp[:, :2 * DS], w_gq)
    dwgq = mmt(dp[:, :2 * DS], s["xq"])
    dxk = torch.mm(dp[:, 2 * DS:], w_kv)
    dwkv = mmt(dp[:, 2 * DS:], s["xk"])
    dx = torch.empty_like(x)
    ext.adaln_bwd(x, s["st1"], dxp, [dxq, dxk], tab, [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SH_Q), cols(A_SH_K)], dx, part, n, sp,
                  _RMS_MODE)
    dtab = part.sum(0)
    return [dx, dtab, dbias2, dwgq[DS:].clone(), dbq, dwkv[:DS].clone(), dwkv[DS:].clone(), dwgq[:DS].clone(), dwo, dwexp, dwsq, dqkw[0].clone(), dqkw[1].clone()]


def _esm_pb_fwd_fake(z, gamma, wb, kmask, nreal):
    return z.new_empty((NH, z.shape[0], 32, ESM_SPAN_), dtype=F32)


@opaque(fake=_esm_pb_fwd_fake, name="local_dit_h100_esm_pair_bias_fwd")
def _esm_pb_fwd(z: torch.Tensor, gamma: torch.Tensor, wb: torch.Tensor, kmask: torch.Tensor | None, nreal: torch.Tensor) -> torch.Tensor:
    """bias2 [4, nwin, 32, 160] fp32 of the trunked pair z [nwin, 32, 192, 16] (``sm100_atom_local.pair_bias_esm``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    with torch.cuda.device(z.device):
        return local.pair_bias_esm(z, gamma, wb, kmask, nreal, z.shape[0] * 32)


def _esm_pb_bwd_fake(z, gamma, wb, dbias, nreal):
    return [torch.empty_like(z), gamma.new_empty((DP,), dtype=F32), wb.new_empty(wb.shape, dtype=F32)]


@opaque(fake=_esm_pb_bwd_fake, name="local_dit_h100_esm_pair_bias_bwd")
def _esm_pb_bwd(z: torch.Tensor, gamma: torch.Tensor, wb: torch.Tensor, dbias: torch.Tensor, nreal: torch.Tensor) -> list[torch.Tensor]:
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    with torch.cuda.device(z.device):
        dz, dgamma, dwb = local.pair_bias_esm_bwd(z, gamma, wb, dbias, nreal)
    return [dz, dgamma, dwb]


class _EsmPairBias(torch.autograd.Function):
    """The windowed pair bias of one ESMFold2 block from the trunked pair z [nwin, 32, 192, 16]."""

    @staticmethod
    def forward(ctx, z, gamma, wb, kmask, nreal):  # noqa: D102
        ctx.save_for_backward(z, gamma, wb, nreal)
        return _esm_pb_fwd(z, gamma, wb, kmask, nreal)

    @staticmethod
    @once_differentiable
    def backward(ctx, dbias):  # noqa: D102
        z, gamma, wb, nreal = ctx.saved_tensors
        dz, dgamma, dwb = _esm_pb_bwd(z, gamma, wb, dbias.contiguous(), nreal)
        return dz, dgamma.to(gamma.dtype), dwb.to(wb.dtype), None, None


class _EsmBlock(torch.autograd.Function):
    """The block: forward saving what the backward reads, backward on the same kernels."""

    @staticmethod
    def forward(ctx, x, tab, bias2, cos, sin, wq, bq, wk, wv, wg, wo, wexp, wsq, wql, wkl, length):  # noqa: D102
        out, *saved = _esm_train_fwd(x, tab, bias2, cos, sin, wq, bq, wk, wv, wg, wo, wexp, wsq, [wql, wkl], length, _RMS_EPS)
        ctx.save_for_backward(x, tab, bias2, cos, sin, *saved, wq, wk, wv, wg, wo, wexp, wsq, wql, wkl)
        ctx.n_saved, ctx.length = len(saved), length
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):  # noqa: D102
        x, tab, bias2, cos, sin, *rest = ctx.saved_tensors
        saved, (wq, wk, wv, wg, wo, wexp, wsq, wql, wkl) = rest[:ctx.n_saved], rest[ctx.n_saved:]
        dx, dtab, dbias2, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq, dwql, dwkl = _esm_train_bwd(
            dout, x, tab, bias2, cos, sin, list(saved), [wql, wkl], ctx.length, _RMS_EPS)
        c = lambda g, w: g.to(w.dtype)  # noqa: E731
        return (dx, dtab.to(tab.dtype), dbias2, None, None, c(dwq, wq), dbq.to(wq.dtype), c(dwk, wk), c(dwv, wv), c(dwg, wg), c(dwo, wo), c(dwexp, wexp),
                c(dwsq, wsq), c(dwql, wql), c(dwkl, wkl), None)


def _esm_train(module, single, cond, pair, mask, rope):
    """The ESMFold2 block's output stream [A, 1, N, 128] with a gradient on the H100 training kernels (``_esm_train_serves`` accepted the call)."""
    import torch.nn.functional as Fn

    att, tr = module.attention, module.transition
    a, _, n0, _ = single.shape
    n = _ld._padded_length(n0)
    m = a * n
    kmask = _ld._keys(mask, n0, n)
    nreal = (kmask.sum(dtype=torch.int32) if kmask is not None else torch.full((), n0, device=single.device, dtype=torch.int32))
    cos, sin = ((t[0].float() if n == n0 else Fn.pad(t[0].float(), (0, 0, 0, n - n0))).contiguous() for t in rope)
    x = _ld._padded(single, n).reshape(m, DS)
    c = _ld._padded(cond[:1], n).reshape(n, DC)
    ws = [att.ada_ln_in.to_scale.weight, att.ada_ln_in.to_bias.weight, att.ada_ln_kv.to_scale.weight, att.ada_ln_kv.to_bias.weight, att.to_scale.weight,
          tr.ada_ln_in.to_scale.weight, tr.ada_ln_in.to_bias.weight, tr.to_scale.weight]
    ones = torch.zeros(8 * DS, device=single.device, dtype=BF)
    for blk in (A_SC_Q, A_SC_K, A_SC_T):
        ones[blk * DS:(blk + 1) * DS] = 1
    tab = torch.mm(Fn.silu(c), torch.cat(ws).to(BF).t()) + ones
    z = _ld._trunk(pair, n)
    bias2 = _EsmPairBias.apply(z.contiguous(), att.ln_pair.weight, att.to_bias.weight, kmask, nreal)
    wexp = torch.cat([tr.expand_a.weight, tr.expand_b.weight])
    out = _EsmBlock.apply(x, tab, bias2, cos, sin, att.to_query.weight, att.to_query.bias, att.to_key.weight, att.to_value.weight, att.to_gate.weight,
                          att.to_out.weight, wexp, tr.squeeze.weight, att.norm_query.weight, att.norm_key.weight, n)
    return _ld._unpadded(out.view(a, 1, n, DS), single.shape)
