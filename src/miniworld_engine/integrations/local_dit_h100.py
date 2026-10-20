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

DS, DC, DP, NH = 128, 128, 16, 4
F32 = torch.float32
BF = torch.bfloat16
_HOIST_COND: dict = {}
_HOIST_BIAS: dict = {}


def _hopper(t) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (9, 0)


def matches(module, single, cond, pair, mask) -> bool:
    """H100, cross-attention AF3 atom block at the atom widths, fp32 inference (no gradient recorded)."""
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
    if a.use_qk_norm or settings.current().engine_backend == "triton":
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    if a._backend != KernelBackend.TRITON:
        return False
    params = [module.get_parameter(n) for n in (*_ld.WEIGHTS, *_ld.CROSS_WEIGHTS)]
    if any(p.device != single.device or not p.is_contiguous() for p in params):
        return False
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


def _slid_edges(o, q, k, v, z, gamma, wb, a, n, kmask, n0):
    """Overwrite the attention output ``o`` [A, N, 128] of the windows whose slid key window is not the clipped one with the attention
    over their slid key windows (torch, fp32). Those are windows 0 and 1 and the last three real windows (a window ends at
    ``32 w + 80``, so at most three end past the last real atom): five windows chosen on the device from the real atom count, so there
    is no host read and the call can be captured. A window chosen twice is computed twice with the same result."""
    import torch.nn.functional as F

    h, d = NH, DS // NH
    nwin = n // 32
    n_real = (
        kmask.sum()
        if kmask is not None
        else torch.full((), n0, device=o.device, dtype=torch.int64)
    )
    last = (n_real - 1) // 32
    idx = torch.stack([last * 0, last * 0 + 1, last - 2, last - 1, last]).clamp(0, nwin - 1)  # [5]
    starts = torch.minimum((32 * idx - 48).clamp(min=0), (n_real - DS).clamp(min=0))
    keys = starts[:, None] + torch.arange(DS, device=o.device)  # [5, 128]
    valid = keys < n_real  # a system under 128 real atoms has keys past them
    qs = q.view(a, nwin, 32, h, d)[:, idx].float()
    ks = k.view(a, n, h, d)[:, keys].float()
    vs = v.view(a, n, h, d)[:, keys].float()
    zz = F.layer_norm(z[idx].float(), (DP,), gamma.float(), None, _ld.EPS)
    bias = F.linear(zz, wb.float()).permute(0, 3, 1, 2)  # [5, h, 32, 128]
    logits = torch.einsum("awqhd,awkhd->awhqk", qs, ks) * d**-0.5 + bias[None]
    logits = logits.masked_fill(~valid[None, :, None, None, :], float("-inf"))
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
        o, lse = attn(q.view(a, n, DS), k.view(a, n, DS), v.view(a, n, DS), bias2)
        if slide:
            _slid_edges(o, q, k, v, z, weights[_ld._GAMMA], weights[_ld._WBIAS], a, n, kmask, _n0)
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
    out = outs
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
        with _ld._tf32():
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
    return [module.get_parameter(n) for n in (*_ld.WEIGHTS, *_ld.CROSS_WEIGHTS)]


def run_block(module, single, cond, pair, mask=None):
    """The block's output stream on the H100 warp-level kernels; ``matches`` must have accepted the call."""
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
