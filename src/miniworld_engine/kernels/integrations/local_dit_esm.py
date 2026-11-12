"""ESMFold2's AF3-form atom block with the atom pair (``modules.atom_dit.AtomDiTESMFold2Pair``, FoldForge R6) on H100 (sm_90a) and A100
(sm_80): restored from the 2.4.x ``local_dit`` ESMFold2 variant (``LocalDiTBlock(**ESMFOLD2)``, commit 2df9a5a82) onto the 2.5 module.

    H100 inference   cond90 (hoisted, -DRMS=1) -> pre90 (RMS) -> qknorm_rope -> pair_bias_esm (hoisted) -> attn_esm -> post90 (RMS)
                     the sm90_atom row kernels with their ESMFold2 conditioning (RMSNorm AdaLNs, tables of silu(c), (1 + s) scales, raw
                     gates: ``sm90_atom/rms90.cuh``), the q / k RMSNorm + rotary of ``augmented_attention/cuda/qknorm_local.cu`` and the 192-key
                     rank-window attention of ``sm100_atom_local/lattn90_esm.cu``.
    training, and    the row passes of ``dit_variants/cuda/train_rows.cu`` (mode bit 1: RMSNorm statistics and raw gates) and cuBLAS for
    A100 inference   every GEMM around the same q / k norm + rotary and window attention (forward with the row LSE, dQ + bias gradient,
                     dK + dV); the conditioning tables and the windowed pair bias are autograd nodes outside the block, so their gradients
                     chain through the module's own layers. H100: train_rows' sm_90a build, A100: the sm_80 builds (lattn90_esm.cu with its
                     dependent-launch wait emptied, ``dit_variants/cuda/sm80``).

The row kernels take the AF3 block's 27 parameters (``local_dit_h100._NAMES``); ESMFold2 has no conditioning LayerNorm and no projection
biases but the query's, so those slots hold constants: the LN weights 1, the scale biases 1 (the identity-centred scale), the gate biases 0.
The q / k RMSNorm is affine-free (FoldForge ``rope_norm``): its kernel weights are a constant 1. fp32 activations (fp32 parameters, no
autocast) run the TF32 kernels; bf16 activations take bf16 or fp32 (master) parameters. One fold (B = 1), the rotary tables [1, N, 16].
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as Fn
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.integrations import local_dit as _ld

DS, DC, DP, NH = 128, 128, 16, 4
ESM_KEYS, ESM_SPAN = 192, 160
F32, BF = torch.float32, torch.bfloat16
RMS_EPS = 1.1920928955078125e-07  # torch.finfo(torch.float32).eps: every RMSNorm of the block (FoldForge's, whatever the dtype)
_HOIST_COND: dict = _capture.weight_cache("local_dit_esm.hoist_cond")
_HOIST_BIAS: dict = _capture.weight_cache("local_dit_esm.hoist_bias")

#: the module's parameters in the row kernels' slots (``local_dit_h100._NAMES``); the other slots are constants
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
_CONST: dict = {}


def _const(device, value: float, width: int = DS) -> torch.Tensor:
    """A persistent fp32 constant vector per device (made eagerly and kept; inside a capture with none kept yet, a fresh one not kept)."""
    key = (str(device), value, width)
    c = _CONST.get(key)
    if c is None:
        c = torch.full((width,), value, device=device, dtype=F32)
        if not torch.cuda.is_current_stream_capturing():
            _CONST[key] = c
    return c


def _names():
    from miniworld_engine.kernels.integrations.local_dit_h100 import _NAMES

    return _NAMES


def _weights(module) -> list[torch.Tensor]:
    """The module's parameters in the row kernels' 27 slots (constants where ESMFold2 has none)."""
    dev = module.attention.to_query.weight.device
    return [module.get_parameter(_ESM_NAMES[n]) if n in _ESM_NAMES else _const(dev, 1.0 if n in _ESM_ONES else 0.0) for n in _names()]


def _h100(t) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (9, 0)


def _a100(t) -> bool:
    """An A100 (sm_80) or an RTX 5090 (capability 12.0: the same sm_80 paths built for sm_120a, kernels/_sm120.py)."""
    return t.is_cuda and torch.cuda.get_device_capability(t.device) in ((8, 0), (12, 0))


def _shared(cond) -> bool:
    """One conditioning for every sample: [1, 1, N, 128] expanded over the samples (stride 0)."""
    return cond.shape[0] > 1 and cond.stride(0) == 0 and cond.stride(2) == DC and cond.stride(3) == 1


# ------------------------------------------------------------------------------------------------------------------------- the gate
def reason(module, single, cond, pair, mask, rope) -> str | None:
    """Why ``AtomDiTESMFold2Pair``'s ``"cuda"`` backend does not take this call, or None."""
    if not (_h100(single) or _a100(single)):
        return "the ESMFold2 atom block kernels run on an H100 (sm_90), A100 (sm_80) or RTX 5090 (sm_120a) CUDA tensor only"
    att, tr = module.attention, module.transition
    geometry = (att.n_head, att.to_query.weight.shape[0], tr.d_cond, att.to_bias.weight.shape[1], tr.expand_a.weight.shape[0])
    if geometry != (NH, DS, DC, DP, 2 * DS):
        return "geometry: the kernels are built for 128 / cond 128 / pair 16, 4 heads x 32, transition n = 2"
    if att.ln_pair.eps != _ld.EPS:
        return "pair LayerNorm eps != 1e-5"
    params = list(module.parameters())
    if any(p.device != single.device or not p.is_contiguous() for p in params):
        return "a parameter on another device or not contiguous"
    dts = (single.dtype, cond.dtype, pair.dtype)
    if dts == (F32, F32, F32):
        if torch.is_autocast_enabled("cuda"):
            return "float32 activations under autocast"
        if any(p.dtype != F32 for p in params):
            return "float32 activations need float32 parameters (the TF32 kernels)"
    elif dts == (BF, BF, BF):
        if any(p.dtype not in (BF, F32) for p in params):
            return "bfloat16 activations take bfloat16 or float32 parameters"
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != BF:
            return "autocast to a dtype other than bfloat16"
    else:
        return f"dtypes {tuple(str(t) for t in dts)} (single / cond / pair all bfloat16 or all float32)"
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[3] != DS or single.shape[2] == 0:
        return f"single must be [A, 1, N, 128] (one fold), got {tuple(single.shape)}"
    n = single.shape[2]
    if tuple(cond.shape) != (*single.shape[:3], DC):
        return f"cond must be [A, 1, N, 128], got {tuple(cond.shape)}"
    if tuple(pair.shape) != (1, (n + 31) // 32, 32, ESM_KEYS, DP):
        return f"pair must be [1, {(n + 31) // 32}, 32, {ESM_KEYS}, {DP}], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (1, n) or mask.device != single.device):
        return "mask must be a bool [1, N] key mask"
    if rope is None or len(rope) != 2:
        return "rope=(cos, sin) is required"
    for t in rope:
        if t.device != single.device or tuple(t.shape) != (1, n, DS // NH // 2) or not t.is_floating_point():
            return f"rope tables must be [1, N, {DS // NH // 2}] on the activations' device, got {tuple(t.shape)}"
        if t.requires_grad and torch.is_grad_enabled():
            return "the rotary tables take no gradient on the kernels"
    return None


# --------------------------------------------------------------------------------------------------------------------- extensions
def _qk_kernels(device):
    """qknorm_local.cu: the sm_90a extension or the sm_80 one."""
    if _h100_dev(device):
        from miniworld_engine.kernels.augmented_attention import cuda as aa

        return aa._ext("qknorm_local")
    from miniworld_engine.kernels.dit_variants.cuda import sm80

    return sm80.qknorm_ext()


def _rows(device):
    """train_rows.cu: the sm_90a build or the sm_80 one."""
    from miniworld_engine.kernels.dit_variants.cuda import sm80

    return sm80.train_ext_sm90() if _h100_dev(device) else sm80.train_ext()


def _h100_dev(device) -> bool:
    return torch.cuda.get_device_capability(device) == (9, 0)


def _tables_of(rope, n0, n):
    """fp32 [n, 16] rotary tables at the padded length (no copy when the caller's already are)."""
    return tuple((t[0].float() if n == n0 else Fn.pad(t[0].float(), (0, 0, 0, n - n0))).contiguous() for t in rope)


# ------------------------------------------------------------------------------------------------------------------ H100 inference
def _infer_fake(single, cond, pair, mask, weights, cos, sin):
    return torch.empty_like(single)


@opaque(fake=_infer_fake, name="local_dit_esm_h100_infer")
def _infer_h100(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """The block output [A, 1, N, 128] (inference): the RMS row kernels, the q / k RMSNorm + rotary and the rank-window attention."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm90_atom as k90
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    tf32 = single.dtype == F32
    a, n0 = single.shape[0], single.shape[2]
    n = _ld._padded_length(n0)
    m = a * n
    opts = ("RMS",)
    with torch.cuda.device(single.device):
        p = k90.pack(weights, tf32)
        s2 = _ld._padded(single, n).reshape(m, DS).contiguous()
        shared = _shared(cond)
        nmod = n if shared else m
        wkey = (tuple((w.data_ptr(), w._version) for w in weights), "esm")

        # the hoisted builds derive everything from the inputs they are keyed on (cond, pair) and the weights (a hit skips the padding
        # copies; kernels._capture.refresh_static_inputs remakes them in place from a new item's inputs)
        def tables():
            c2 = (_ld._padded(cond[:1], n).reshape(n, DC) if shared else _ld._padded(cond, n).reshape(m, DC)).contiguous()
            return k90.cond_fwd(c2, p, tf32, options=opts)

        mod, mkv = _ld._hoisted(_HOIST_COND, cond, (wkey, n), tables)

        def pair_bias():
            z = _ld._trunk(pair, n).contiguous()
            kmask = _ld._keys(mask, n0, n)  # the caller's [1, N] mask padded here: the table is keyed on the caller's tensor
            nreal = kmask.sum(dtype=torch.int32) if kmask is not None else torch.full((), n0, device=single.device, dtype=torch.int32)
            return local.pair_bias_esm(z, p["gamma"], p["wbias"], kmask, nreal, n)

        mk = (n0, None if mask is None else _ld._tkey(mask))
        bias2 = _ld._hoisted(_HOIST_BIAS, pair, (wkey, n, mk), pair_bias, () if mask is None else (mask,))
        qraw, sg, kraw, v = cast(
            "tuple[torch.Tensor, ...]", k90.pre_fwd(s2, mod, mkv, p, tf32, eps=RMS_EPS, nmod=nmod, parallel_kv=1, options=opts)
        )
        q, k = torch.empty_like(qraw), torch.empty_like(kraw)
        one = _const(single.device, 1.0, DS // NH)
        _qk_kernels(single.device).qknorm_rope_fwd(qraw, kraw, one, one, RMS_EPS, RMS_EPS, cos, sin, q, k)
        del qraw, kraw
        o = local.attn_esm(q.view(a, n, DS), k.view(a, n, DS), v.view(a, n, DS), bias2, tf32).view(m, DS)
        out = cast(torch.Tensor, k90.post_fwd(s2, o, sg, mod, p, tf32, eps=RMS_EPS, nmod=nmod, options=opts))
    return _ld._unpadded(out.view(a, 1, n, DS), single.shape)


# -------------------------------------------------------------------------------------------------- training (and A100 inference)
# tables [tl, 8 x 128]   silu(cond) . [Ws1; Wb1; Wskv; Wbkv; Wso; Ws2; Wb2; Wst]^T, the scales + 1 (identity-centred), the gates raw
#                        (tl = N for a conditioning shared by the samples, A N for one per sample: the row kernels read row (row mod tl))
# bias2 [4, nwin, 32, 160]   pair_bias_esm
# block                  x [A N, 128] in, tables / bias / weights in; dx, dtables, dbias, dweights out
A_SC_Q, A_SH_Q, A_SC_K, A_SH_K, A_G_A, A_SC_T, A_SH_T, A_G_T = range(8)
_RMS_MODE = 2  # train_rows.cu: bit 1 = ESMFold2 (RMSNorm statistics, raw gates), bit 0 = sigmoid scale (not here)
_SPLITS = 2
_TRAIN_NAMES = (
    "out", "xq", "xk", "xt", "q", "k", "v", "qn", "kn", "g", "og", "lse", "ya", "y2", "xp", "st1", "st2", "ab", "h",
    "w_gq", "w_kv", "w_o", "w_exp", "w_sq",
)  # fmt: skip
_TRAIN_SAVED = _TRAIN_NAMES[1:]


def _train_fwd_fake(x, tab, bias2, cos, sin, wq, bq, wk, wv, wg, wo, wexp, wsq, length, tl=0):
    m = x.shape[0]
    a = m // length
    t = lambda *shape, d=x.dtype: x.new_empty(shape, dtype=d)
    spec = {"ab": (m, 4 * DS), "h": (m, 2 * DS), "w_gq": (2 * DS, DS), "w_kv": (2 * DS, DS), "w_o": (DS, DS), "w_exp": (4 * DS, DS),
            "w_sq": (DS, 2 * DS)}  # fmt: skip
    out = []
    for name in _TRAIN_NAMES:
        if name == "lse":
            out.append(t(a, NH, length, d=F32))
        elif name == "xp":
            out.append(t(m, DS, d=F32))
        elif name in ("st1", "st2"):
            out.append(t(m, 2, d=F32))
        else:
            out.append(t(*spec.get(name, (m, DS))))
    return out


@opaque(fake=_train_fwd_fake, name="local_dit_esm_train_fwd")
def _train_fwd(
    x: torch.Tensor,
    tab: torch.Tensor,
    bias2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wq: torch.Tensor,
    bq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    wexp: torch.Tensor,
    wsq: torch.Tensor,
    length: int,
    tl: int = 0,
) -> list[torch.Tensor]:
    """One block's forward: [out, saved ...] (``_TRAIN_NAMES``). x [A N, 128] (bf16, or fp32: TF32 attention, the GEMMs at the caller's
    matmul precision), tab [tl, 8 x 128] in x's dtype, bias2 [4, nwin, 32, 160] fp32 (log2, -inf off the band), cos / sin fp32 [N, 16], fp32
    or bf16 weights (cast here; ``wexp`` = [Wa; Wb])."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    ext, qk = _rows(x.device), _qk_kernels(x.device)
    m = x.shape[0]
    n, a = length, m // length
    tl = tl or n
    cols = lambda c: c * DS
    at = x.dtype
    w_q, w_k, w_v, w_g, w_o, w_exp, w_sq = (w.to(at, copy=True) for w in (wq, wk, wv, wg, wo, wexp, wsq))
    w_gq, w_kv = torch.cat([w_g, w_q]), torch.cat([w_k, w_v])
    new = lambda *shape, d=at: torch.empty(*shape, device=x.device, dtype=d)
    xq, xk, xt, g, k, v = new(m, DS), new(m, DS), new(m, DS), new(m, DS), new(m, DS), new(m, DS)
    st1, st2, xp = new(m, 2, d=F32), new(m, 2, d=F32), new(m, DS, d=F32)
    with torch.cuda.device(x.device):
        ext.ln_mod_fwd(x, tab, [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SH_Q), cols(A_SH_K)], [xq, xk], st1, tl, RMS_EPS, _RMS_MODE)
        q = torch.addmm(bq.to(at), xq, w_q.t())
        torch.mm(xk, w_k.t(), out=k)
        torch.mm(xk, w_v.t(), out=v)
        torch.mm(xq, w_g.t(), out=g)
        qn, kn = torch.empty_like(q), torch.empty_like(k)
        one = _const(x.device, 1.0, DS // NH)
        qk.qknorm_rope_fwd(q, k, one, one, RMS_EPS, RMS_EPS, cos, sin, qn, kn)
        o, lse = local.attn_esm_fwd_lse(qn.view(a, n, DS), kn.view(a, n, DS), v.view(a, n, DS), bias2)
        o = o.view(m, DS)
        og = torch.empty_like(o)
        ext.gate_fwd(o, g, og)
        ya = torch.mm(og, w_o.t())
        ext.res_ln_fwd(x, ya, tab, cols(A_G_A), cols(A_SC_T), cols(A_SH_T), xp, xt, st2, tl, RMS_EPS, _RMS_MODE)
        ab = torch.mm(xt, w_exp.t())
        h = new(m, 2 * DS)
        ext.glu_fwd(ab, h, False)
        y2 = torch.mm(h, w_sq.t())
        out = torch.empty_like(x)
        ext.res_out_fwd(xp, y2, tab, cols(A_G_T), None, 0, out, tl, True)
    res = {"out": out, "xq": xq, "xk": xk, "xt": xt, "q": q, "k": k, "v": v, "qn": qn, "kn": kn, "g": g, "og": og, "lse": lse, "ya": ya,
           "y2": y2, "xp": xp, "st1": st1, "st2": st2, "ab": ab, "h": h, "w_gq": w_gq, "w_kv": w_kv, "w_o": w_o, "w_exp": w_exp,
           "w_sq": w_sq}  # fmt: skip
    return [res[name] for name in _TRAIN_NAMES]


def _train_bwd_fake(dout, x, tab, bias2, cos, sin, saved, length, tl=0):
    e = lambda *s: x.new_empty(s, dtype=F32)
    return [torch.empty_like(x), tab.new_empty(tab.shape, dtype=F32), bias2.new_empty(bias2.shape, dtype=F32), e(DS, DS), e(DS),
            e(DS, DS), e(DS, DS), e(DS, DS), e(DS, DS), e(4 * DS, DS), e(DS, 2 * DS)]  # fmt: skip


@opaque(fake=_train_bwd_fake, name="local_dit_esm_train_bwd")
def _train_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    tab: torch.Tensor,
    bias2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    saved: list[torch.Tensor],
    length: int,
    tl: int = 0,
) -> list[torch.Tensor]:
    """[dx, dtab, dbias2, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq] (fp32 but dx)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    ext, qk = _rows(x.device), _qk_kernels(x.device)
    s = dict(zip(_TRAIN_SAVED, saved, strict=True))
    m = x.shape[0]
    n, a = length, m // length
    tl = tl or n
    dev = x.device
    sp = min(m // tl, _SPLITS)
    cols = lambda c: c * DS
    dout = dout.to(x.dtype).contiguous()
    w_gq, w_kv, w_o, w_exp, w_sq = s["w_gq"], s["w_kv"], s["w_o"], s["w_exp"], s["w_sq"]
    mmt = lambda p, r: torch.mm(p.t(), r, out_dtype=F32)
    at = x.dtype
    with torch.cuda.device(dev):
        part = torch.empty(sp, tl, 8 * DS, device=dev, dtype=F32)
        dy2 = torch.empty(m, DS, device=dev, dtype=at)
        ext.res_bwd(dout, s["y2"], tab, cols(A_G_T), dy2, None, 0, None, part, cols(A_G_T), 0, tl, sp, True)
        dh = torch.mm(dy2, w_sq)
        dwsq = mmt(dy2, s["h"])
        dab = torch.empty_like(s["ab"])
        ext.glu_bwd(s["ab"], dh, dab, False)
        dt = torch.mm(dab, w_exp)
        dwexp = mmt(dab, s["xt"])
        dxp = torch.empty(m, DS, device=dev, dtype=F32)
        dya = torch.empty(m, DS, device=dev, dtype=at)
        ext.res_adaln_bwd(dout, dt, s["xp"], s["st2"], tab, cols(A_SC_T), cols(A_G_A), s["ya"], dxp, dya, part, cols(A_SC_T), cols(A_SH_T),
                          cols(A_G_A), tl, sp, _RMS_MODE)  # fmt: skip
        dog = torch.mm(dya, w_o)
        dwo = mmt(dya, s["og"])
        d_o = torch.empty(m, DS, device=dev, dtype=at)
        dp = torch.empty(m, 4 * DS, device=dev, dtype=at)  # [dg | dq | dk | dv]
        dd = torch.empty(a, NH, n, device=dev, dtype=F32)
        ext.gate_o_bwd(dog, s["og"], s["g"], d_o, dp[:, :DS], dd, n, DS // NH, at == F32)
        dbias2 = local.attn_esm_bwd(
            s["qn"].view(a, n, DS), s["kn"].view(a, n, DS), s["v"].view(a, n, DS), d_o.view(a, n, DS), bias2, s["lse"], dd, dp[:, DS:]
        )
        one = _const(dev, 1.0, DS // NH)
        qk.qknorm_rope_bwd(dp[:, DS : 2 * DS], dp[:, 2 * DS : 3 * DS], s["q"], s["k"], one, one, RMS_EPS, RMS_EPS, cos, sin)
        dbq = dp[:, DS : 2 * DS].sum(0, dtype=F32)
        dxq = torch.mm(dp[:, : 2 * DS], w_gq)
        dwgq = mmt(dp[:, : 2 * DS], s["xq"])
        dxk = torch.mm(dp[:, 2 * DS :], w_kv)
        dwkv = mmt(dp[:, 2 * DS :], s["xk"])
        dx = torch.empty_like(x)
        ext.adaln_bwd(x, s["st1"], dxp, [dxq, dxk], tab, [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SC_Q), cols(A_SC_K)],
                      [cols(A_SH_Q), cols(A_SH_K)], dx, part, tl, sp, _RMS_MODE)  # fmt: skip
        dtab = part.sum(0)
    return [dx, dtab, dbias2, dwgq[DS:].clone(), dbq, dwkv[:DS].clone(), dwkv[DS:].clone(), dwgq[:DS].clone(), dwo, dwexp, dwsq]


def _pb_fwd_fake(z, gamma, wb, kmask, nreal):
    return z.new_empty((NH, z.shape[0], 32, ESM_SPAN), dtype=F32)


@opaque(fake=_pb_fwd_fake, name="local_dit_esm_pair_bias_fwd")
def _pb_fwd(z: torch.Tensor, gamma: torch.Tensor, wb: torch.Tensor, kmask: torch.Tensor | None, nreal: torch.Tensor) -> torch.Tensor:
    """bias2 [4, nwin, 32, 160] fp32 of the trunked pair z [nwin, 32, 192, 16] (``sm100_atom_local.pair_bias_esm``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    with torch.cuda.device(z.device):
        return local.pair_bias_esm(z, gamma, wb, kmask, nreal, z.shape[0] * 32)


def _pb_bwd_fake(z, gamma, wb, dbias, nreal):
    return [torch.empty_like(z), gamma.new_empty((DP,), dtype=F32), wb.new_empty(wb.shape, dtype=F32)]


@opaque(fake=_pb_bwd_fake, name="local_dit_esm_pair_bias_bwd")
def _pb_bwd(z: torch.Tensor, gamma: torch.Tensor, wb: torch.Tensor, dbias: torch.Tensor, nreal: torch.Tensor) -> list[torch.Tensor]:
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    with torch.cuda.device(z.device):
        dz, dgamma, dwb = local.pair_bias_esm_bwd(z, gamma, wb, dbias, nreal)
    return [dz, dgamma, dwb]


class _PairBias(torch.autograd.Function):
    """The windowed pair bias of one block from the trunked pair z [nwin, 32, 192, 16]."""

    @staticmethod
    def forward(ctx, z, gamma, wb, kmask, nreal):
        ctx.save_for_backward(z, gamma, wb, nreal)
        return _pb_fwd(z, gamma, wb, kmask, nreal)

    @staticmethod
    @once_differentiable
    def backward(ctx, dbias):
        z, gamma, wb, nreal = ctx.saved_tensors
        dz, dgamma, dwb = _pb_bwd(z, gamma, wb, dbias.contiguous(), nreal)
        return dz, dgamma.to(gamma.dtype), dwb.to(wb.dtype), None, None


class _Block(torch.autograd.Function):
    """The block: forward saving what the backward reads, backward on the same kernels."""

    @staticmethod
    def forward(ctx, x, tab, bias2, cos, sin, wq, bq, wk, wv, wg, wo, wexp, wsq, length, tl=0):
        out, *saved = _train_fwd(x, tab, bias2, cos, sin, wq, bq, wk, wv, wg, wo, wexp, wsq, length, tl)
        ctx.tl, ctx.length, ctx.n_saved = tl, length, len(saved)
        ctx.save_for_backward(x, tab, bias2, cos, sin, *saved, wq, wk, wv, wg, wo, wexp, wsq)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        x, tab, bias2, cos, sin, *rest = ctx.saved_tensors
        saved, (wq, wk, wv, wg, wo, wexp, wsq) = rest[: ctx.n_saved], rest[ctx.n_saved :]
        dx, dtab, dbias2, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq = _train_bwd(
            dout, x, tab, bias2, cos, sin, list(saved), ctx.length, ctx.tl
        )
        c = lambda g, w: g.to(w.dtype)
        return (dx, dtab.to(tab.dtype), dbias2, None, None, c(dwq, wq), dbq.to(wq.dtype), c(dwk, wk), c(dwv, wv), c(dwg, wg), c(dwo, wo),
                c(dwexp, wexp), c(dwsq, wsq), None, None)  # fmt: skip


def _block(module, single, cond, pair, kmask, cos, sin):
    """The block's output stream [A, 1, N, 128] on the row-pass training block (differentiable; also A100 inference)."""
    att, tr = module.attention, module.transition
    a, _, n0, _ = single.shape
    n = _ld._padded_length(n0)
    m = a * n
    nreal = kmask.sum(dtype=torch.int32) if kmask is not None else torch.full((), n0, device=single.device, dtype=torch.int32)
    x = _ld._padded(single, n).reshape(m, DS)
    shared = a == 1 or _shared(cond)
    c = _ld._padded(cond[:1], n).reshape(n, DC) if shared else _ld._padded(cond, n).reshape(m, DC)
    ws = [att.ada_ln_in.to_scale.weight, att.ada_ln_in.to_bias.weight, att.ada_ln_kv.to_scale.weight, att.ada_ln_kv.to_bias.weight,
          att.to_scale.weight, tr.ada_ln_in.to_scale.weight, tr.ada_ln_in.to_bias.weight, tr.to_scale.weight]  # fmt: skip
    dt = single.dtype
    ones = torch.zeros(8 * DS, device=single.device, dtype=dt)
    for blk in (A_SC_Q, A_SC_K, A_SC_T):
        ones[blk * DS : (blk + 1) * DS] = 1
    tab = torch.mm(Fn.silu(c), torch.cat(ws).to(dt).t()) + ones
    z = _ld._trunk(pair, n)
    bias2 = _PairBias.apply(z.contiguous(), att.ln_pair.weight, att.to_bias.weight, kmask, nreal)
    wexp = torch.cat([tr.expand_a.weight, tr.expand_b.weight])
    out = _Block.apply(x, tab, bias2, cos, sin, att.to_query.weight, att.to_query.bias, att.to_key.weight, att.to_value.weight,
                       att.to_gate.weight, att.to_out.weight, wexp, tr.squeeze.weight, n, n if shared else m)  # fmt: skip
    return _ld._unpadded(out.view(a, 1, n, DS), single.shape)


def run_block(module, single, cond, pair, mask=None, rope=None):
    """The block's output stream; :func:`reason` must have accepted the call. H100 without a gradient: the fused row kernels; otherwise
    (a gradient, or an A100) the row-pass block."""
    assert rope is not None  # reason() accepted the call
    n0 = single.shape[2]
    n = _ld._padded_length(n0)
    cos, sin = _tables_of(rope, n0, n)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *module.parameters()))
    if not grad and _h100(single):
        if not (cond.is_contiguous() or _shared(cond)):
            cond = cond.contiguous()
        return _infer_h100(single.contiguous(), cond, pair.contiguous(), mask, _weights(module), cos, sin)
    return _block(module, single.contiguous(), cond, pair.contiguous(), _ld._keys(mask, n0, n), cos, sin)
