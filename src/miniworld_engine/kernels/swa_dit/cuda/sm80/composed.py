"""The whole SWA atom block on A100 (sm_80) for what the fused bf16 inference stages (``block_fwd``) do not cover: a gradient, fp32
activations and global attention. ``AtomDiTMiniWorld._forward_a100`` calls :func:`block` (inference) or :class:`SWABlockA100` (training).

* bf16 (``master``: fp32 weights rounded to bf16, gradients handed back unrounded in fp32): the sm_80 stages of this package -- ``mod_fwd``,
  ``qkvg_fwd``, the window attention, ``ffn_fwd`` and their backwards ``ffn_bwd`` -> ``oproj_bwd`` -> ``attn_bwd`` -> ``qkvg_bwd`` ->
  ``mod_bwd`` (one modulation row per sequence row: ``MODE_SINGLE``); the weight gradients are cuBLAS GEMMs with an fp32 result.
* fp32 inference under TF32 (``block``): the two fused TF32 row kernels of ``fp32_fwd_sm80.cuh`` around the attention (``_infer_fp32``).
* fp32: the H100 native fp32 stages (``dispatch._native_h100_fwd`` / ``_bwd``) ported -- the row stages of ``rows_sm80.cuh`` around
  cuBLAS projections (fp32, TF32 as torch's matmul flag says) and the bf16 window attention of this package (the H100 fp32 block also
  runs its attention on bf16 Q / K / V).
* global attention (a half window ``is_global``): PyTorch's bundled FlashAttention-2 (``aten._flash_attention_forward`` / ``_backward``,
  sm_80) between the same stages, in place of the window kernels -- the A100 twin of the H100 FlashAttention-4 path. Each sample is two
  variable-length sequences, its valid atoms ``[0, seqused)`` and its padding; the padding sequence sees no keys (``seqused_k`` 0), so its
  output is 0 and its lse +inf, and its backward is 0.

d_atom 128 = 4 heads x 32, d_cond 128, SwiGLU hidden 256; ``mod`` [N S, 768] fp32 (one modulation row per sequence row).
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels import _sm120
from miniworld_engine.kernels.swa_dit._common import FP32_EPS, is_global
from miniworld_engine.kernels.swa_dit.cuda import sm80

C, H, D = sm80.C, sm80.H, sm80.D
_BF, _F32 = torch.bfloat16, torch.float32


def _rows(stage: int, ins: list, outs: list, n: int, s: int, bf: bool) -> None:
    """One row stage of ``rows_sm80.cuh`` over N = n sequences of s rows, each with its own modulation row (A = 1, B = n)."""
    sm80._ext().native_rows(ins, outs, stage, 1, n, s, bf, FP32_EPS)


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with an fp32 result (a bf16 product keeps cuBLAS's fp32 accumulator)."""
    return torch.mm(a, b, out_dtype=_F32) if a.dtype == _BF else torch.mm(a, b)


# ------------------------------------------------------------------------------------------------------------------- the attention
def _varlen(seqused: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(cu_seqlens [2N + 1], seqused_k [2N]) int32: sample n is the sequences [n s, n s + u_n) and [n s + u_n, (n + 1) s), the second
    without keys (one small kernel, ``ops.cu`` swa_varlen)."""
    n = seqused.shape[0]
    cu = torch.empty(2 * n + 1, device=seqused.device, dtype=torch.int32)
    used_k = torch.empty(2 * n, device=seqused.device, dtype=torch.int32)
    sm80._ext().swa_varlen(seqused.to(torch.int32).contiguous(), s, cu, used_k)
    return cu, used_k


def _token_major(t: torch.Tensor) -> torch.Tensor:
    """Head-major [N, 4, S, 32] -> [N S, 4, 32] contiguous."""
    n, h, s, d = t.shape
    return t.transpose(1, 2).reshape(n * s, h, d)


def _head_major(t: torch.Tensor, n: int, s: int) -> torch.Tensor:
    """[N S, 4, 32] -> head-major [N, 4, S, 32] contiguous."""
    return t.view(n, s, H, D).transpose(1, 2).contiguous()


def attention_fwd(qh, kh, vh, seqused, half_window: int, out_dtype: torch.dtype = _BF):
    """The attention of the block on bf16 head-major Q / K / V [N, 4, S, 32]: ``(O [N S, 128] bf16, lse)``, rows >= seqused 0. The
    window (half window 64) on the sm_80 kernel; global attention on FlashAttention-2 (its own lse layout, read back only by
    :func:`attention_bwd`). ``out_dtype`` fp32 (the window kernel only): O
    unrounded, as the fp32 block takes it."""
    if not is_global(half_window):
        if out_dtype == _BF:
            return sm80.attn_fwd(qh, kh, vh, seqused)
        n, _, s, _ = qh.shape
        o = torch.empty(n * s, C, device=qh.device, dtype=_F32)
        lse = torch.empty(n, H, s, device=qh.device, dtype=_F32)
        sm80._ext().swa_attn_fwd(qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2), seqused, o, lse)
        return o, lse
    n, _, s, _ = qh.shape
    cu, used_k = _varlen(seqused, s)
    q, k, v = (_token_major(t) for t in (qh, kh, vh))
    o, lse, _rng, _unused, _dbg = torch.ops.aten._flash_attention_forward(
        q, k, v, cu, cu, s, s, 0.0, False, False, scale=D**-0.5, seqused_k=used_k
    )
    return o.reshape(n * s, C), lse


def attention_bwd(qh, kh, vh, o, d_o, lse, delta, seqused, half_window: int):
    """``(dQh, dKh, dVh)`` head-major bf16 of :func:`attention_fwd` from ``d_o`` [N S, 128] bf16 and ``delta`` = rowsum(dO o) [N, 4, S]
    fp32 (the window kernel's; FlashAttention computes its own)."""
    if not is_global(half_window):
        return sm80.attn_bwd(qh, kh, vh, d_o, lse, delta, seqused)
    n, _, s, _ = qh.shape
    cu, _used_k = _varlen(seqused, s)
    q, k, v = (_token_major(t) for t in (qh, kh, vh))
    empty = torch.empty(0, device=qh.device)
    dq, dk, dv = torch.ops.aten._flash_attention_backward(
        d_o.view(n * s, H, D), q, k, v, o.view(n * s, H, D), lse, cu, cu, s, s, 0.0, False, empty, empty, scale=D**-0.5
    )
    return tuple(_head_major(t, n, s) for t in (dq, dk, dv))


# ------------------------------------------------------------------------------------------------------------------- bf16 stages
def _fwd_bf16(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window: int, save: bool) -> list[torch.Tensor]:
    """``sm80.block_fwd`` with the attention of :func:`attention_fwd`: ``[out]`` or ``[out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y,
    FF]``."""
    n, s, _ = q.shape
    qf = q.reshape(n * s, C)
    qh, kh, vh, g, xs, pqs, pks = sm80.qkvg_fwd(qf, mod, cos, sin, wqkv, wg, n, s, FP32_EPS, FP32_EPS, save)
    o, lse = attention_fwd(qh, kh, vh, seqused, half_window)
    out, q1s, atts, ys, ffs = sm80.ffn_fwd(qf, g, o, mod, wo, wu, wd, n, s, FP32_EPS, save)
    outputs = [out.view(n, s, C)]
    if save:
        outputs += [qh, kh, vh, g, o, lse, q1s, xs, pqs, pks, atts, ys, ffs]
    return outputs


def _bwd_bf16(
    dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, half_window: int
) -> list[torch.Tensor]:
    """``[dq, dmod, dWqkv, dWg, dWo, dWu, dWd]`` (dq bf16, the rest fp32) of :func:`_fwd_bf16` on the sm_80 backward stages."""
    n, s, _ = q.shape
    m = n * s
    mode = sm80.MODE_SINGLE
    dmod = sm80.dmod_buffer(mode, n, n, s, q.device)
    dq1, dffn, hh, dab = sm80.ffn_bwd(dy.reshape(m, C), q1, y, ffn, mod, wu, wd, dmod, n, s, FP32_EPS, mode)
    d_o, dg, datt, gated, delta = sm80.oproj_bwd(dq1, o, g, att, mod, wo, dmod, n, s, mode)
    dqh, dkh, dvh = attention_bwd(qh, kh, vh, o, d_o, lse, delta, seqused, half_window)
    dq, dp = sm80.qkvg_bwd(q.reshape(m, C), pq, pk, dqh, dkh, dvh, dg, dq1, mod, cos, sin, wqkv, wg, dmod, n, s, FP32_EPS, FP32_EPS, mode)
    return [
        dq.view(n, s, C),
        dmod,
        _mm(dp[:, : 3 * C].t(), x),
        _mm(dp[:, 3 * C :].t(), x),
        _mm(datt.t(), gated),
        _mm(dab.t(), y),
        _mm(dffn.t(), hh),
    ]


# ------------------------------------------------------------------------------------------------------------------- fp32 stages
def _fwd_fp32(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window: int, save: bool) -> list[torch.Tensor]:
    """The fp32 block (``dispatch._native_h100_fwd``'s fp32 branch on the sm_80 row stages): the same outputs as :func:`_fwd_bf16`, every
    tensor fp32 but Q / K / V (bf16), O (fp32 with the window; bf16 from FlashAttention) and lse.

    bf16 ``q`` (the RTX 5090, whose 99 KB of shared memory per block do not hold the fused bf16 stages' 128 KB weight tiles): the same stages
    in bf16 -- every activation bf16, the projections bf16 cuBLAS products (fp32 accumulation, one rounding, as the module's bf16 Linear),
    the row stages with their bf16 rounding points (``rows_sm80.cuh``), O bf16."""
    n, s, _ = q.shape
    m = n * s
    dev = q.device
    bf = q.dtype == _BF
    act = q.dtype
    mm = torch.mm if bf else _mm

    def empty(*shape, dtype=act):
        return torch.empty(shape, device=dev, dtype=dtype)

    x = empty(m, C)
    _rows(0, [q, mod], [x], n, s, bf)
    from miniworld_engine.kernels.swa_dit.dispatch import eval_form

    wstack = eval_form("swa_a100_fp32_wstack", (wqkv, wg), lambda: torch.cat([wqkv, wg]), save)  # per weight version in inference
    p = mm(x, wstack.t())
    qh, kh, vh = (empty(n, H, s, D, dtype=_BF) for _ in range(3))
    g, pq, pk = (empty(m, C) for _ in range(3))
    _rows(1, [p, cos, sin], [qh, kh, vh, g, pq, pk], n, s, bf)
    o, lse = attention_fwd(qh, kh, vh, seqused, half_window, act)  # fp32 O from the window kernel (global: bf16); bf16 O for bf16
    gated = empty(m, C)
    _rows(2, [o, g], [gated], n, s, bf)
    att = mm(gated, wo.t())
    q1, y = empty(m, C), empty(m, C)
    _rows(3, [q, att, mod], [q1, y], n, s, bf)
    ab = mm(y, wu.t())
    hh = empty(m, 2 * C)
    _rows(4, [ab], [hh], n, s, bf)
    ffn = mm(hh, wd.t())
    out = empty(m, C)
    _rows(5, [q1, ffn, mod], [out], n, s, bf)
    outputs = [out.view(n, s, C)]
    if save:
        outputs += [qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn]
    return outputs


def _bwd_fp32(
    dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, half_window: int
) -> list[torch.Tensor]:
    """``[dq, dmod, dWqkv, dWg, dWo, dWu, dWd]`` fp32 of :func:`_fwd_fp32` (``dispatch._native_h100_bwd``'s fp32 branch); for bf16 (the
    RTX 5090) the activation gradients bf16 as the forward's, dmod and the weight gradients fp32."""
    n, s, _ = q.shape
    m = n * s
    dev = q.device
    bf = q.dtype == _BF
    act = q.dtype
    mm = torch.mm if bf else _mm

    def empty(*shape, dtype=act):
        return torch.empty(shape, device=dev, dtype=dtype)

    dmod = torch.empty(mod.shape, device=dev, dtype=_F32)
    dffn = empty(m, C)
    _rows(6, [dy, mod, ffn], [dffn, dmod], n, s, bf)
    ab, dh = mm(y, wu.t()), mm(dffn, wd)
    dab, hh = empty(m, 4 * C), empty(m, 2 * C)
    _rows(7, [ab, dh], [dab, hh], n, s, bf)
    d_y = mm(dab, wu)
    dq1 = empty(m, C)
    _rows(8, [dy, d_y, q1, mod], [dq1, dmod], n, s, bf)
    datt, gated = empty(m, C), empty(m, C)
    _rows(9, [dq1, mod, att, o, g], [datt, gated, dmod], n, s, bf)
    dgated = mm(datt, wo)
    d_o, dg, delta = empty(m, C, dtype=_BF), empty(m, C), empty(n, H, s, dtype=_F32)
    _rows(10, [dgated, o, g], [d_o, dg, delta], n, s, bf)
    dqh, dkh, dvh = attention_bwd(qh, kh, vh, o, d_o, lse, delta, seqused, half_window)
    dp = empty(m, 4 * C)
    _rows(11, [pq, pk, dqh, dkh, dvh, dg, cos, sin], [dp], n, s, bf)
    wstack = torch.cat([wqkv, wg])
    dx = mm(dp, wstack)
    dq = empty(m, C)
    _rows(12, [dq1, dx, q, mod], [dq, dmod], n, s, bf)
    return [
        dq.view(n, s, C),
        dmod,
        _mm(dp[:, : 3 * C].t(), x),
        _mm(dp[:, 3 * C :].t(), x),
        _mm(datt.t(), gated),
        _mm(dab.t(), y),
        _mm(dffn.t(), hh),
    ]


# ------------------------------------------------------------------------------------------------------------------- inference
# The inference forward (no saves) of what ``block`` serves: bf16 global attention (the bf16 window goes to ``sm80.block_fwd``) and fp32.
# fp32 under TF32 (torch's matmul flag, as the composed stages' cuBLAS projections follow it) runs the two fused TF32 row kernels of
# ``fp32_fwd_sm80.cuh`` around the attention -- 3 launches in place of 12; exact fp32 products (TF32 off) keep the composed stages.
# Global attention takes Q / K / V token-major straight from the qkvg kernel (FlashAttention-2's varlen layout, no transposes).


def _tf32_rna(w: torch.Tensor) -> torch.Tensor:
    """w rounded to tf32 (cvt.rna: to nearest, ties away from zero -- the kernels round their activations so), fp32."""
    wi = w.contiguous().view(torch.int32)
    return ((wi + 0x1000) & -8192).view(torch.float32)


def _pack(w: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
    """[N, K] fp32 weight (its rows reordered by ``rows``: packed row -> source row) -> tf32 B fragments in ``fp32_fwd_sm80.cuh``'s
    ``warp_gemm`` order, [8 warps][K / 16][NT][8 g][4 t][2][2] with NT = N / 64: warp w's n tile nt is packed rows 8 (w NT + nt) .."""
    if rows is not None:
        w = w.index_select(0, rows)
    n, k = w.shape
    nt = n // 64
    return _tf32_rna(w).view(8, nt, 8, k // 16, 2, 4, 2).permute(0, 3, 1, 2, 5, 4, 6).contiguous()


def _qkvg_rows(device) -> torch.Tensor:
    """Packed row -> output channel of [Wq; Wk; Wv; Wg] (the f1 order, ``qf_channel``: thread t's 4 n tiles of a head are 8 consecutive
    channels 8t ..)."""
    s = torch.arange(512, device=device)
    loc = s % 64
    return (s - loc) + 32 * ((loc >> 5) & 1) + 8 * ((loc >> 1) & 3) + 2 * ((loc >> 3) & 3) + (loc & 1)


def _wu_rows(device) -> torch.Tensor:
    """Packed row -> row of Wu: warp w's n tiles 0..3 are a units 32 w .., 4..7 the b units 256 + 32 w .."""
    w = torch.arange(8, device=device)[:, None, None]
    half = torch.arange(2, device=device)[None, :, None]
    r = torch.arange(32, device=device)[None, None, :]
    return (half * 256 + 32 * w + r).reshape(-1)


def _fp32_packs(wqkv, wg, wo, wu, wd) -> list[torch.Tensor]:
    dev = wqkv.device
    return [_pack(torch.cat([wqkv, wg]), _qkvg_rows(dev)), _pack(wo), _pack(wu, _wu_rows(dev)), _pack(wd)]


#: rows per CTA tile / 16 of the fp32 TF32 kernels (the FFN kernel at most 2): None = the shortest tile whose tile count fits one wave of
#: CTAs (one per SM), else 4 (a weight pass per tile: longer tiles stream fewer); a tuple ((max rows, mt), ..) overrides (experiments)
_F32_MT = None


def _f32_mt(m: int) -> int:
    if _F32_MT is not None:
        return next((mt for top, mt in _F32_MT if m <= top), 4)
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return next((mt for mt in (1, 2) if -(-m // (16 * mt)) <= sms), 4)


def _attention_infer(qkv: tuple, seqused, n: int, s: int, half_window: int, o32: bool) -> torch.Tensor:
    """O [N S, 128] of the inference attention: the window kernel on head-major Q / K / V (fp32 O for the fp32 block), FlashAttention-2
    on token-major ones (bf16 O)."""
    qh, kh, vh = qkv
    if not is_global(half_window):
        o = torch.empty(n * s, C, device=qh.device, dtype=_F32 if o32 else _BF)
        sm80._ext().swa_attn_fwd(
            qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2), seqused, o, torch.empty(0, device=qh.device, dtype=_F32)
        )
        return o
    cu, used_k = _varlen(seqused, s)
    o = torch.ops.aten._flash_attention_forward(qh, kh, vh, cu, cu, s, s, 0.0, False, False, scale=D**-0.5, seqused_k=used_k)[0]
    return o.reshape(n * s, C)


def _infer_bf16(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window: int) -> torch.Tensor:
    """The bf16 inference block with global attention: ``_fwd_bf16`` without saves, Q / K / V token-major for FlashAttention-2."""
    n, s, _ = q.shape
    qf = q.reshape(n * s, C)
    qcfg, fcfg = sm80.infer_cfg(n * s)
    qh, kh, vh, g, *_ = sm80.qkvg_fwd(qf, mod, cos, sin, wqkv, wg, n, s, FP32_EPS, FP32_EPS, False, tm=True, **qcfg)
    o = _attention_infer((qh, kh, vh), seqused, n, s, half_window, False)
    return sm80.ffn_fwd(qf, g, o, mod, wo, wu, wd, n, s, FP32_EPS, False, cfg=fcfg)[0].view(n, s, C)


def _infer_fp32(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window: int) -> torch.Tensor:
    """The fp32 inference block on the TF32 row kernels: qkvg -> attention -> out-projection + FFN (``_fwd_fp32``'s statements)."""
    from miniworld_engine.kernels.swa_dit.dispatch import eval_form

    n, s, _ = q.shape
    m = n * s
    dev = q.device
    w1, wop, wup, wdp = eval_form("swa_a100_fp32_tf32_packs", (wqkv, wg, wo, wu, wd), lambda: _fp32_packs(wqkv, wg, wo, wu, wd))
    glob = is_global(half_window)
    qkv = [torch.empty(*((m, H, D) if glob else (n, H, s, D)), device=dev, dtype=_BF) for _ in range(3)]
    g = torch.empty(m, C, device=dev, dtype=_F32)
    mt = _f32_mt(m)
    sm80._ext().swa_qkvg_f32(q, mod, cos, sin, w1, *qkv, g, s, n, FP32_EPS, FP32_EPS, not glob, mt)
    o = _attention_infer(qkv, seqused, n, s, half_window, True)
    out = torch.empty(m, C, device=dev, dtype=_F32)
    sm80._ext().swa_ffn_f32(q, g, o, mod, wop, wup, wdp, out, s, n, FP32_EPS, min(mt, 2))
    return out.view(n, s, C)


# ------------------------------------------------------------------------------------------------------------------- modulation
def _mod_fwd(c: torch.Tensor, wmod: torch.Tensor, save: bool) -> list[torch.Tensor]:
    """``[mod]`` = silu(c) Wmod^T [R, 768] fp32 (with ``save`` also the backward's operand a = silu(c)): bf16 on ``sm80.mod_fwd``, fp32 as
    the H100 native modulation (``dispatch.native_h100_mod_fwd``: the silu copy, then cuBLAS)."""
    if c.dtype == _BF:
        return sm80.mod_fwd(c, wmod, save)
    a = torch.empty_like(c)
    sm80._ext().native_copy([c], [a], 1)
    return [_mm(a, wmod.t()), a]


def _mod_bwd(dmod: torch.Tensor, c: torch.Tensor, a: torch.Tensor, wmod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(dc in c's dtype, dWmod fp32)."""
    if c.dtype == _BF:
        return sm80.mod_bwd(dmod, c, a, wmod, dw_dtype=_F32)
    da, dw = _mm(dmod, wmod), _mm(dmod.t(), a)
    dc = torch.empty_like(c)
    sm80._ext().native_rows([c, da], [dc], 14, 1, 1, c.shape[0], False, FP32_EPS)
    return dc, dw


# ------------------------------------------------------------------------------------------------------------------- the block
def _prepare(x, cond, cos, sin, weights, save: bool = True):
    from miniworld_engine.kernels.swa_dit.dispatch import eval_form

    dt = x.dtype
    # the weights in x's dtype (casts of fp32 masters): once per weight version in inference (``save`` false), read by a capture as its warm-up
    # made them; a training call casts every time
    ws = eval_form(("swa_a100_composed", dt), weights, lambda: [w.detach().to(dt).contiguous() for w in weights], save)
    c = cond.reshape(-1, cond.shape[-1]).to(dt).contiguous()
    cos, sin = (t.reshape(-1, t.shape[-1]).contiguous() for t in (cos, sin))
    return x.contiguous(), c, cos, sin, ws


def block(x, cond, cos, sin, seqused, half_window: int, weights) -> torch.Tensor:
    """Inference: ``x`` / ``cond`` [N, S, 128] (bf16 or fp32, the same), ``cos`` / ``sin`` fp32 [N, S, 16], ``seqused`` [N] int32,
    ``weights`` = (Wmod, Wqkv, Wg, Wo, Wu, Wd) in x's dtype or fp32 -> [N, S, 128] in x's dtype."""
    x, c, cos, sin, ws = _prepare(x, cond, cos, sin, weights, save=False)
    with torch.autocast("cuda", enabled=False):
        mod = _mod_fwd(c, ws[0], False)[0]
        if _sm120.is_sm120(x.device):  # the RTX 5090: the row stages around cuBLAS (the fused stages' weight tiles exceed its 99 KB)
            return _fwd_fp32(x, mod, cos, sin, seqused, *ws[1:], half_window, False)[0]
        if x.dtype == _BF:
            if is_global(half_window):
                return _infer_bf16(x, mod, cos, sin, seqused, *ws[1:], half_window)
            return _fwd_bf16(x, mod, cos, sin, seqused, *ws[1:], half_window, False)[0]
        if torch.backends.cuda.matmul.allow_tf32:
            return _infer_fp32(x, mod, cos, sin, seqused, *ws[1:], half_window)
        return _fwd_fp32(x, mod, cos, sin, seqused, *ws[1:], half_window, False)[0]


class SWABlockA100(torch.autograd.Function):
    """:func:`block` with a backward: gradients of x, cond and the six weights (in each one's own dtype; an fp32 master weight gets the
    unrounded fp32 gradient). cos / sin / seqused get none."""

    @staticmethod
    def forward(ctx, x, cond, cos, sin, seqused, half_window, wmod, wqkv, wg, wo, wu, wd):
        raw = (wmod, wqkv, wg, wo, wu, wd)
        ctx.meta = (half_window, [w.dtype for w in raw], cond.shape, cond.dtype)
        x, c, cos, sin, ws = _prepare(x, cond, cos, sin, raw)
        with torch.autocast("cuda", enabled=False):
            mod, a = _mod_fwd(c, ws[0], True)
            fwd = _fwd_bf16 if x.dtype == _BF and not _sm120.is_sm120(x.device) else _fwd_fp32
            outputs = fwd(x, mod, cos, sin, seqused, *ws[1:], half_window, True)
        ctx.save_for_backward(x, c, a, mod, cos, sin, seqused, *ws, *outputs[1:])
        return outputs[0]

    @staticmethod
    def backward(ctx, dy):
        half_window, dtypes, cond_shape, cond_dtype = ctx.meta
        x, c, a, mod, cos, sin, seqused, wmod, *rest = ctx.saved_tensors
        weights, saves = rest[:5], rest[5:]
        with torch.autocast("cuda", enabled=False):
            bwd = _bwd_bf16 if x.dtype == _BF and not _sm120.is_sm120(x.device) else _bwd_fp32
            dx, dmod, *dw = bwd(dy.to(x.dtype).contiguous(), x, mod, cos, sin, seqused, *weights, *saves, half_window)
            dc, dwmod = _mod_bwd(dmod, c, a, wmod)
        grads = [g.to(dt) for g, dt in zip((dwmod, *dw), dtypes, strict=True)]
        return dx, dc.view(cond_shape).to(cond_dtype), None, None, None, None, *grads
