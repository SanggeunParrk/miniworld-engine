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

import os

import torch

from foldjet_engine.kernels import _sm120
from foldjet_engine.kernels.swa_dit._common import FP32_EPS, is_global
from foldjet_engine.kernels.swa_dit.cuda import sm80

C, H, D = sm80.C, sm80.H, sm80.D
_BF, _F32 = torch.bfloat16, torch.float32
#: RTX 5090 bf16 window inference on the fused stages within 99 KB (``sm80.block_infer_sm120``); False: the composed row stages (``_fwd_fp32``)
FUSED_SM120 = True
#: RTX 5090 bf16 training forward on the fused stages (``sm80.block_infer_sm120`` with ``save``: the saves of the backward written by the stages themselves) from this many rows; None: the composed row stages
TRAIN_FUSED_SM120_MIN_ROWS: int | None = 4096
#: RTX 5090 bf16 training backward: the FFN backward's gate kernel and the out-projection backward of the sm_80 stages (``sm80.ffn_bwd_gate_sm120`` / ``oproj_bwd``) in place of the composed row stages around cuBLAS products
BWD_FUSED_SM120 = True
#: RTX 5090 training backward: the FFN backward's dy kernel and the qkvg backward kernel with their weights STREAMED (``bwd_ks_sm120.cuh``; the sm_80 kernels keep 128 KB resident) in place of
#: the cuBLAS products and row stages around them
BWD_STREAM_SM120 = True
#: RTX 5090 training backward: the five weight-gradient GEMMs as one grouped kernel (``sm80.wgrad5_sm120``, ``wgrad5_sm120.cu``) in place of five cuBLAS products; needs the row count a multiple of 256
WGRAD5_SM120 = True
#: ... up to this many rows (measured: 48 samples x 2048 .. 8192 rows, 98304 .. 393216: cuBLAS and the kernel are level)
WGRAD5_MAX_ROWS_SM120 = 16384
#: the out-projection backward kernel's launch shape on the RTX 5090 (``sm80.oproj_bwd`` cfg: warps per CTA 8 / 4 / 2 / 1 for 0 / 1 / 2 / 3)
OPROJ_BWD_CFG_SM120 = 1
#: RTX 5090: the bf16 modulation backward on cuBLAS products (``_mod_bwd``) at every row count (the A100 table: up to 3072 rows)
MOD_BWD_GEMM_SM120 = True
#: RTX 5090 global-attention inference on ``attn_global_sm120.cuh`` (False: FlashAttention-2); one sample from this many rows stays on FlashAttention-2 (measured: 8192 rows 270 vs 254 us)
GLOBAL_SM120 = True
GLOBAL_SM120_SINGLE_MAX = 8192
#: RTX 5090 global-attention TRAINING on the sm_120a forward with lse and backward of ``attn_global_sm120.cuh`` (False: FlashAttention-2 fwd / bwd)
GLOBAL_TRAIN_SM120 = True
#: ... for sequences up to this length (graph replays, S 1024 / 2048 / 4096 / 5120: forward + backward 48 / 95 / 309 / 383 us against FlashAttention-2 69 / 125 / 298 / 371: the backward kernel loses from 4096)
GLOBAL_TRAIN_SM120_MAX_S = 3072


def _ks_ok(m: int, s: int) -> bool:
    """The K-split backward kernels (``bwd_ks_sm120.cuh``) hold one tile group per SM: at most 4 row tiles of 16 per CTA."""
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return -(-(-(-s // 16) * (m // s)) // 4) <= sms


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


def attention_fwd(qh, kh, vh, seqused, half_window: int, out_dtype: torch.dtype = _BF, inference: bool = False, o_out=None, lse_out=None):
    """The attention of the block on bf16 head-major Q / K / V [N, 4, S, 32]: ``(O [N S, 128] bf16, lse)``, rows >= seqused 0. The
    window (half window 64) on the sm_80 kernel; global attention on FlashAttention-2 (its own lse layout, read back only by
    :func:`attention_bwd`). ``out_dtype`` fp32 (the window kernel only): O
    unrounded, as the fp32 block takes it. ``inference`` (no lse wanted; the RTX 5090): global attention on the sm_120a kernel of ``attn_global_sm120.cuh``
    (head-major planes read as they are, keys split over two warps; ``lse`` None), from the row counts where it beats FlashAttention-2."""
    if (
        inference
        and is_global(half_window)
        and _sm120.is_sm120(qh.device)
        and GLOBAL_SM120
        and (qh.shape[0] > 1 or qh.shape[2] < GLOBAL_SM120_SINGLE_MAX)
    ):
        n, _, s, _ = qh.shape
        o = torch.empty(n * s, C, device=qh.device, dtype=_BF)
        sm80._ext_sm120().swa_attn_global_fwd(qh, kh, vh, seqused.contiguous(), o, None)
        return o, None
    if (
        not inference
        and is_global(half_window)
        and out_dtype == _BF
        and GLOBAL_TRAIN_SM120
        and qh.shape[2] <= GLOBAL_TRAIN_SM120_MAX_S
        and _sm120.is_sm120(qh.device)
    ):
        n, _, s, _ = qh.shape  # training: the sm_120a kernel with its lse ([N, 4, S] natural log) for the backward of the same file
        o = torch.empty(n * s, C, device=qh.device, dtype=_BF)
        lse = torch.empty(n, H, s, device=qh.device, dtype=_F32)
        sm80._ext_sm120().swa_attn_global_fwd(qh, kh, vh, seqused.contiguous(), o, lse)
        return o, lse
    if not is_global(half_window):
        if out_dtype == _BF:
            return sm80.attn_fwd(qh, kh, vh, seqused)
        n, _, s, _ = qh.shape
        o = (
            o_out if o_out is not None else torch.empty(n * s, C, device=qh.device, dtype=_F32)
        )  # (o_out / lse_out: rows of a chunked call's buffers)
        lse = lse_out if lse_out is not None else torch.empty(n, H, s, device=qh.device, dtype=_F32)
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
    if GLOBAL_TRAIN_SM120 and _sm120.is_sm120(qh.device) and lse.dim() == 3:  # the lse of the sm_120a forward ([N, 4, S] fp32)
        dq, dk, dv = (torch.empty_like(qh) for _ in range(3))
        sm80._ext_sm120().swa_attn_global_bwd(qh, kh, vh, d_o, lse, delta, seqused.contiguous(), dq, dk, dv)
        return dq, dk, dv
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
    tc = sm80.train_cfg(n * s) if save else {}  # the training launch shapes (the same values)
    qh, kh, vh, g, xs, pqs, pks = sm80.qkvg_fwd(qf, mod, cos, sin, wqkv, wg, n, s, FP32_EPS, FP32_EPS, save, **tc.get("qkvg_fwd", {}))
    o, lse = attention_fwd(qh, kh, vh, seqused, half_window)
    out, q1s, atts, ys, ffs = sm80.ffn_fwd(qf, g, o, mod, wo, wu, wd, n, s, FP32_EPS, save, **tc.get("ffn_fwd", {}))
    outputs = [out.view(n, s, C)]
    if save:
        outputs += [qh, kh, vh, g, o, lse, q1s, xs, pqs, pks, atts, ys, ffs]
    return outputs


def _bwd_bf16(
    dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, half_window: int, forms=None
) -> list[torch.Tensor]:
    """``[dq, dmod, dWqkv, dWg, dWo, dWu, dWd]`` (dq bf16, the rest fp32) of :func:`_fwd_bf16` on the sm_80 backward stages; ``forms``: the
    transposed weights of :func:`train_forms` (else each stage transposes its own)."""
    f = forms or {}
    n, s, _ = q.shape
    m = n * s
    mode = sm80.MODE_SINGLE
    dmod = sm80.dmod_buffer(mode, n, n, s, q.device)
    tc = sm80.train_cfg(m)
    ffn_kw = dict(wdt=f.get("wdt"), wut=f.get("wut"), **tc.get("ffn_bwd", {}))
    dq1, dffn, hh, dab = sm80.ffn_bwd(dy.reshape(m, C), q1, y, ffn, mod, wu, wd, dmod, n, s, FP32_EPS, mode, **ffn_kw)
    d_o, dg, datt, gated, delta = sm80.oproj_bwd(dq1, o, g, att, mod, wo, dmod, n, s, mode, **tc.get("oproj_bwd", {}), wot=f.get("wot"))
    dqh, dkh, dvh = attention_bwd(qh, kh, vh, o, d_o, lse, delta, seqused, half_window)
    qkvg_args = (q.reshape(m, C), pq, pk, dqh, dkh, dvh, dg, dq1, mod, cos, sin, wqkv, wg, dmod, n, s, FP32_EPS, FP32_EPS, mode)
    dq, dp = sm80.qkvg_bwd(*qkvg_args, wt=f.get("wt"), **tc.get("qkvg_bwd", {}))
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
def _fwd_fp32(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window: int, save: bool, sink=None) -> list[torch.Tensor]:
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
    if bf and save and TRAIN_FUSED_SM120_MIN_ROWS is not None and m >= TRAIN_FUSED_SM120_MIN_ROWS and _sm120.is_sm120(dev):
        return sm80.block_infer_sm120(
            q, mod, cos, sin, seqused.contiguous(), wqkv, wg, wo, wu, wd, FP32_EPS, FP32_EPS, is_global(half_window), True
        )

    def empty(*shape, dtype=act, keep=None):
        if sink is not None and keep is not None:  # a saved tensor of a chunked call: its rows of the full buffer (``_fwd_fp32_chunked``)
            return sink(keep, shape, dtype)
        return torch.empty(shape, device=dev, dtype=dtype)

    x = empty(m, C, keep="x")
    _rows(0, [q, mod], [x], n, s, bf)
    from foldjet_engine.kernels.swa_dit.dispatch import eval_form

    wstack = eval_form("swa_a100_fp32_wstack", (wqkv, wg), lambda: torch.cat([wqkv, wg]), save)  # per weight version in inference
    p = mm(x, wstack.t())
    qh, kh, vh = (empty(n, H, s, D, dtype=_BF, keep=k) for k in ("qh", "kh", "vh"))
    g, pq, pk = (empty(m, C, keep=k) for k in ("g", "pq", "pk"))
    _rows(1, [p, cos, sin], [qh, kh, vh, g, pq, pk], n, s, bf)
    o_buf = (sink("o", (m, C), _F32), sink("lse", (n, H, s), _F32)) if sink is not None and save else (None, None)
    o, lse = attention_fwd(
        qh, kh, vh, seqused, half_window, act, inference=not save, o_out=o_buf[0], lse_out=o_buf[1]
    )  # fp32 O from the window kernel (global: bf16); bf16 O for bf16
    gated = empty(m, C)
    _rows(2, [o, g], [gated], n, s, bf)
    att = mm(gated, wo.t()) if sink is None else torch.mm(gated, wo.t(), out=sink("att", (m, C), act))
    q1, y = empty(m, C, keep="q1"), empty(m, C, keep="y")
    _rows(3, [q, att, mod], [q1, y], n, s, bf)
    ab = mm(y, wu.t())
    hh = empty(m, 2 * C)
    _rows(4, [ab], [hh], n, s, bf)
    ffn = mm(hh, wd.t()) if sink is None else torch.mm(hh, wd.t(), out=sink("ffn", (m, C), act))
    out = empty(m, C, keep="out")
    _rows(5, [q1, ffn, mod], [out], n, s, bf)
    outputs = [out.view(n, s, C)]
    if save:
        outputs += [qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn]
    return outputs


def _bwd_fp32(
    dy,
    q,
    mod,
    cos,
    sin,
    seqused,
    wqkv,
    wg,
    wo,
    wu,
    wd,
    qh,
    kh,
    vh,
    g,
    o,
    lse,
    q1,
    x,
    pq,
    pk,
    att,
    y,
    ffn,
    half_window: int,
    forms=None,
    dq_out=None,
    acc=None,
    dmod_out=None,
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

    dmod = dmod_out if dmod_out is not None else torch.empty(mod.shape, device=dev, dtype=_F32)
    f = forms or {}
    fused = (
        bf and BWD_FUSED_SM120 and "wdt" in f and "wot" in f and _sm120.is_sm120(dev)
    )  # the RTX 5090: the sm_80 kernels whose weights fit its 99 KB
    if fused:  # stages 6 + 7 and their two products: the FFN backward's gate kernel (a streamed 72 KB weight ring)
        dffn, hh, dab = sm80.ffn_bwd_gate_sm120(dy.reshape(m, C), y, ffn, mod, wu, f["wdt"], dmod, n, s)
    else:
        dffn = empty(m, C)
        _rows(6, [dy, mod, ffn], [dffn, dmod], n, s, bf)
        ab, dh = mm(y, wu.t()), mm(dffn, wd)
        dab, hh = empty(m, 4 * C), empty(m, 2 * C)
        _rows(7, [ab, dh], [dab, hh], n, s, bf)
    if (
        fused and BWD_STREAM_SM120 and _ks_ok(m, s)
    ):  # stage 8 and the product before it: dq1 and d shift_f | d scale_f from dab Wu, the weight streamed
        dq1 = sm80.ffn_bwd_dy_sm120(dab, q1, dy.reshape(m, C), mod, f["wut"], dmod, n, s, FP32_EPS)
    else:
        d_y = mm(dab, wu)
        dq1 = empty(m, C)
        _rows(8, [dy, d_y, q1, mod], [dq1, dmod], n, s, bf)
    if fused:  # stages 9 + 10 and the product between them: the out-projection backward (32 KB of Wo^T)
        d_o, dg, datt, gated, delta = sm80.oproj_bwd(
            dq1, o, g, att, mod, wo, dmod, n, s, sm80.MODE_SINGLE, OPROJ_BWD_CFG_SM120, wot=f["wot"]
        )
    else:
        datt, gated = empty(m, C), empty(m, C)
        _rows(9, [dq1, mod, att, o, g], [datt, gated, dmod], n, s, bf)
        dgated = mm(datt, wo)
        d_o, dg, delta = empty(m, C, dtype=_BF), empty(m, C), empty(n, H, s, dtype=_F32)
        _rows(10, [dgated, o, g], [d_o, dg, delta], n, s, bf)
    dqh, dkh, dvh = attention_bwd(qh, kh, vh, o, d_o, lse, delta, seqused, half_window)
    if (
        fused and BWD_STREAM_SM120 and "wt" in f and _ks_ok(m, s)
    ):  # stages 11 + 12 and the product between them: dP, dx = dP W and the adaLN backward, the weight streamed
        dq, dp = sm80.qkvg_bwd_sm120(
            q.reshape(m, C), pq, pk, dqh, dkh, dvh, dg, dq1, mod, cos, sin, f["wt"], dmod, n, s, FP32_EPS, FP32_EPS
        )
    else:
        dp = empty(m, 4 * C)
        _rows(11, [pq, pk, dqh, dkh, dvh, dg, cos, sin], [dp], n, s, bf)
        wstack = torch.cat([wqkv, wg])
        dx = mm(dp, wstack)
        dq = dq_out if dq_out is not None else empty(m, C)
        _rows(12, [dq1, dx, q, mod], [dq, dmod], n, s, bf)
    if (
        fused and WGRAD5_SM120 and m <= WGRAD5_MAX_ROWS_SM120 and sm80.wgrad5_ok_sm120(m)
    ):  # the five weight gradients in one grouped split-K GEMM + one reduction (cuBLAS runs them at ~40 TFLOPs there)
        dwqkv, dwg, dwo, dwu, dwd = sm80.wgrad5_sm120(dp, x, dab, y, datt, gated, dffn, hh)
        return [dq.view(n, s, C), dmod, dwqkv, dwg, dwo, dwu, dwd]

    def wgrad(i: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # acc: the running sums of a chunked call (beta 1 in the product)
        if acc is None:
            return _mm(a, b)
        return torch.addmm(acc[i], a, b, out=acc[i])

    return [
        dq.view(n, s, C),
        dmod,
        wgrad(0, dp[:, : 3 * C].t(), x),
        wgrad(1, dp[:, 3 * C :].t(), x),
        wgrad(2, datt.t(), gated),
        wgrad(3, dab.t(), y),
        wgrad(4, dffn.t(), hh),
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


#: RTX 5090 fp32 inference: the out-projection + FFN kernel streams its weight packs as fp16 (the tf32-rounded values, exact inside fp16's range; ``_fp16_packs``): half the bytes
#: through L2 for every 16- / ``FFN_F16W_MT * 16``-row tile (s5090/four).  False: the fp32 packs.
FFN_F16W_SM120 = os.environ.get("MWE_SWA_FFN_F16W", "1") != "0"
FFN_F16W_MT = int(
    os.environ.get("MWE_SWA_FFN_MT", "0")
)  # 0: 32-row tiles when there are more 16-row tiles than SMs (one weight pass per tile: fewer passes), else 16-row


def _ffn_mt(m: int) -> int:
    if FFN_F16W_MT in (1, 2):
        return FFN_F16W_MT
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return 2 if -(-m // 16) > sms else 1


def _fp16_packs(wop, wup, wdp) -> list[torch.Tensor] | None:
    """The three tf32 packs of the FFN stage as fp16 (exact: tf32 and fp16 both carry 10 mantissa bits), or None when a magnitude is beyond fp16's range (the fp32 packs
    serve) or the pack is built under a stream capture (the eager warm-up builds it first, a graph replays its result)."""
    if torch.cuda.is_current_stream_capturing():
        return None
    if not all(bool((p.abs().max() <= 6.0e4).item()) for p in (wop, wup, wdp)):
        return None
    return [p.to(torch.float16).contiguous() for p in (wop, wup, wdp)]


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


def _infer_fp32(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window: int, sm120: bool = False) -> torch.Tensor:
    """The fp32 inference block on the TF32 row kernels: qkvg -> attention -> out-projection + FFN (``_fwd_fp32``'s statements).  ``sm120``: the RTX 5090's
    build of the two kernels (``fused_sm120.cu``: weight rings two chunks deep, the FFN on 16-row tiles) within its 99 KB."""
    from foldjet_engine.kernels.swa_dit.dispatch import eval_form

    n, s, _ = q.shape
    m = n * s
    dev = q.device
    w1, wop, wup, wdp = eval_form("swa_a100_fp32_tf32_packs", (wqkv, wg, wo, wu, wd), lambda: _fp32_packs(wqkv, wg, wo, wu, wd))
    glob = is_global(half_window)
    qkv = [torch.empty(*((m, H, D) if glob else (n, H, s, D)), device=dev, dtype=_BF) for _ in range(3)]
    g = torch.empty(m, C, device=dev, dtype=_F32)
    mt = _f32_mt(m)
    ext = sm80._ext_sm120() if sm120 else sm80._ext()
    ext.swa_qkvg_f32(q, mod, cos, sin, w1, *qkv, g, s, n, FP32_EPS, FP32_EPS, not glob, mt)
    o = _attention_infer(qkv, seqused, n, s, half_window, True)
    out = torch.empty(m, C, device=dev, dtype=_F32)
    if sm120:
        h16 = eval_form("swa_sm120_fp32_f16_packs", (wo, wu, wd), lambda: _fp16_packs(wop, wup, wdp)) if FFN_F16W_SM120 else None
        if h16 is not None:
            ext.swa_ffn_f32h(q, g, o, mod, *h16, out, s, n, FP32_EPS, _ffn_mt(m))
        else:
            ext.swa_ffn_f32(q, g, o, mod, wop, wup, wdp, out, s, n, FP32_EPS)
    else:
        ext.swa_ffn_f32(q, g, o, mod, wop, wup, wdp, out, s, n, FP32_EPS, min(mt, 2))
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
    """(dc in c's dtype, dWmod fp32). bf16: the fused sm_80 kernels (``sm80.mod_bwd``: g as two bf16 terms), or for a call of few rows
    (``sm80.train_cfg``: ``mod_bwd_gemm``) the fp32 path's form -- cuBLAS products on fp32 copies of the bf16 operands (TF32 as torch's matmul flag
    says) and the silu backward on the bf16-rounded gradient (row stage 14, the fused kernel's rounding point): the fused kernels' dc tiles and
    dWmod partial sums run on a few CTAs there (L 1024: 40 us of a 270 us step)."""
    tc = sm80.train_cfg(c.shape[0])
    bf = c.dtype == _BF
    gemm = tc.get("mod_bwd_gemm") or (
        bf and MOD_BWD_GEMM_SM120 and _sm120.is_sm120(c.device)
    )  # the RTX 5090: cuBLAS at every row count (L 5120: 12 % of the step)
    if bf and not gemm:
        return sm80.mod_bwd(dmod, c, a, wmod, dw_dtype=_F32, **tc.get("mod_bwd", {}))
    if bf:  # the gemm form
        w32, a32 = torch.empty(wmod.shape, device=c.device, dtype=_F32), torch.empty(a.shape, device=c.device, dtype=_F32)
        sm80._ext().native_copy([wmod, a], [w32, a32], 0)  # exact: bf16 values in fp32
        wmod, a = w32, a32
    da, dw = _mm(dmod, wmod), _mm(dmod.t(), a)
    dc = torch.empty_like(c)
    sm80._ext().native_rows([c, da], [dc], 14, 1, 1, c.shape[0], bf, FP32_EPS)
    return dc, dw


# ------------------------------------------------------------------------------------------------------------------- the block
def _prepare(x, cond, cos, sin, weights, save: bool = True):
    from foldjet_engine.kernels.swa_dit.dispatch import eval_form

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
        if _sm120.is_sm120(x.device):
            glob = is_global(half_window)
            if (
                x.dtype == _BF and (not glob or GLOBAL_SM120) and FUSED_SM120 and x.shape[0] * x.shape[1] >= sm80.FUSED_MIN_ROWS_SM120
            ):  # the fused bf16 stages within 99 KB (a global window: around attn_global_sm120.cuh)
                return sm80.block_infer_sm120(x, mod, cos, sin, seqused.contiguous(), *ws[1:], FP32_EPS, FP32_EPS, glob)
            if (
                x.dtype == _F32 and not is_global(half_window) and torch.backends.cuda.matmul.allow_tf32 and FUSED_SM120
            ):  # the TF32 row kernels with shallower weight rings
                return _infer_fp32(x, mod, cos, sin, seqused, *ws[1:], half_window, sm120=True)
            # the row stages around cuBLAS (the A100 fused stages' weight tiles exceed its 99 KB)
            return _fwd_fp32(x, mod, cos, sin, seqused, *ws[1:], half_window, False)[0]
        if x.dtype == _BF:
            if is_global(half_window):
                return _infer_bf16(x, mod, cos, sin, seqused, *ws[1:], half_window)
            return _fwd_bf16(x, mod, cos, sin, seqused, *ws[1:], half_window, False)[0]
        if torch.backends.cuda.matmul.allow_tf32:
            return _infer_fp32(x, mod, cos, sin, seqused, *ws[1:], half_window)
        return _fwd_fp32(x, mod, cos, sin, seqused, *ws[1:], half_window, False)[0]


def train_forms(weights) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """The weight forms of a bf16 training call in ONE launch (``weight_forms``, ops.cu): the six weights (Wmod, Wqkv, Wg, Wo, Wu, Wd; fp32
    masters or bf16) cast to bf16 for the forward stages, and the transposes the backward stages read -- Wo^T, Wd^T, Wu^T, (Wqkv | Wg)^T -- from the
    same weights (rounding and transposing commute: the values of the per-stage casts and transposes, bitwise). At L 1024 the six casts and four
    transposes were 12 launches, ~30 us of a ~250 us training step."""
    dev = weights[0].device
    raw = [w.detach().contiguous() for w in weights]
    wmod, wqkv, wg, wo, wu, wd = raw
    ws = [torch.empty(w.shape, device=dev, dtype=_BF) for w in raw]
    forms = {
        "wot": torch.empty(wo.shape[1], wo.shape[0], device=dev, dtype=_BF),
        "wdt": torch.empty(wd.shape[1], wd.shape[0], device=dev, dtype=_BF),
        "wut": torch.empty(wu.shape[1], wu.shape[0], device=dev, dtype=_BF),
        "wt": torch.empty(C, 4 * C, device=dev, dtype=_BF),
    }
    xs = [*raw, wo, wd, wu, wqkv, wg]
    ys = [*ws, forms["wot"], forms["wdt"], forms["wut"], forms["wt"], forms["wt"]]
    cols = [0] * 6 + [wo.shape[1], wd.shape[1], wu.shape[1], C, C]
    ld = [0] * 6 + [wo.shape[0], wd.shape[0], wu.shape[0], 4 * C, 4 * C]
    off = [0] * 6 + [0, 0, 0, 0, 3 * C]
    sm80._ext().weight_forms(xs, ys, cols, ld, off)
    return ws, forms


#: RTX 5090, fp32 training: the stages of a call of more rows than this run on chunks of whole samples of about this many rows. The stage-to-stage tensors of a chunk
#: (100 MB each at 196608 rows) stay in the 96 MB L2 then (measured, window 128 float32 train, ns a row: 24576 rows 52.3, 49152 56.3, 196608 57.1)
CHUNK_ROWS_SM120 = 24576
CHUNK_MIN_ROWS_SM120 = 32768
_FWD_SAVES = ("qh", "kh", "vh", "g", "o", "lse", "q1", "x", "pq", "pk", "att", "y", "ffn")


def _chunk_samples(x: torch.Tensor, half_window: int) -> int | None:
    """Samples per chunk of the fp32 training stages on the RTX 5090 (window attention: FlashAttention-2's varlen lse of a global window is not per sample), or None (one call over every sample)."""
    n, s, _ = x.shape
    if x.dtype != _F32 or is_global(half_window) or not CHUNK_ROWS_SM120 or not _sm120.is_sm120(x.device) or n * s < CHUNK_MIN_ROWS_SM120:
        return None
    nsub = max(1, CHUNK_ROWS_SM120 // s)
    return nsub if nsub < n else None


def _cut(t: torch.Tensor, n: int, s: int, n0: int, nc: int) -> torch.Tensor:
    """The samples n0 .. n0 + nc of a tensor whose leading dimension is the samples (``n``) or the rows (``n s``: modulation, cos / sin, the row saves)."""
    return t[n0 : n0 + nc] if t.shape[0] == n else t[n0 * s : (n0 + nc) * s]


def _fwd_fp32_chunked(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window: int, nsub: int) -> list[torch.Tensor]:
    """:func:`_fwd_fp32` (training) over chunks of ``nsub`` samples, the saved tensors written into their rows of one buffer each (same outputs row for row)."""
    n, s, _ = q.shape
    bufs: dict[str, torch.Tensor] = {}
    at = {"n0": 0, "nc": 0}

    def sink(name: str, shape, dtype) -> torch.Tensor:
        per = shape[0] // at["nc"]  # leading dim: samples (1 a sample) or rows (s a sample)
        buf = bufs.get(name)
        if buf is None:
            buf = bufs[name] = torch.empty((per * n, *shape[1:]), device=q.device, dtype=dtype)
        return buf[at["n0"] * per : (at["n0"] + at["nc"]) * per]

    for n0 in range(0, n, nsub):
        at["n0"], at["nc"] = n0, min(nsub, n - n0)
        cut = [_cut(t, n, s, n0, at["nc"]) for t in (q, mod, cos, sin, seqused)]
        _fwd_fp32(*cut, wqkv, wg, wo, wu, wd, half_window, True, sink)
    return [bufs.pop("out").view(n, s, C), *(bufs[k] for k in _FWD_SAVES)]


def _bwd_fp32_chunked(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, *saves, half_window: int, nsub: int) -> list[torch.Tensor]:
    """:func:`_bwd_fp32` over chunks of ``nsub`` samples: dq written into its rows, dmod per row, the five weight gradients summed over the chunks (beta 1)."""
    n, s, _ = q.shape
    dq = torch.empty(q.shape, device=q.device, dtype=_F32)
    dmod = torch.empty(mod.shape, device=q.device, dtype=_F32)
    shapes = [(3 * C, C), (C, C), (C, C), (4 * C, C), (C, 2 * C)]  # dWqkv, dWg, dWo, dWu, dWd
    acc = [torch.zeros(sh, device=q.device, dtype=_F32) for sh in shapes]
    for n0 in range(0, n, nsub):
        nc = min(nsub, n - n0)
        cut = [_cut(t, n, s, n0, nc) for t in (dy, q, mod, cos, sin, seqused)]
        cs = [_cut(t, n, s, n0, nc) for t in saves]  # per-sample (qh, kh, vh, lse) or per-row (the rest) saves
        _bwd_fp32(*cut, wqkv, wg, wo, wu, wd, *cs, half_window, None, _cut(dq, n, s, n0, nc).view(nc * s, C), acc, _cut(dmod, n, s, n0, nc))
    return [dq, dmod, *acc]


class SWABlockA100(torch.autograd.Function):
    """:func:`block` with a backward: gradients of x, cond and the six weights (in each one's own dtype; an fp32 master weight gets the
    unrounded fp32 gradient). cos / sin / seqused get none."""

    @staticmethod
    def forward(ctx, x, cond, cos, sin, seqused, half_window, wmod, wqkv, wg, wo, wu, wd):
        raw = (wmod, wqkv, wg, wo, wu, wd)
        ctx.meta = (half_window, [w.dtype for w in raw], cond.shape, cond.dtype)
        forms: dict[str, torch.Tensor] = {}
        if x.dtype == _BF and all(
            w.shape[-1] == C for w in raw[:5]
        ):  # bf16: the casts and the backward's transposes in one launch (the RTX 5090: its composed backward reads no transposed form)
            ws, forms = train_forms(raw)
            x, c, cos, sin, _ = _prepare(x, cond, cos, sin, ())
        else:
            x, c, cos, sin, ws = _prepare(x, cond, cos, sin, raw)
        with torch.autocast("cuda", enabled=False):
            mod, a = _mod_fwd(c, ws[0], True)
            fwd = _fwd_bf16 if x.dtype == _BF and not _sm120.is_sm120(x.device) else _fwd_fp32
            nsub = _chunk_samples(x, half_window)
            if nsub is not None:
                outputs = _fwd_fp32_chunked(x, mod, cos, sin, seqused, *ws[1:], half_window, nsub)
            else:
                outputs = fwd(x, mod, cos, sin, seqused, *ws[1:], half_window, True)
        ctx.form_keys = tuple(forms)
        ctx.save_for_backward(x, c, a, mod, cos, sin, seqused, *ws, *outputs[1:], *forms.values())
        return outputs[0]

    @staticmethod
    def backward(ctx, dy):
        half_window, dtypes, cond_shape, cond_dtype = ctx.meta
        x, c, a, mod, cos, sin, seqused, wmod, *rest = ctx.saved_tensors
        nf = len(ctx.form_keys)
        weights, saves = rest[:5], rest[5 : len(rest) - nf]
        forms = dict(zip(ctx.form_keys, rest[len(rest) - nf :], strict=True))
        with torch.autocast("cuda", enabled=False):
            args = (dy.to(x.dtype).contiguous(), x, mod, cos, sin, seqused, *weights, *saves, half_window)
            if x.dtype == _BF and not _sm120.is_sm120(x.device):
                dx, dmod, *dw = _bwd_bf16(*args, forms=forms)
            elif (nsub := _chunk_samples(x, half_window)) is not None:  # the RTX 5090, fp32: chunks of samples (``_fwd_fp32_chunked``)
                dx, dmod, *dw = _bwd_fp32_chunked(*args[:-1], half_window=half_window, nsub=nsub)
            else:  # fp32, or the RTX 5090 (the bf16 stages keep 128 KB of weights resident: the composed fp32-structured path, bf16 rounded)
                dx, dmod, *dw = _bwd_fp32(*args, forms=forms)
            dc, dwmod = _mod_bwd(dmod, c, a, wmod)
        grads = [g.to(dt) for g, dt in zip((dwmod, *dw), dtypes, strict=True)]
        return dx, dc.view(cond_shape).to(cond_dtype), None, None, None, None, *grads
