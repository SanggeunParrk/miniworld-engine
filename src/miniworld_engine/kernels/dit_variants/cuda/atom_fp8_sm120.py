"""OPT-IN e4m3 GEMMs of the RTX 5090 (sm_120a) atom DiT inference block (docs/fp8.md "RTX 5090", atom DiT): the ``atom_dit`` slot of
``fp8.SM120_ROUTES``, the switch and interface of the H100 module ``integrations/local_dit_h100`` (``FP8_ROWS``, ``enable_fp8``, ``_fp8_parts``,
``STATS``)::

    MINIWORLD_ATOMDIT_FP8=1        # (MINIWORLD_FP8=safe / engine-fp8 set this) 1 / all: pre + post; pre; post

On the 5090 every atom DiT block (Protenix, OpenFold3, Boltz-2, RoseTTAFold3, Chai-1: ``modules/dit_variant_cuda.VariantAtomCuda`` /
``_LocalAtomDiT``) runs ``dit_variants/cuda/train.atom_stack_train``: the train_rows.cu row passes, cuBLAS for every projection (K = 128 / 256:
cuBLAS picks wmma / 64x64 kernels at 50-110 TFLOPS), the mma.sync window attention. Here the bf16 inference call of a block (no gradient, bf16
activations and conditioning table) runs :func:`block` instead of the autograd Function: the same algorithm, with

* ``pre``: the AdaLN rows of the query / key-value inputs written as e4m3 (``ln_mod_q8``, Protenix's chained K / V AdaLN included), q (bias in
  the GEMM), gate, k, v on cuBLASLt e4m3 with row-wise scales (``torch._scaled_mm``);
* ``post``: the gated core output (``gate_q8``), the transition AdaLN input (``res_ln_q8``, or the block input's AdaLN rows of the parallel
  block) and the SwiGLU hidden (``glu_q8``, Boltz-2's up gate: silu(a) b u) as e4m3: out projection, expand, squeeze on e4m3.

Scales: activations a DYNAMIC per-row scale (row max / 448), weights per output channel (max / 448, once per weight version): no calibration, no
guard (nothing saturates). The window attention, the pair bias / edge / tail windows, RoseTTAFold3's q / k LayerNorm and the residual passes stay
as they are. Training, fp32 and a conditioning table in fp32 run the default.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from pathlib import Path

import torch

from miniworld_engine.fp8 import env as _fp8_env

_dir = Path(__file__).parent
E4M3 = torch.float8_e4m3fn
BF = torch.bfloat16
_FP8_PARTS = ("pre", "post")
D = 128


def _fp8_parts(value: bool | str | Iterable[str] | None) -> frozenset:
    if value is True or value in ("1", "all"):
        return frozenset(_FP8_PARTS)
    if not value or value == "0":
        return frozenset()
    parts = frozenset(str(value).split(",")) if isinstance(value, str) else frozenset(value)
    if not parts <= set(_FP8_PARTS):
        raise ValueError(f"FP8 parts must be among {_FP8_PARTS}, got {sorted(parts)}")
    return parts


#: the e4m3 parts in force (``pre`` / ``post``; empty: the default)
FP8_ROWS: frozenset = _fp8_parts(_fp8_env("MINIWORLD_ATOMDIT_FP8", ""))
#: route call counters (``fp8.route_stats()["atom_dit"]``; plain ints at Python call time): ``calls`` = block calls on e4m3, ``pre`` / ``post`` of
#: those, ``fallbacks`` = block calls the route is on for that ran the default (``why_save``: a gradient, ``why_tf32``: fp32 activations or table,
#: ``why_rows``: below :data:`MIN_ROWS`); ``why_opts`` / ``why_cond`` kept for the H100 keys (never counted here: every option and per-sample
#: conditioning tables are served).
STATS = {"calls": 0, "pre": 0, "post": 0, "fallbacks": 0, "why_save": 0, "why_tf32": 0, "why_opts": 0, "why_cond": 0, "why_rows": 0}
#: the fewest block rows M = A N the route takes (RTX 5090, same-process A-B, 6 rounds, docs/fp8.md "RTX 5090" atom DiT): pre + post 0.98 at 1024
#: rows (A1 N1024, Protenix and OpenFold3), 1.08-1.13 from 4096 (A1 N4096, A5 N1024), 1.15-1.42 from 10240
MIN_ROWS = 4096
#: the same gate for ESMFold2's atom block (:func:`esm_block`; its 192-key window attention is cheaper, so the GEMMs are a smaller share): pre + post
#: 0.65 at 1024 rows, 0.94 at 4096 (A1), 0.99 at 5120 (A5 N1024), 1.17-1.20 at 10240 / 16384, 1.27-1.49 from 20480 (same-process A-B, 6 rounds)
ESM_MIN_ROWS = 8192


def enable_fp8(on: bool | str | Iterable[str] = True) -> frozenset:
    """Switch the e4m3 route: True / "all" (both parts), "pre", "post", an iterable of those, or False; returns the previous parts."""
    global FP8_ROWS
    prev, FP8_ROWS = FP8_ROWS, _fp8_parts(on)
    return prev


@functools.lru_cache(maxsize=1)
def ext():
    """The sm_120a extension ``atom_f8_sm120`` (the e4m3 row passes)."""
    from ..._nvcc import ensure_cuda_home, host_flags, load_extension
    from ..._sm120_builds import SM120

    ensure_cuda_home()
    return load_extension(
        name="atom_f8_sm120",
        sources=[str(_dir / "atom_f8_sm120.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", SM120, "-DMWE_SM120"],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def parts(x: torch.Tensor, tab: torch.Tensor, esm: bool = False) -> frozenset:
    """The parts this block call takes (and the counters): bf16 inference on an RTX 5090 with a bf16 conditioning table."""
    if not FP8_ROWS:
        return frozenset()
    from ..._sm120_builds import is_sm120

    if not (x.is_cuda and is_sm120(x.device)):
        return frozenset()
    why = None
    if torch.is_grad_enabled():
        why = "why_save"
    elif x.dtype is not BF or tab.dtype is not BF:
        why = "why_tf32"
    elif x.shape[0] < (ESM_MIN_ROWS if esm else MIN_ROWS):
        why = "why_rows"
    if why is not None:
        STATS["fallbacks"] += 1
        STATS[why] += 1
        return frozenset()
    STATS["calls"] += 1
    for p in FP8_ROWS:
        STATS[p] += 1
    return FP8_ROWS


def _w8(tag: str, *ws: torch.Tensor):
    """e4m3 [N, K] of cat(ws) as the column-major operand of torch._scaled_mm + fp32 [1, N] per-channel scales, once per weight version."""
    from miniworld_engine.kernels.integrations.adaln_sm80 import cached

    def build():
        w = (torch.cat(ws) if len(ws) > 1 else ws[0]).float()
        s = w.abs().amax(1).clamp_min(1e-30) / 448.0
        return (w / s[:, None]).to(E4M3).contiguous().t(), s.view(1, -1).contiguous()

    return cached(f"af8_{tag}", tuple(ws), build)


def _bf(tag: str, t: torch.Tensor) -> torch.Tensor:
    from miniworld_engine.kernels.integrations.adaln_sm80 import cached

    return t if t.dtype is BF else cached(f"af8_bf_{tag}", (t,), lambda: t.to(BF))


def _mm(x8, sx, w, bias=None):
    return torch._scaled_mm(x8, w[0], sx, w[1], bias=bias, out_dtype=BF)


def _q8(M: int, W: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty((M, W), device=dev, dtype=E4M3), torch.empty((M, 1), device=dev, dtype=torch.float32)


def block(*args) -> torch.Tensor:
    """:func:`_block` with autocast off (the e4m3 operands are its own; a caller's bf16 autocast must not recast the glue)."""
    with torch.autocast("cuda", enabled=False):
        return _block(*args)


def _block(x, tab, bias2, widx, keys, ebias, tinfo, tb, wq, bq, wk, wv, wg, wo, wexp, wsq, qkw, cfg, use) -> torch.Tensor:
    """``train.atom_fwd``'s output (the block's stream [A N, 128] bf16) at inference with the e4m3 parts ``use``; the arguments of ``_AtomBlock``
    (``wexp`` = [W_expand; W_up] parameters as a tuple)."""
    from . import train as T

    par, sig, gating, proj, kq, up, slide, length, eps, eps_qk, *rest = cfg
    chain = bool(rest and rest[0])
    core = bool(len(rest) > 1 and rest[1])
    rx, local, f8 = T._atom_ext(x), T._local(x), ext()
    M = x.shape[0]
    N, A = length, M // length
    TR = tab.shape[0]
    dev = x.device
    cols = lambda c: c * D  # noqa: E731
    pre, post = "pre" in use, "post" in use
    xt8 = sxt = xt = None
    # ---- the AdaLN rows of the attention inputs (+ the transition input of the parallel block)
    if pre:
        n_out = 3 if par and post else 2
        xq8, sxq = _q8(M, D, dev)
        xk8, sxk = _q8(M, D, dev)
        outs, scs = [xq8, xk8], [sxq, sxk]
        sc_c, sh_c = [cols(T.A_SC_Q), cols(T.A_SC_K)], [cols(T.A_SH_Q), cols(T.A_SH_K)]
        if n_out == 3:
            xt8, sxt = _q8(M, D, dev)
            outs.append(xt8), scs.append(sxt), sc_c.append(cols(T.A_SC_T)), sh_c.append(cols(T.A_SH_T))
        f8.ln_mod_q8(x, tab, sc_c, sh_c, outs, [s.view(-1) for s in scs], TR, eps, int(bool(sig)), chain and not par)
        if par and not post:
            xt = torch.empty_like(x)
            rx.ln_mod_fwd(x, tab, [cols(T.A_SC_T)], [cols(T.A_SH_T)], [xt], torch.empty(M, 2, device=dev), TR, eps, sig)
        q = _mm(xq8, sxq, _w8("q", wq), bias=_bf("bq", bq))
        k = _mm(xk8, sxk, _w8("k", wk))
        v = _mm(xk8, sxk, _w8("v", wv))
        g = _mm(xq8, sxq, _w8("g", wg)) if gating else None
    else:
        xq, xk = torch.empty_like(x), torch.empty_like(x)
        st = torch.empty(M, 2, device=dev)
        if par:
            xt = torch.empty_like(x)
            rx.ln_mod_fwd(x, tab, [cols(T.A_SC_Q), cols(T.A_SC_K), cols(T.A_SC_T)], [cols(T.A_SH_Q), cols(T.A_SH_K), cols(T.A_SH_T)], [xq, xk, xt], st, TR, eps, sig)
        elif chain:
            rx.ln_mod_fwd(x, tab, [cols(T.A_SC_Q)], [cols(T.A_SH_Q)], [xq], st, TR, eps, sig)
            rx.ln_mod_fwd(xq, tab, [cols(T.A_SC_K)], [cols(T.A_SH_K)], [xk], torch.empty(M, 2, device=dev), TR, eps, sig)
        else:
            rx.ln_mod_fwd(x, tab, [cols(T.A_SC_Q), cols(T.A_SC_K)], [cols(T.A_SH_Q), cols(T.A_SH_K)], [xq, xk], st, TR, eps, sig)
        q = torch.addmm(_bf("bq", bq), xq, _bf("wq", wq).t())
        k, v = torch.mm(xk, _bf("wk", wk).t()), torch.mm(xk, _bf("wv", wv).t())
        g = torch.mm(xq, _bf("wg", wg).t()) if gating else None
    if par and post and xt8 is None:  # the parallel block's transition input from the default AdaLN rows
        xt8, sxt = _q8(M, D, dev)
        f8.quant_q8(xt, xt8, sxt.view(-1))
    # ---- the attention core (unchanged)
    if kq:
        qn, kn = torch.empty_like(q), torch.empty_like(k)
        rx.qkv_prep(q, k, v, qn, kn, None, torch.empty(M, 4, device=dev), qkw[0], qkw[1], qkw[2], qkw[3], 1.0, eps_qk, True, False)
    else:
        qn, kn = q, k
    if core:
        b2 = T._b200_local()
        o, _ = b2.attn_fwd(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), bias2, widx if widx.numel() else None, slide=slide)
    else:
        o, _ = local.attn_fwd_mma(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), bias2)
    if slide:
        if not core:
            T._edge(x).edge_attn_fwd(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), keys, widx, ebias, o)
        T._edge(x).tail_attn_fwd(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), tinfo, tb, o)
    o = o.view(M, D)
    # ---- the attention output and the transition
    out = torch.empty_like(x)
    if post:
        if proj:
            og8, sog = _q8(M, D, dev)
            if gating:
                f8.gate_q8(o, g, og8, sog.view(-1))
            else:
                f8.quant_q8(o, og8, sog.view(-1))
            ya = _mm(og8, sog, _w8("o", wo))
        elif gating:
            ya = torch.empty_like(o)
            rx.gate_fwd(o, g, ya)
        else:
            ya = o
        if par:
            xp = x
        else:
            xp = torch.empty(M, D, device=dev, dtype=torch.float32)
            xt8, sxt = _q8(M, D, dev)
            f8.res_ln_q8(x, ya, tab, cols(T.A_G_A), cols(T.A_SC_T), cols(T.A_SH_T), xp, xt8, sxt.view(-1), TR, eps, int(bool(sig)))
        ab = _mm(xt8, sxt, _w8("exp", *wexp))
        h8, sh = _q8(M, 256, dev)
        f8.glu_q8(ab, h8, sh.view(-1), up)
        y2 = _mm(h8, sh, _w8("sq", wsq))
    else:
        if gating:
            og = torch.empty_like(o)
            rx.gate_fwd(o, g, og)
        else:
            og = o
        ya = torch.mm(og, _bf("wo", wo).t()) if proj else og
        if par:
            xp = x
        else:
            xp = torch.empty(M, D, device=dev, dtype=torch.float32)
            xt = torch.empty_like(x)
            rx.res_ln_fwd(x, ya, tab, cols(T.A_G_A), cols(T.A_SC_T), cols(T.A_SH_T), xp, xt, torch.empty(M, 2, device=dev), TR, eps, sig)
        ab = torch.mm(xt, _forms_cat("exp", wexp).t())
        h = torch.empty(M, 256, device=dev, dtype=BF)
        rx.glu_fwd(ab, h, up)
        y2 = torch.mm(h, _bf("wsq", wsq).t())
    if par:
        rx.res_out_fwd(x, ya, tab, cols(T.A_G_A), y2, cols(T.A_G_T), out, TR, False)
    else:
        rx.res_out_fwd(xp, y2, tab, cols(T.A_G_T), None, 0, out, TR, False)
    return out


def esm_block(*args) -> torch.Tensor:
    """:func:`_esm_block` with autocast off."""
    with torch.autocast("cuda", enabled=False):
        return _esm_block(*args)


def _esm_block(x, tab, bias2, cos, sin, wq, bq, wk, wv, wg, wo, wa, wb, wsq, length, tl, use) -> torch.Tensor:
    """ESMFold2's atom block (``AtomDiTESMFold2Pair``) at inference with the e4m3 parts ``use``: ``integrations.local_dit_esm._infer_a100`` (the
    block the RTX 5090 runs: train_rows' RMS mode, cuBLAS GEMMs, the q / k RMSNorm + rotary, the 192-key rank-window attention) with the rows
    that feed a GEMM written as e4m3 by this extension in its RMS mode (``mode`` 2: RMSNorm statistics, raw residual gate), as :func:`_block` does
    for the other classes: ``pre`` = the AdaLN rows of q / gate and k / v inputs + q (bias in the GEMM), gate, k, v on e4m3; ``post`` = the gated
    core output, the transition input (xp = x + gA ya in fp32, RMSNorm(xp) sc + sh as e4m3) and the SwiGLU hidden on e4m3 (out, expand [Wa; Wb],
    squeeze). The q / k norm + rotary, the window attention, the tables and the pair bias are the default's. Returns the block stream [A N, 128]."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local
    from miniworld_engine.kernels.integrations import local_dit_esm as E

    rx, qk, f8 = E._rows(x.device), E._qk_kernels(x.device), ext()
    M = x.shape[0]
    N, A = length, M // length
    TR = tl or N
    dev = x.device
    eps, mode = E.RMS_EPS, E._RMS_MODE
    cols = lambda c: c * D  # noqa: E731
    pre, post = "pre" in use, "post" in use
    if pre:
        xq8, sxq = _q8(M, D, dev)
        xk8, sxk = _q8(M, D, dev)
        f8.ln_mod_q8(x, tab, [cols(E.A_SC_Q), cols(E.A_SC_K)], [cols(E.A_SH_Q), cols(E.A_SH_K)], [xq8, xk8], [sxq.view(-1), sxk.view(-1)], TR, eps, mode, False)
        q = _mm(xq8, sxq, _w8("esm_q", wq), bias=_bf("esm_bq", bq))
        k = _mm(xk8, sxk, _w8("esm_k", wk))
        v = _mm(xk8, sxk, _w8("esm_v", wv))
        g = _mm(xq8, sxq, _w8("esm_g", wg))
    else:
        xq, xk = torch.empty_like(x), torch.empty_like(x)
        rx.ln_mod_fwd(x, tab, [cols(E.A_SC_Q), cols(E.A_SC_K)], [cols(E.A_SH_Q), cols(E.A_SH_K)], [xq, xk], torch.empty(M, 2, device=dev), TR, eps, mode)
        q = torch.addmm(_bf("esm_bq", bq), xq, _bf("esm_wq", wq).t())
        k, v = torch.mm(xk, _bf("esm_wk", wk).t()), torch.mm(xk, _bf("esm_wv", wv).t())
        g = torch.mm(xq, _bf("esm_wg", wg).t())
    qn, kn = torch.empty_like(q), torch.empty_like(k)
    one = E._const(dev, 1.0, D // E.NH)
    qk.qknorm_rope_fwd(q, k, one, one, eps, eps, cos, sin, qn, kn)
    o, _lse = local.attn_esm_fwd_lse(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), bias2)
    o = o.view(M, D)
    xp = torch.empty(M, D, device=dev, dtype=torch.float32)
    if post:
        og8, sog = _q8(M, D, dev)
        f8.gate_q8(o, g, og8, sog.view(-1))
        ya = _mm(og8, sog, _w8("esm_o", wo))
        xt8, sxt = _q8(M, D, dev)
        f8.res_ln_q8(x, ya, tab, cols(E.A_G_A), cols(E.A_SC_T), cols(E.A_SH_T), xp, xt8, sxt.view(-1), TR, eps, mode)
        ab = _mm(xt8, sxt, _w8("esm_exp", wa, wb))
        h8, sh = _q8(M, 256, dev)
        f8.glu_q8(ab, h8, sh.view(-1), False)
        y2 = _mm(h8, sh, _w8("esm_sq", wsq))
    else:
        og = torch.empty_like(o)
        rx.gate_fwd(o, g, og)
        ya = torch.mm(og, _bf("esm_wo", wo).t())
        xt = torch.empty_like(x)
        rx.res_ln_fwd(x, ya, tab, cols(E.A_G_A), cols(E.A_SC_T), cols(E.A_SH_T), xp, xt, torch.empty(M, 2, device=dev), TR, eps, mode)
        ab = torch.mm(xt, _forms_cat("esm_exp", (wa, wb)).t())
        h = torch.empty(M, 256, device=dev, dtype=BF)
        rx.glu_fwd(ab, h, False)
        y2 = torch.mm(h, _bf("esm_wsq", wsq).t())
    out = torch.empty_like(x)
    rx.res_out_fwd(xp, y2, tab, cols(E.A_G_T), None, 0, out, TR, True)
    return out


def _forms_cat(tag: str, ws) -> torch.Tensor:
    from miniworld_engine.kernels.integrations.adaln_sm80 import cached

    return cached(f"af8_cat_{tag}", tuple(ws), lambda: torch.cat(ws).to(BF))


def prebuild_items():
    yield ext


__all__ = ["ESM_MIN_ROWS", "FP8_ROWS", "MIN_ROWS", "STATS", "block", "enable_fp8", "esm_block", "ext", "parts"]
