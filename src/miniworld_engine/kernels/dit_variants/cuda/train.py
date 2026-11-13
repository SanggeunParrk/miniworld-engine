"""The A100 (sm_80) path of the ATOM DiT blocks (``modules.atom_dit``; every A100 atom call, inference included), on the sm_80 builds of
``cuda/sm80.py`` (H100 runs the one atom DiT implementation, ``kernels/integrations/local_dit_h100``): hand-written CUDA row kernels (``train_rows.cu``, forward and backward), cuBLAS for every GEMM and the local DiT's window attention, in an
``autograd.Function`` per block whose forward and backward are each one opaque op.

What the block Function owns is the work over the A x N sample rows; everything on the N atom rows (the conditioning tables, the pair bias)
is made by ordinary torch ops outside it, so autograd chains the gradients of the tables / bias / parameters through them. The residual
stream is fp32 inside a sequential block, every reduction fp32, the saved activations and GEMM operands bf16 (or fp32 with TF32 GEMMs).

The token blocks (TokenDiTBoltz2 / RoseTTAFold3 / Chai1) are not here: they run the one token DiT algorithm, TokenDiTProtenix's ``"cuda"``
route (``kernels/integrations/token_dit_train`` with the model's options).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque

LOG2E = 1.4426950408889634
#: rows per block of the dq / dk / dv backward kernel (its column sums leave as one partial row per block)
QKV_RPB = 16


def _mm_t(a, b):
    """a^T b (a, b [M, *]) in fp32: bf16 products with fp32 accumulation (never rounded to bf16), TF32 under the caller's allow_tf32."""
    if a.dtype == torch.bfloat16:
        return torch.mm(a.t(), b, out_dtype=torch.float32)
    return torch.mm(a.t(), b)


# ====================================================================================================================== atom stack
# The atom DiT blocks (128 wide, 4 heads x 32, 32 queries x 128 keys per window): the same row kernels at D = 128, the local DiT's mma.sync
# window attention (``attn_fwd_mma`` / ``attn_bwd_mma``), cuBLAS. Per block the tables are [N, 8 D]: q-AdaLN scale | shift, k-AdaLN scale |
# shift, attention gate, transition scale | shift | gate. The slid key window's five edge windows (the only ones whose keys differ from the
# clipped window the kernel reads, with the tail of windows past the last real one) are redone over their own keys on edge.cu, forward and
# backward.
ATOM_D, ATOM_H, ATOM_DH, ATOM_WQ, ATOM_WK = 128, 4, 32, 32, 128
A_SC_Q, A_SH_Q, A_SC_K, A_SH_K, A_G_A, A_SC_T, A_SH_T, A_G_T = range(8)
ATOM_SPLITS = 2
_ANAMES = (
    "out",
    "xq",
    "xk",
    "xt",
    "q",
    "k",
    "v",
    "qn",
    "kn",
    "stqk",
    "g",
    "og",
    "lse",
    "ya",
    "y2",
    "xp",
    "st1",
    "st2",
    "ab",
    "h",
    "w_gq",
    "w_kv",
    "w_o",
    "w_exp",
    "w_sq",
    "stk",
)
_ASAVED = _ANAMES[1:]


def _atom_ext(t: torch.Tensor):
    """The training row kernels (``train_rows.cu``, sm_80)."""
    from miniworld_engine.kernels.dit_variants.cuda import sm80

    return sm80.train_ext()


def _local(t: torch.Tensor):
    """The window attention / pair bias kernels: the sm_80 builds of ``sm100_atom_local``'s mma.sync wrappers."""
    from miniworld_engine.kernels.dit_variants.cuda import sm80

    return sm80.LOCAL


def _edge(t: torch.Tensor):
    """The slid edge window kernels (``edge.cu``, sm_80)."""
    from miniworld_engine.kernels.dit_variants.cuda import sm80

    return sm80.edge_ext()


class _EdgeBias(torch.autograd.Function):
    """The slid edge windows' bias [5, 4, 32, 128] (natural units, -inf off the valid keys) from the trunked pair z: LN(z[widx]) gamma . Wb."""

    @staticmethod
    def forward(ctx, z, gamma, wb, widx, valid, eps):  # noqa: D102
        ctx.save_for_backward(z, gamma, wb, widx)
        ctx.eps = eps
        return _edge(z).edge_bias_fwd(z.index_select(0, widx).contiguous(), gamma.float().contiguous(), wb.float().contiguous(), valid, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, deb):  # noqa: D102
        z, gamma, wb, widx = ctx.saved_tensors
        dz_e, part = _edge(z).edge_bias_bwd(
            z.index_select(0, widx).contiguous(), gamma.float().contiguous(), wb.float().contiguous(), deb.contiguous(), ctx.eps
        )
        dz = torch.zeros_like(z)
        dz.index_add_(0, widx, dz_e)
        tot = part.sum(0)
        return dz, tot[:16].to(gamma.dtype), tot[16:].view(4, 16).to(wb.dtype), None, None, None


class _TailBias(torch.autograd.Function):
    """The slid window's tail bias tb [1, nwin, 4, 32, 128] (natural units, -inf off the valid keys; ``local_dit_h100.tail_info``: the query
    windows past the last real one, which all read the same 128 keys) from the trunked pair z: LN(z[w]) gamma . Wb for the windows from t0 on
    (``tail_bias_fwd``; the rows below t0 are not written and never read)."""

    @staticmethod
    def forward(ctx, z, gamma, wb, info, eps):  # noqa: D102
        ctx.save_for_backward(z, gamma, wb, info)
        ctx.eps = eps
        tb = torch.empty(1, z.shape[0], ATOM_H, ATOM_WQ, ATOM_WK, device=z.device, dtype=torch.float32)
        _edge(z).tail_bias_fwd(z, gamma.float().contiguous(), wb.float().contiguous(), info, eps, tb[0])
        return tb

    @staticmethod
    @once_differentiable
    def backward(ctx, dtb):  # noqa: D102
        z, gamma, wb, info = ctx.saved_tensors
        dz = torch.zeros_like(z)
        part = _edge(z).tail_bias_bwd(z, gamma.float().contiguous(), wb.float().contiguous(), dtb.contiguous(), info, dz, ctx.eps)
        tot = part.sum(0)
        return dz, tot[:16].to(gamma.dtype), tot[16:].view(4, 16).to(wb.dtype), None, None


def _afwd_specs(M, N, at, xdt, par, gating, proj, kq, up, chain=False):
    D = ATOM_D
    f32 = torch.float32
    hid = 2 * D
    nab = 3 if up else 2
    return {
        "out": ((M, D), xdt),
        "xq": ((M, D), at),
        "xk": ((M, D), at),
        "xt": ((M, D), at),
        "q": ((M, D), at),
        "k": ((M, D), at),
        "v": ((M, D), at),
        "qn": ((M, D) if kq else (0,), at),
        "kn": ((M, D) if kq else (0,), at),
        "stqk": ((M, 4) if kq else (0,), f32),
        "g": ((M, D) if gating else (0,), at),
        "og": ((M, D), at),
        "lse": ((M // N, ATOM_H, N), f32),
        "ya": ((M, D) if proj else (0,), at),
        "y2": ((M, D), at),
        "xp": ((0,) if par else (M, D), f32),
        "st1": ((M, 2), f32),
        "st2": ((0,) if par else (M, 2), f32),
        "ab": ((M, nab * hid), at),
        "h": ((M, hid), at),
        "w_gq": (((2 if gating else 1) * D, D), at),
        "w_kv": ((2 * D, D), at),
        "w_o": ((D, D) if proj else (0,), at),
        "w_exp": ((nab * hid, D), at),
        "w_sq": ((D, hid), at),
        "stk": ((M, 2) if chain else (0,), f32),
    }


def _atom_fwd_fake(
    x,
    tab,
    bias2,
    widx,
    keys,
    ebias,
    tinfo,
    tb,
    wq,
    bq,
    wk,
    wv,
    wg,
    wo,
    wexp,
    wsq,
    qkw,
    par,
    sig,
    gating,
    proj,
    kq,
    up,
    slide,
    length,
    eps,
    eps_qk,
    chain=False,
):
    """The saved tensors of the atom block forward (``_ANAMES``): shapes and dtypes from ``_afwd_specs``."""
    spec = _afwd_specs(x.shape[0], length, tab.dtype, x.dtype, par, gating, proj, kq, up, chain)
    return [x.new_empty(spec[n][0], dtype=spec[n][1]) for n in _ANAMES]


@opaque(fake=_atom_fwd_fake, name="dit_variants_atom_train_fwd")
def atom_fwd(
    x: torch.Tensor,
    tab: torch.Tensor,
    bias2: torch.Tensor,
    widx: torch.Tensor,
    keys: torch.Tensor,
    ebias: torch.Tensor,
    tinfo: torch.Tensor,
    tb: torch.Tensor,
    wq: torch.Tensor,
    bq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    wexp: torch.Tensor,
    wsq: torch.Tensor,
    qkw: list[torch.Tensor],
    par: bool,
    sig: bool,
    gating: bool,
    proj: bool,
    kq: bool,
    up: bool,
    slide: bool,
    length: int,
    eps: float,
    eps_qk: float,
    chain: bool = False,
) -> list[torch.Tensor]:
    """One atom block's forward: [out, saved ...] (``_ANAMES``). x [A N, 128], tab [N, 8 D], bias2 [4, nwin, 32, 128] fp32 (log2, -inf off the
    keys); the slid edge windows (widx [5], keys [5, 128], ebias [5, 4, 32, 128] natural units) and its tail (tinfo [3], tb [1, nwin, 4, 32,
    128] natural units: ``_TailBias``) when ``slide``."""
    ext = _atom_ext(x)
    local = _local(x)
    M, D = x.shape
    N, A = length, M // length
    at = tab.dtype
    T = tab.shape[0]  # the table rows: N (one conditioning for every sample) or A N (one per sample)
    w_q, w_k, w_v, w_exp, w_sq = (w.to(at, copy=True) for w in (wq, wk, wv, wexp, wsq))
    w_gq = torch.cat([wg, wq]).to(at) if gating else w_q.clone()
    w_kv = torch.cat([wk, wv]).to(at)
    w_o = wo.to(at, copy=True) if proj else wo.new_empty(0)
    spec = _afwd_specs(M, N, at, x.dtype, par, gating, proj, kq, up, chain)
    skip = ("w_gq", "w_kv", "w_o", "w_exp", "w_sq", "q", "og", "lse", "ya")
    buf = {n: torch.empty(spec[n][0], device=x.device, dtype=spec[n][1]) for n in _ANAMES if n not in skip}
    cols = lambda c: c * D  # noqa: E731
    xq, xk, xt, st1, k, v = buf["xq"], buf["xk"], buf["xt"], buf["st1"], buf["k"], buf["v"]
    if par:
        ext.ln_mod_fwd(
            x, tab, [cols(A_SC_Q), cols(A_SC_K), cols(A_SC_T)], [cols(A_SH_Q), cols(A_SH_K), cols(A_SH_T)], [xq, xk, xt], st1, T, eps, sig
        )
    elif chain:  # the K / V AdaLN of the query AdaLN's output (Protenix): LN-modulate xq again
        ext.ln_mod_fwd(x, tab, [cols(A_SC_Q)], [cols(A_SH_Q)], [xq], st1, T, eps, sig)
        ext.ln_mod_fwd(xq, tab, [cols(A_SC_K)], [cols(A_SH_K)], [xk], buf["stk"], T, eps, sig)
    else:
        ext.ln_mod_fwd(x, tab, [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SH_Q), cols(A_SH_K)], [xq, xk], st1, T, eps, sig)
    q = torch.addmm(bq.to(at), xq, w_q.t())
    torch.mm(xk, w_kv[:D].t(), out=k)
    torch.mm(xk, w_kv[D:].t(), out=v)
    g = buf["g"]
    if gating:
        torch.mm(xq, w_gq[:D].t(), out=g)
    f32 = at == torch.float32
    if kq:
        qn, kn = buf["qn"], buf["kn"]
        ext.qkv_prep(q, k, v, qn, kn, v if f32 else None, buf["stqk"], qkw[0], qkw[1], qkw[2], qkw[3], 1.0, eps_qk, True, f32)
    else:
        qn, kn = q, k
        if f32:  # the TF32 core's operands, rounded to nearest in place
            ext.qkv_prep(q, k, v, q, k, v, None, None, None, None, None, 1.0, eps_qk, False, True)
    o, lse = (
        local.attn_fwd_mma(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), bias2)
        if at == torch.bfloat16
        else local.attn_fwd_mma32(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), bias2)
    )
    if slide:
        _edge(x).edge_attn_fwd(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), keys, widx, ebias, o)
        _edge(x).tail_attn_fwd(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), tinfo, tb, o)
    o = o.view(M, D)
    if gating:
        og = torch.empty_like(o)
        ext.gate_fwd(o, g, og)
    else:
        og = o
    ya = torch.mm(og, w_o.t()) if proj else og
    out, y2 = buf["out"], buf["y2"]
    ab, h = buf["ab"], buf["h"]
    if par:
        torch.mm(xt, w_exp.t(), out=ab)
        ext.glu_fwd(ab, h, up)
        torch.mm(h, w_sq.t(), out=y2)
        ext.res_out_fwd(x, ya, tab, cols(A_G_A), y2, cols(A_G_T), out, T, False)
    else:
        ext.res_ln_fwd(x, ya, tab, cols(A_G_A), cols(A_SC_T), cols(A_SH_T), buf["xp"], xt, buf["st2"], T, eps, sig)
        torch.mm(xt, w_exp.t(), out=ab)
        ext.glu_fwd(ab, h, up)
        torch.mm(h, w_sq.t(), out=y2)
        ext.res_out_fwd(buf["xp"], y2, tab, cols(A_G_T), None, 0, out, T, False)
    res = dict(buf, q=q, og=og, lse=lse, ya=ya if proj else ya.new_empty(0), w_gq=w_gq, w_kv=w_kv, w_o=w_o, w_exp=w_exp, w_sq=w_sq)
    return [res[n] for n in _ANAMES]


def _atom_bwd_fake(
    dout,
    x,
    tab,
    bias2,
    widx,
    keys,
    ebias,
    tinfo,
    tb,
    saved,
    wq,
    wk,
    wv,
    wg,
    wo,
    wexp,
    wsq,
    qkw,
    par,
    sig,
    gating,
    proj,
    kq,
    up,
    slide,
    length,
    eps,
    eps_qk,
    chain=False,
):
    """dx, dtab, dbias2, debias, the weight gradients (fp32), the four qk-norm vectors and dtb."""
    D = ATOM_D
    f32 = torch.float32
    return [
        torch.empty_like(x),
        tab.new_empty(tab.shape, dtype=f32),
        bias2.new_empty(bias2.shape, dtype=f32),
        ebias.new_empty(ebias.shape, dtype=f32),
        wq.new_empty(wq.shape, dtype=f32),
        wq.new_empty((D,), dtype=f32),
        wk.new_empty(wk.shape, dtype=f32),
        wv.new_empty(wv.shape, dtype=f32),
        wg.new_empty(wg.shape, dtype=f32),
        wo.new_empty(wo.shape, dtype=f32),
        wexp.new_empty(wexp.shape, dtype=f32),
        wsq.new_empty(wsq.shape, dtype=f32),
        wq.new_empty((D,), dtype=f32),
        wq.new_empty((D,), dtype=f32),
        wq.new_empty((D,), dtype=f32),
        wq.new_empty((D,), dtype=f32),
        tb.new_empty(tb.shape, dtype=f32),
    ]


@opaque(fake=_atom_bwd_fake, name="dit_variants_atom_train_bwd")
def atom_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    tab: torch.Tensor,
    bias2: torch.Tensor,
    widx: torch.Tensor,
    keys: torch.Tensor,
    ebias: torch.Tensor,
    tinfo: torch.Tensor,
    tb: torch.Tensor,
    saved: list[torch.Tensor],
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    wexp: torch.Tensor,
    wsq: torch.Tensor,
    qkw: list[torch.Tensor],
    par: bool,
    sig: bool,
    gating: bool,
    proj: bool,
    kq: bool,
    up: bool,
    slide: bool,
    length: int,
    eps: float,
    eps_qk: float,
    chain: bool = False,
) -> list[torch.Tensor]:
    """The atom block's backward: [dx, dtab, dbias2, debias, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq, dwq_ln, dbq_ln, dwk_ln, dbk_ln]."""
    ext = _atom_ext(x)
    local = _local(x)
    s = dict(zip(_ASAVED, saved, strict=True))
    M, D = x.shape
    N, A = length, M // length
    at = tab.dtype
    T = tab.shape[0]  # the table rows: N (one conditioning for every sample) or A N (one per sample)
    S = min(M // T, ATOM_SPLITS)
    dev = x.device
    f32 = torch.float32
    dout = dout.to(x.dtype).contiguous()
    part = torch.empty(S, T, 8 * D, device=dev, dtype=f32)
    cols = lambda c: c * D  # noqa: E731
    w_gq, w_kv, w_o, w_exp, w_sq = s["w_gq"], s["w_kv"], s["w_o"], s["w_exp"], s["w_sq"]
    xq, xk, xt, og, ya, y2, ab, h = (s[n] for n in ("xq", "xk", "xt", "og", "ya", "y2", "ab", "h"))
    if not proj:
        ya = og
    dya = torch.empty(M, D, device=dev, dtype=at)
    dy2 = torch.empty(M, D, device=dev, dtype=at)
    if par:
        ext.res_bwd(dout, ya, tab, cols(A_G_A), dya, y2, cols(A_G_T), dy2, part, cols(A_G_A), cols(A_G_T), T, S, False)
    else:
        ext.res_bwd(dout, y2, tab, cols(A_G_T), dy2, None, 0, None, part, cols(A_G_T), 0, T, S, False)
    dh = torch.mm(dy2, w_sq)
    dwsq = _mm_t(dy2, h)
    dab = torch.empty_like(ab)
    ext.glu_bwd(ab, dh, dab, up)
    dt = torch.mm(dab, w_exp)
    dwexp = _mm_t(dab, xt)
    if not par:
        dxp = torch.empty(M, D, device=dev, dtype=f32)
        ext.res_adaln_bwd(
            dout,
            dt,
            s["xp"],
            s["st2"],
            tab,
            cols(A_SC_T),
            cols(A_G_A),
            ya,
            dxp,
            dya,
            part,
            cols(A_SC_T),
            cols(A_SH_T),
            cols(A_G_A),
            T,
            S,
            sig,
        )
    dog = torch.mm(dya, w_o) if proj else dya
    dwo = _mm_t(dya, og) if proj else wo.new_empty(wo.shape, dtype=f32)
    dO = torch.empty(M, D, device=dev, dtype=at)
    dp = torch.empty(M, 4 * D, device=dev, dtype=at)  # [dg | dq | dk | dv]
    dd = torch.empty(A, ATOM_H, N, device=dev, dtype=f32)
    ext.gate_o_bwd(dog, og, s["g"] if gating else None, dO, dp[:, :D] if gating else None, dd, N, ATOM_DH, at == torch.float32)
    qa, ka = (s["qn"], s["kn"]) if kq else (s["q"], s["k"])
    if slide:
        # the five edge windows are the edge kernels' (their dO / D rows become zeros for the main kernel, which then adds nothing for them)
        dqe = torch.empty(A, 5, ATOM_WQ, D, device=dev, dtype=f32)
        dke = torch.empty(A, 5, ATOM_WK, D, device=dev, dtype=f32)
        dve = torch.empty(A, 5, ATOM_WK, D, device=dev, dtype=f32)
        dbp = torch.empty(A, 5, ATOM_H, ATOM_WQ, ATOM_WK, device=dev, dtype=f32)
        _edge(x).edge_bwd_compute(
            qa.view(A, N, D), ka.view(A, N, D), s["v"].view(A, N, D), dO.view(A, N, D), dd, keys, widx, ebias, dqe, dke, dve, dbp
        )
        # then the tail windows (every query window past the last real one; local_dit_h100.tail_info): the same, on the tail kernels
        from miniworld_engine.kernels.integrations.local_dit_h100 import tail_chunks

        nloop = tb.shape[1]
        dqt = torch.empty(A, N, D, device=dev, dtype=at)
        dkvt = torch.empty(A, ATOM_WK, 2 * D, device=dev, dtype=f32)
        dtbp = torch.empty(tail_chunks(A, nloop), nloop, ATOM_H, ATOM_WQ, ATOM_WK, device=dev, dtype=f32)
        _edge(x).tail_bwd(qa.view(A, N, D), ka.view(A, N, D), s["v"].view(A, N, D), dO.view(A, N, D), dd, tinfo, tb, dqt, dkvt, dtbp)
    dbias2 = local.attn_bwd_mma(
        qa.view(A, N, D), ka.view(A, N, D), s["v"].view(A, N, D), dO.view(A, N, D), bias2, s["lse"], dd, dp[:, D:], at == torch.float32
    )
    if slide:
        _edge(x).edge_bwd_apply(dp[:, D:], keys, widx, dqe, dke, dve, A, N)
        _edge(x).tail_bwd_apply(dp[:, D:], tinfo, dqt, dkvt, A, N, nloop)
        debias = dbp.sum(0)
        dtb = _edge(x).tail_dbias_sum(dtbp, tinfo)
    else:
        debias = ebias.new_empty(ebias.shape, dtype=f32)
        dtb = tb.new_empty(tb.shape, dtype=f32)
    if kq:
        nblk = (M + QKV_RPB - 1) // QKV_RPB
        qpart = torch.empty(nblk, 5 * D, device=dev, dtype=f32)
        ext.qkv_bwd(
            dp[:, D : 2 * D],
            dp[:, 2 * D : 3 * D],
            dp[:, 3 * D :],
            s["q"],
            s["k"],
            s["stqk"],
            qkw[0],
            qkw[2],
            dp[:, D : 2 * D],
            dp[:, 2 * D : 3 * D],
            dp[:, 3 * D :],
            qpart,
            QKV_RPB,
            True,
        )
        qsum = qpart.sum(0)
        dwq_ln, dbq_ln, dwk_ln, dbk_ln, dbq = (qsum[i * D : (i + 1) * D].clone() for i in range(5))
    else:
        nblk = (M + QKV_RPB - 1) // QKV_RPB
        qpart = torch.empty(nblk, D, device=dev, dtype=f32)
        ext.qkv_bwd(dp[:, D : 2 * D], None, None, None, None, None, None, None, None, None, None, qpart, QKV_RPB, False)
        dbq = qpart.sum(0)
        dwq_ln, dbq_ln, dwk_ln, dbk_ln = (torch.zeros(D, device=dev, dtype=f32) for _ in range(4))
    if gating:
        dxq = torch.mm(dp[:, : 2 * D], w_gq)
        dwgq = _mm_t(dp[:, : 2 * D], xq)
        dwg, dwq = dwgq[:D], dwgq[D:]
    else:
        dxq = torch.mm(dp[:, D : 2 * D], w_gq)
        dwq = _mm_t(dp[:, D : 2 * D], xq)
        dwg = wg.new_empty(wg.shape, dtype=f32)
    dxk = torch.mm(dp[:, 2 * D :], w_kv)
    dwkv = _mm_t(dp[:, 2 * D :], xk)
    dx = torch.empty_like(x)
    if par:
        ext.adaln_bwd(
            x,
            s["st1"],
            dout,
            [dxq, dxk, dt],
            tab,
            [cols(A_SC_Q), cols(A_SC_K), cols(A_SC_T)],
            [cols(A_SC_Q), cols(A_SC_K), cols(A_SC_T)],
            [cols(A_SH_Q), cols(A_SH_K), cols(A_SH_T)],
            dx,
            part,
            T,
            S,
            sig,
        )
    elif chain:  # through the K / V AdaLN into xq, then the query AdaLN
        dxq_t = torch.empty_like(xq)
        ext.adaln_bwd(xq, s["stk"], dxq, [dxk], tab, [cols(A_SC_K)], [cols(A_SC_K)], [cols(A_SH_K)], dxq_t, part, T, S, sig)
        ext.adaln_bwd(x, s["st1"], dxp, [dxq_t], tab, [cols(A_SC_Q)], [cols(A_SC_Q)], [cols(A_SH_Q)], dx, part, T, S, sig)
    else:
        ext.adaln_bwd(
            x,
            s["st1"],
            dxp,
            [dxq, dxk],
            tab,
            [cols(A_SC_Q), cols(A_SC_K)],
            [cols(A_SC_Q), cols(A_SC_K)],
            [cols(A_SH_Q), cols(A_SH_K)],
            dx,
            part,
            T,
            S,
            sig,
        )
    dtab = part.sum(0)
    return [
        dx,
        dtab,
        dbias2,
        debias,
        dwq.clone(),
        dbq,
        dwkv[:D].clone(),
        dwkv[D:].clone(),
        dwg.clone(),
        dwo,
        dwexp,
        dwsq,
        dwq_ln,
        dbq_ln,
        dwk_ln,
        dbk_ln,
        dtb,
    ]


class _AtomBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tab, bias2, widx, keys, ebias, tinfo, tb, wq, bq, wk, wv, wg, wo, wexp, wsq, wql, bql, wkl, bkl, cfg):  # noqa: D102
        par, sig, gating, proj, kq, up, slide, length, eps, eps_qk, *chain = cfg
        chain = bool(chain and chain[0])
        qkw = [wql, bql, wkl, bkl] if kq else []
        out, *saved = atom_fwd(
            x,
            tab,
            bias2,
            widx,
            keys,
            ebias,
            tinfo,
            tb,
            wq,
            bq,
            wk,
            wv,
            wg,
            wo,
            wexp,
            wsq,
            qkw,
            par,
            sig,
            gating,
            proj,
            kq,
            up,
            slide,
            length,
            eps,
            eps_qk,
            chain,
        )
        ctx.save_for_backward(x, tab, bias2, widx, keys, ebias, tinfo, tb, *saved, wq, wk, wv, wg, wo, wexp, wsq, wql, bql, wkl, bkl)
        ctx.cfg, ctx.n_saved = cfg, len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):  # noqa: D102
        par, sig, gating, proj, kq, up, slide, length, eps, eps_qk, *chain = ctx.cfg
        chain = bool(chain and chain[0])
        x, tab, bias2, widx, keys, ebias, tinfo, tb, *rest = ctx.saved_tensors
        saved, (wq, wk, wv, wg, wo, wexp, wsq, wql, bql, wkl, bkl) = rest[: ctx.n_saved], rest[ctx.n_saved :]
        qkw = [wql, bql, wkl, bkl] if kq else []
        res = atom_bwd(
            dout,
            x,
            tab,
            bias2,
            widx,
            keys,
            ebias,
            tinfo,
            tb,
            list(saved),
            wq,
            wk,
            wv,
            wg,
            wo,
            wexp,
            wsq,
            qkw,
            par,
            sig,
            gating,
            proj,
            kq,
            up,
            slide,
            length,
            eps,
            eps_qk,
            chain,
        )
        dx, dtab, dbias2, debias, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq, dwql, dbql, dwkl, dbkl, dtb = res
        return (
            dx,
            dtab.to(tab.dtype),
            dbias2,
            None,
            None,
            debias if slide else None,
            None,
            dtb if slide else None,
            dwq,
            dbq,
            dwk,
            dwv,
            dwg if gating else None,
            dwo if proj else None,
            dwexp,
            dwsq,
            *((dwql, dbql, dwkl, dbkl) if kq else (None, None, None, None)),
            None,
        )


class _AtomPairBias(torch.autograd.Function):
    """bias2 [4, nwin, 32, 128] fp32 (log2 units, -inf off the keys) of one block from the trunked pair z: ``pair_bias_mma`` / ``pair_bias_bwd_mma``."""

    @staticmethod
    def forward(ctx, z, gamma, wb, kmask, n, eps):  # noqa: D102
        ctx.save_for_backward(z, gamma, wb)
        ctx.eps = eps
        return _local(z).pair_bias_mma(z, gamma, wb, kmask, n, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, dbias):  # noqa: D102
        z, gamma, wb = ctx.saved_tensors
        dz, dgamma, dwb = _local(z).pair_bias_bwd_mma(z, gamma, wb, dbias.contiguous(), ctx.eps)
        return dz, dgamma.to(gamma.dtype), dwb.to(wb.dtype), None, None, None


def atom_serves(stack, single, cond, pair, mask, uid, *, need_grad: bool = True) -> bool:
    """An atom stack call with a gradient (``need_grad``; A100 runs its inference here too) on A100: bf16 or fp32, 128 wide, N a
    multiple of 64, a conditioning row set shared by the samples [N, d] or one per sample [A, N, d] (its tables then [A N, 8 d]), pair channels 16."""

    if stack.implementation == "pytorch" or not single.is_cuda or single.ndim != 3:
        return False
    if need_grad and not (torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *stack.parameters()))):
        return False
    if torch.cuda.get_device_capability(single.device) != (8, 0):  # A100 only (H100: the one atom DiT implementation, local_dit_h100)
        return False
    a, n, d = single.shape
    return (
        single.dtype in (torch.bfloat16, torch.float32)
        and d == ATOM_D
        and n % 64 == 0
        and tuple(cond.shape) in ((n, d), (a, n, d))
        and tuple(pair.shape) == (n // 32, 32, 128, 16)
        and (mask is None or tuple(mask.shape) == (n,))
        and (uid is None or tuple(uid.shape) == (n,))
        and stack.blocks[0].expand.weight.shape[0] == 4 * d
        and a > 0
    )


def _atom_tables(blk, c):
    def mod(norm):
        if norm.identity_scale:
            return norm.to_scale(c) + 1, norm.to_shift(c)
        cn = F.layer_norm(c.float(), c.shape[-1:], weight=norm.cond_norm_weight.float(), eps=1e-5).to(c.dtype)
        return norm.to_scale(cn), norm.to_shift(cn)

    scq, shq = mod(blk.q_adaln)
    sck, shk = mod(blk.k_adaln)
    sct, sht = mod(blk.trans_adaln)
    parts = [scq, shq, sck, shk, blk.attn_zero(c), sct, sht, blk.trans_zero(c)]
    dt = parts[0].dtype
    return torch.cat([p.to(dt) for p in parts], dim=-1).contiguous()


def atom_stack_train(stack, single, cond, pair, mask, uid):
    """The atom stack's forward under autograd on the H100 training kernels: x [A, N, 128] -> [A, N, 128]."""
    blocks = list(stack.blocks)
    a, n, d = single.shape
    if cond.ndim == 3:  # one conditioning per sample (A100): per-row tables [A N, 8 d], which the row kernels index by row
        cond = cond.reshape(a * n, cond.shape[-1])
    nwin = n // ATOM_WQ
    dev = single.device
    kmask = torch.ones(n, dtype=torch.bool, device=dev) if mask is None else mask.bool()
    slide = stack.key_window in ("slide", "slide_qblock")
    j = torch.arange(ATOM_WK, device=dev)
    # the clipped window's key of slot j: 32 w - 48 + j (outside [0, N): -inf already); a padding query keeps its keys
    same = None
    if uid is not None:
        idx = (ATOM_WQ * torch.arange(nwin, device=dev) - 48)[:, None] + j
        same = (uid.view(nwin, ATOM_WQ, 1) == uid[idx.clamp(0, n - 1)][:, None, :]) | ~kmask.view(nwin, ATOM_WQ, 1)
    widx = keys = valid = tinfo = None
    if slide:
        n_real = kmask.sum()
        last = (n_real - 1) // 32
        widx = torch.stack([last * 0, last * 0 + 1, last - 2, last - 1, last]).clamp(0, nwin - 1)
        # slide_qblock (IntelliFold2): the slid rule against R rounded up to a whole query window; its keys past R are masked below
        n_win = (n_real + ATOM_WQ - 1) // ATOM_WQ * ATOM_WQ if stack.key_window == "slide_qblock" else n_real
        starts = torch.minimum((ATOM_WQ * widx - 48).clamp(min=0), (n_win - ATOM_WK).clamp(min=0))
        keys = starts[:, None] + j
        valid = (keys < n_real)[:, None, :]
        keys = keys.clamp(max=n - 1)
        if uid is not None:
            rows = ATOM_WQ * widx[:, None] + torch.arange(ATOM_WQ, device=dev)
            valid = valid & ((uid[rows][:, :, None] == uid[keys][:, None, :]) | ~kmask[rows][:, :, None])
        valid = valid.expand(5, ATOM_WQ, ATOM_WK).to(torch.uint8).contiguous()
        # the tail: every query window past the last real one reads the same 128 keys (local_dit_h100.tail_info)
        from miniworld_engine.kernels.integrations.local_dit_h100 import tail_info

        tinfo = tail_info(n_real, stack.key_window == "slide_qblock")
    z = pair.contiguous()
    x = single.reshape(a * n, d).contiguous()
    empty = single.new_empty(0, dtype=torch.float32)
    for i, blk in enumerate(blocks):
        norm = blk.pair_norm if stack.per_block_pair_norm else stack.pair_norm
        wb = blk.to_pair_bias.weight if stack.per_block_pair_norm else stack.to_pair_bias.weight[ATOM_H * i : ATOM_H * (i + 1)]
        bias2 = _AtomPairBias.apply(z, norm.weight, wb, kmask, n, norm.eps)
        if same is not None:
            bias2 = bias2.masked_fill(~same[None], float("-inf"))
        if slide:
            ebias = _EdgeBias.apply(z, norm.weight, wb, widx, valid, norm.eps)
            tb = _TailBias.apply(z, norm.weight, wb, tinfo, norm.eps)
        else:
            ebias = tb = empty
        tab = _atom_tables(blk, cond)
        qk = (blk.q_norm.weight, blk.q_norm.bias, blk.k_norm.weight, blk.k_norm.bias) if blk.kq_norm else (empty,) * 4
        cfg = (
            stack.parallel,
            not blk.q_adaln.identity_scale,
            blk.gating,
            blk.project_output,
            blk.kq_norm,
            blk.up_gate,
            slide,
            n,
            blk.adaln_eps,
            blk.q_norm.eps if blk.kq_norm else 1e-5,
            getattr(stack, "chain", False),
        )
        wg = blk.to_g.weight if blk.gating else empty
        wo = blk.to_out.weight if blk.project_output else empty
        wexp = torch.cat([blk.expand.weight, *((blk.up.weight,) if blk.up_gate else ())])
        x = _AtomBlock.apply(
            x,
            tab,
            bias2,
            widx if slide else empty,
            keys if slide else empty,
            ebias,
            tinfo if slide else empty,
            tb,
            blk.to_q.weight,
            blk.to_q.bias,
            blk.to_k.weight,
            blk.to_v.weight,
            wg,
            wo,
            wexp,
            blk.squeeze.weight,
            *qk,
            cfg,
        )
    return x.view(a, n, d)
