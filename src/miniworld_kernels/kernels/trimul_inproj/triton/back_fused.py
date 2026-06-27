"""Fused front-backward: EW (gated-GEMM bwd) fused INTO the two grad GEMMs so the
(M,4D) d_preact intermediate never hits HBM (dt-v1 materializes it; we don't).

Inputs are channel-major (matches the cute front's bdll output + saved preact):
  d_lr   : (2D, M)  grads of [left|right]            (= d_lr.reshape(2D,M))
  preact : (4D, M)  pre-GLU logits, interleaved [g,p] per col, left then right
  x_n    : (M, D)
  W*     : (D, D)   x@W form
EW (per element):  gL=σ(gLlog); d_pL=dL*gL; d_gLlog=dL*pL*gL(1-gL)  (and R).
Outputs: dx_n (M,D), dWL/dWLg/dWR/dWRg (D,D).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dx_kernel(d_lr, preact, W, dx, M, D: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mm = rm < M
    rd = tl.arange(0, D)
    D2 = 2 * D
    b = rm[:, None]
    dl = tl.load(d_lr + rd[None, :] * M + b, mask=mm[:, None], other=0.0).to(tl.float32)
    dr = tl.load(d_lr + (D + rd[None, :]) * M + b, mask=mm[:, None], other=0.0).to(tl.float32)
    gLlog = tl.load(preact + (2 * rd[None, :]) * M + b, mask=mm[:, None], other=0.0).to(tl.float32)
    pL = tl.load(preact + (2 * rd[None, :] + 1) * M + b, mask=mm[:, None], other=0.0).to(tl.float32)
    gRlog = tl.load(preact + (D2 + 2 * rd[None, :]) * M + b, mask=mm[:, None], other=0.0).to(tl.float32)
    pR = tl.load(preact + (D2 + 2 * rd[None, :] + 1) * M + b, mask=mm[:, None], other=0.0).to(tl.float32)
    gL = tl.sigmoid(gLlog)
    gR = tl.sigmoid(gRlog)
    d_pL = (dl * gL).to(tl.bfloat16)
    d_gLlog = (dl * pL * gL * (1 - gL)).to(tl.bfloat16)
    d_pR = (dr * gR).to(tl.bfloat16)
    d_gRlog = (dr * pR * gR * (1 - gR)).to(tl.bfloat16)
    rk = tl.arange(0, D)
    WLg = tl.load(W + rk[:, None] * (4 * D) + (0 * D + rd[None, :])).to(tl.bfloat16)
    WL = tl.load(W + rk[:, None] * (4 * D) + (1 * D + rd[None, :])).to(tl.bfloat16)
    WRg = tl.load(W + rk[:, None] * (4 * D) + (2 * D + rd[None, :])).to(tl.bfloat16)
    WR = tl.load(W + rk[:, None] * (4 * D) + (3 * D + rd[None, :])).to(tl.bfloat16)
    acc = tl.dot(d_gLlog, tl.trans(WLg))
    acc += tl.dot(d_pL, tl.trans(WL))
    acc += tl.dot(d_gRlog, tl.trans(WRg))
    acc += tl.dot(d_pR, tl.trans(WR))
    tl.store(dx + rm[:, None] * D + rd[None, :], acc.to(tl.bfloat16), mask=mm[:, None])


@triton.jit
def _dw_kernel(d_lr, preact, x_n, dW, M, D: tl.constexpr, BK: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    rd = tl.arange(0, D)
    D2 = 2 * D
    aLg = tl.zeros((D, D), tl.float32)
    aL = tl.zeros((D, D), tl.float32)
    aRg = tl.zeros((D, D), tl.float32)
    aR = tl.zeros((D, D), tl.float32)
    n_tiles = tl.cdiv(M, BK)
    for kt in range(pid, n_tiles, NPROG):
        rk = kt * BK + tl.arange(0, BK)
        mk = rk < M
        b = rk[:, None]
        dl = tl.load(d_lr + rd[None, :] * M + b, mask=mk[:, None], other=0.0).to(tl.float32)
        dr = tl.load(d_lr + (D + rd[None, :]) * M + b, mask=mk[:, None], other=0.0).to(tl.float32)
        gLlog = tl.load(preact + (2 * rd[None, :]) * M + b, mask=mk[:, None], other=0.0).to(tl.float32)
        pL = tl.load(preact + (2 * rd[None, :] + 1) * M + b, mask=mk[:, None], other=0.0).to(tl.float32)
        gRlog = tl.load(preact + (D2 + 2 * rd[None, :]) * M + b, mask=mk[:, None], other=0.0).to(tl.float32)
        pR = tl.load(preact + (D2 + 2 * rd[None, :] + 1) * M + b, mask=mk[:, None], other=0.0).to(tl.float32)
        gL = tl.sigmoid(gLlog)
        gR = tl.sigmoid(gRlog)
        d_pL = (dl * gL).to(tl.bfloat16)
        d_gLlog = (dl * pL * gL * (1 - gL)).to(tl.bfloat16)
        d_pR = (dr * gR).to(tl.bfloat16)
        d_gRlog = (dr * pR * gR * (1 - gR)).to(tl.bfloat16)
        xt = tl.load(x_n + b * D + rd[None, :], mask=mk[:, None], other=0.0).to(tl.bfloat16)  # (BK,D)
        xt_t = tl.trans(xt)  # (D, BK)
        aLg += tl.dot(xt_t, d_gLlog)   # dWLg[k,d] = sum_m x[m,k] d_gLlog[m,d]
        aL += tl.dot(xt_t, d_pL)
        aRg += tl.dot(xt_t, d_gRlog)
        aR += tl.dot(xt_t, d_pR)
    # dW layout (D, 4D) = [WLg|WL|WRg|WR] blocks
    rk2 = tl.arange(0, D)
    row = rk2[:, None] * (4 * D)
    tl.atomic_add(dW + row + (0 * D + rd[None, :]), aLg)
    tl.atomic_add(dW + row + (1 * D + rd[None, :]), aL)
    tl.atomic_add(dW + row + (2 * D + rd[None, :]), aRg)
    tl.atomic_add(dW + row + (3 * D + rd[None, :]), aR)


@triton.autotune(
    configs=[triton.Config({"BLK": b}, num_warps=nw) for b in (1024, 2048, 4096) for nw in (4, 8)],
    key=["DM"],
)
@triton.jit
def _dconcat_kernel(dL_ptr, dR_ptr, preact, out, M, DM, D: tl.constexpr, BLK: tl.constexpr):
    """1D channel-major elementwise: out (4D,M) = [d_gLlog; d_pL; d_gRlog; d_pR].
    Iterate over DM=D*M positions (d,m); dL=dL_ptr[idx], dR=dR_ptr[idx] (separate left/right
    buffers — no d_lr cat in the caller); preact is interleaved so needs (2d)*M+m indexing.
    1D (no full-D tile) → D-general, no reg blowup."""
    # int64 offsets: at large d·L (e.g. d_pair=512, L=1024) the max flat offset 4·DM = 2^32
    # overflows int32/uint32 → illegal memory access. Promote idx, M, DM to int64 so every
    # derived offset (k·DM+idx, (2d±)·M+m) is computed in int64.
    Mi = M.to(tl.int64)
    DMi = DM.to(tl.int64)
    idx = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK).to(tl.int64)
    mask = idx < DMi
    d = idx // Mi
    m = idx - d * Mi
    D2 = 2 * D
    dL = tl.load(dL_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dR = tl.load(dR_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    gLlog = tl.load(preact + (2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pL = tl.load(preact + (2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gRlog = tl.load(preact + (D2 + 2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pR = tl.load(preact + (D2 + 2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gL, gR = tl.sigmoid(gLlog), tl.sigmoid(gRlog)
    et = out.dtype.element_ty
    tl.store(out + idx, (dL * pL * gL * (1 - gL)).to(et), mask=mask)            # d_gLlog
    tl.store(out + DMi + idx, (dL * gL).to(et), mask=mask)                      # d_pL
    tl.store(out + 2 * DMi + idx, (dR * pR * gR * (1 - gR)).to(et), mask=mask)  # d_gRlog
    tl.store(out + 3 * DMi + idx, (dR * gR).to(et), mask=mask)                  # d_pR


def front_bwd_fused(d_left, d_right, preact, x_n, WL, WLg, WR, WRg):
    """d_left/d_right:(B,H,L,L) preact:(B,4H,L,L) x_n:(B,L,L,Din) W*:(Din,H). dx_n + 4 wgrads.

    d_left/d_right are passed SEPARATELY (not cat'd into a (B,2H,L,L) d_lr) — the elementwise
    kernel reads them from two pointers, killing the per-backward (B,2H,L,L) cat copy.

    Dimension-general: the per-side HIDDEN width H is INDEPENDENT of the input width Din
    (from the weights / x_n). They coincide for square trimul (H=Din=D) but NOT for
    bidirectional (H=2·d_hidden, Din=d_pair) — keep them separate or the x_n reshape/dxn break.

    Kernel-optimized path: a light channel-major elementwise (d_concat) feeds TWO cuBLAS
    GEMMs (dW = d_concatᵀ-stacked; dxn). cuBLAS hits far higher efficiency on these
    (esp. the huge-K wgrad reduction) than the hand-written split-K _dw_kernel (~30 TFLOPS)."""
    dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
        d_left, d_right, preact, x_n, WL, WLg, WR, WRg)
    B, _, L, _ = d_left.shape
    Din = WL.shape[0]
    dxn = (dconc.t() @ W_stack).reshape(B, L, L, Din)   # cuBLAS (non-merged path)
    return dxn, dWL, dWLg, dWR, dWRg


def front_bwd_dW(d_left, d_right, preact, x_n, WL, WLg, WR, WRg):
    """The front bwd EXCEPT the final dxn GEMM: builds d_concat (elementwise), the 4 weight
    grads (cuBLAS huge-K, STAYS cuBLAS), and the stacked W operand. Returns
    (dconc (4H,M), dWL, dWLg, dWR, dWRg, W_stack (4H,Din)). The caller forms dxn = dconcᵀ@W_stack
    (and `BidirBackHalf` fuses the gate's dx_gate add into that GEMM in cute)."""
    B, H, L, _ = d_left.shape   # per-side hidden width
    Din = WL.shape[0]           # input width (= d_pair); may differ from H (bidirectional)
    M = B * L * L
    dt = x_n.dtype
    dL2 = d_left.reshape(H * M)
    dR2 = d_right.reshape(H * M)
    preact2 = preact.reshape(4 * H, M)
    xf = x_n.reshape(M, Din)

    dconc = torch.empty(4 * H, M, device=x_n.device, dtype=dt)   # [d_gLlog;d_pL;d_gRlog;d_pR]
    DM = H * M
    _dconcat_kernel[lambda meta: (triton.cdiv(DM, meta["BLK"]),)](dL2, dR2, preact2, dconc, M, DM, D=H)

    # dW: (4H,M)@(M,Din) — dispatched (huge-K reduction reliably picks cuBLAS; quack 2.6-5x
    # slower there — measured. dispatch confirms + self-documents).
    from miniworld_kernels.kernels.trimul_inproj.cute import dispatch
    dWs = dispatch.mm("dWs", dconc, xf)
    dWLg = dWs[:H].t().contiguous()
    dWL = dWs[H:2 * H].t().contiguous()
    dWRg = dWs[2 * H:3 * H].t().contiguous()
    dWR = dWs[3 * H:].t().contiguous()
    W_stack = torch.cat([WLg.t(), WL.t(), WRg.t(), WR.t()], dim=0)   # (4H, Din)
    return dconc, dWL, dWLg, dWR, dWRg, W_stack
