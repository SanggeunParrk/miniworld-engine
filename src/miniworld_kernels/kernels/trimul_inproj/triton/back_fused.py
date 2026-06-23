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


def front_bwd_fused(d_lr, preact, x_n, WL, WLg, WR, WRg):
    """d_lr:(B,2D,L,L) preact:(B,4D,L,L) x_n:(B,L,L,D). Returns dx_n + 4 weight grads."""
    B, D2, L, _ = d_lr.shape
    D = D2 // 2
    M = B * L * L
    dt = x_n.dtype
    d_lr2 = d_lr.reshape(D2, M)
    preact2 = preact.reshape(4 * D, M)
    xf = x_n.reshape(M, D)
    W = torch.cat([WLg, WL, WRg, WR], dim=1).contiguous()   # (D,4D) [WLg|WL|WRg|WR]
    dx = torch.empty(M, D, device=x_n.device, dtype=dt)
    dW = torch.zeros(D, 4 * D, device=x_n.device, dtype=torch.float32)
    BM = 64
    _dx_kernel[(triton.cdiv(M, BM),)](d_lr2, preact2, W, dx, M, D, BM, num_warps=4)
    NPROG = 256
    BK = 128
    _dw_kernel[(NPROG,)](d_lr2, preact2, xf, dW, M, D, BK, NPROG, num_warps=4)
    dWg = dW.to(dt)
    dWLg, dWL, dWRg, dWR = dWg[:, :D], dWg[:, D:2 * D], dWg[:, 2 * D:3 * D], dWg[:, 3 * D:]
    return dx.reshape(B, L, L, D), dWL.contiguous(), dWLg.contiguous(), dWR.contiguous(), dWRg.contiguous()
