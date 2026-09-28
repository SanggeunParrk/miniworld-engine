"""Tile-sized dX GEMM + LN backward + residual epilogue for wide Transition.

Unlike the historical 128-row hand-CUDA design, use a smaller row tile and
load x only after the GEMM accumulator is complete. This avoids keeping an
x prefetch alive across the mainloop. Output column reductions are local to
the CTA. Deterministic FP32 affine partials replace atomics and dxn HBM traffic.
"""
import torch
import triton
import triton.language as tl
from common import wide


@triton.jit
def _dx_ln(DAB, WAB, X, DY, G, RS, C1, DX, PART,
           M: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
           BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BD)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BD), 0, tl.float32)
    for k0 in range(0, 8*D, BK):
        aa = tl.load(DAB + rows[:, None]*(8*D) + (kk[None, :]+k0), rows[:, None] < M, 0)
        bb = tl.load(WAB + (kk[:, None]+k0)*D + cols[None, :], cols[None, :] < D, 0)
        acc = tl.dot(aa, bb, acc)
    # Preserve existing width-specific rounding contract, not just equations.
    if D >= 384:
        acc = acc.to(tl.bfloat16).to(tl.float32)
    mask = (rows[:, None] < M) & (cols[None, :] < D)
    xx = tl.load(X + rows[:, None]*D + cols[None, :], mask, 0).to(tl.float32)
    dy = tl.load(DY + rows[:, None]*D + cols[None, :], mask, 0).to(tl.float32)
    gam = tl.load(G + cols, cols < D, 0)
    rs = tl.load(RS + rows, rows < M, 0)
    c1 = tl.load(C1 + rows, rows < M, 0)
    xhat = xx * rs[:, None] - c1[:, None]
    xhat = tl.where(cols[None, :] < D, xhat, 0.)
    weighted = acc * gam[None, :]
    mean1 = tl.sum(weighted, 1) / D
    mean2 = tl.sum(weighted*xhat, 1) / D
    dx = (weighted - mean1[:, None] - xhat*mean2[:, None]) * rs[:, None]
    if D >= 384:
        dx = dx.to(tl.bfloat16).to(tl.float32)
    dx = dx + dy
    tl.store(DX + rows[:, None]*D + cols[None, :], dx, mask)
    pdg = tl.sum(tl.where(rows[:, None] < M, acc*xhat, 0.), 0)
    pdb = tl.sum(tl.where(rows[:, None] < M, acc, 0.), 0)
    tl.store(PART + pid*2*D + cols, pdg, cols < D)
    tl.store(PART + pid*2*D + D + cols, pdb, cols < D)


def dx_ln(dab,wab,x,dy,gamma,rstd,c1,bm=32,bk=64,warps=8,stages=2):
    m,d=x.shape
    dx=torch.empty_like(x)
    parts=torch.empty((triton.cdiv(m,bm),2,d),device=x.device,dtype=torch.float32)
    ker=_dx_ln[(triton.cdiv(m,bm),)](dab,wab,x,dy,gamma,rstd,c1,dx,parts,m,d,triton.next_power_of_2(d),bm,bk,
                                    num_warps=warps,num_stages=stages)
    grads=parts.sum(0)
    return dx,grads[0],grads[1],ker


def candidate(v,config):
    x,gamma,beta,wa,wb,ws,dy=v
    y,xn,rstd,c1,_=wide._fwd_launch(x,gamma,beta,wa,wb,ws,1e-5,True)
    hid,dab=wide._ext_for(x).gate(xn,dy,wide._pack(wa,wb,128),ws.t().contiguous(),True)
    dws=wide._mm_f32(dy.t(),hid)
    dwab=wide._mm_f32(dab.t(),xn)
    dx,dg,db,_=dx_ln(dab,torch.cat((wa,wb)),x,dy,gamma,rstd,c1,**config)
    return y,dx,dg,db,dwab[:4*x.shape[-1]].to(wa.dtype),dwab[4*x.shape[-1]:].to(wb.dtype),dws.to(ws.dtype)
