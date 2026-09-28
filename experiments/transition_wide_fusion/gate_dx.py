"""Experimental row-owned gate + dX GEMM fusion, never production dispatch.

Keep one [BM,D] dX accumulator across hidden tiles. a, b, dh, dA, dB
live only for one hidden tile; round dh and dAB exactly at the baseline's
BF16 boundaries. Store dAB for dWa/dWb but consume it on chip for dX.
The forward and FP32-output weight gradient GEMMs remain unchanged.
"""
import torch
import triton
import triton.language as tl
from common import wide, _transition_ln_bwd


@triton.jit
def _gate_dx(XN, DY, WA, WB, WS, HID, DAB, DXN,
             M: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
             BM: tl.constexpr, BH: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BD)
    hidden = tl.arange(0, BH)
    xn = tl.load(XN + rows[:, None] * D + cols[None, :],
                 (rows[:, None] < M) & (cols[None, :] < D), 0)
    dy = tl.load(DY + rows[:, None] * D + cols[None, :],
                 (rows[:, None] < M) & (cols[None, :] < D), 0)
    dxn = tl.full((BM, BD), 0, tl.float32)
    for h0 in range(0, 4 * D, BH):
        hh = h0 + hidden
        wa = tl.load(WA + hh[None, :] * D + cols[:, None], cols[:, None] < D, 0)
        wb = tl.load(WB + hh[None, :] * D + cols[:, None], cols[:, None] < D, 0)
        ws = tl.load(WS + cols[:, None] * (4*D) + hh[None, :], cols[:, None] < D, 0)
        a = tl.dot(xn, wa)
        b = tl.dot(xn, wb)
        dh = tl.dot(dy, ws).to(tl.bfloat16).to(tl.float32)
        tanh = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", constraints="=f,f",
                                        args=[.5*a], dtype=tl.float32, is_pure=True, pack=1)
        s = .5*tanh + .5
        silu = a * s
        h = (silu * b).to(tl.bfloat16)
        da = ((dh * b) * (s + silu * (1-s))).to(tl.bfloat16)
        db = (dh * silu).to(tl.bfloat16)
        tl.store(HID + rows[:, None]*(4*D) + hh[None, :], h, rows[:, None] < M)
        tl.store(DAB + rows[:, None]*(8*D) + hh[None, :], da, rows[:, None] < M)
        tl.store(DAB + rows[:, None]*(8*D) + 4*D + hh[None, :], db, rows[:, None] < M)
        dxn = tl.dot(da, tl.trans(wa), dxn)
        dxn = tl.dot(db, tl.trans(wb), dxn)
    tl.store(DXN + rows[:, None]*D + cols[None, :], dxn,
             (rows[:, None] < M) & (cols[None, :] < D))


def gate_dx(xn, dy, wa, wb, ws, bm=32, bh=64, warps=8):
    m, d = xn.shape
    hid = xn.new_empty((m, 4*d))
    dab = xn.new_empty((m, 8*d))
    dxn = torch.empty((m,d), device=xn.device, dtype=torch.float32 if d==256 else xn.dtype)
    compiled = _gate_dx[(triton.cdiv(m,bm),)](xn,dy,wa,wb,ws,hid,dab,dxn,m,d,triton.next_power_of_2(d),bm,bh,
                                            num_warps=warps,num_stages=1)
    return hid, dab, dxn, compiled


def candidate(v, config):
    x, gamma, beta, wa, wb, ws, dy = v
    y, xn, rstd, c1, _ = wide._fwd_launch(x, gamma, beta, wa, wb, ws, 1e-5, True)
    hid, dab, dxn, _ = gate_dx(xn, dy, wa, wb, ws, **config)
    dws = wide._mm_f32(dy.t(), hid)
    dwab = wide._mm_f32(dab.t(), xn)
    dx, dg, db = _transition_ln_bwd(dxn,x,rstd,c1,gamma)
    dx.add_(dy)
    return y, dx, dg, db, dwab[:4*x.shape[-1]].to(wa.dtype), dwab[4*x.shape[-1]:].to(wb.dtype), dws.to(ws.dtype)
