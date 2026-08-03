"""FULL pure-CUTLASS ConditionedTransition-tail training: EVERY GEMM in CUTLASS (no cuBLAS).

GEMMs via ct_gemm_ext (gemm_nt = A@B^T, gemm_tn = A^T@B wgrad), tuned per shape.
swiglu-bwd input-fused + dual-output (dh once) via ct_bwd_ext.fused_dab_packed.
Elementwise (SwiGLU fwd, gate fwd, gate-bwd) via the champion's fused triton kernels
(pure elementwise, no GEMM) — these are not cuBLAS and are the same fused passes the champion uses.
"""
from __future__ import annotations
import torch
import ct_gemm_ext as G
import ct_bwd_ext as B
import ct_train_ext as T          # fused_h, fused_y (gate-fused squeeze)
from miniworld_engine.kernels.conditioned_transition.triton.training import (
    _swiglu, _gate, _gate_bwd,
)

# best configs per (regime, op); loaded from gemm_pick.py output. default 0 if unknown.
import json, os
_cfgf = "/home/psk6950/miniworld-engine/_ct_cutlass/gemm_cfgs.json"
if os.path.exists(_cfgf):
    _c = json.load(open(_cfgf)); NT_BEST = _c["NT"]; TN_BEST = _c["TN"]
else:
    NT_BEST = {}; TN_BEST = {}


def _reg(d):
    return "atom" if d <= 128 else "token"


# StreamK configs (cfg>=6) are NOT CUDA-graph-capturable (host-side reduction setup hangs
# capture). For the graphed training path, replace any StreamK pick with a safe non-StreamK
# fallback (cfg 2 = 64x128x64 pingpong, best persistent for large-K).
_SK_FALLBACK = 2


def _safe(c):
    return c if c < 6 else _SK_FALLBACK


def nt(A, Bm, regime, L, op):
    return G.gemm_nt(A, Bm, _safe(NT_BEST.get(f"{regime}_{L}_{op}", 0)))


def tn(A, Bm, regime, L, op):
    return G.gemm_tn(A, Bm, TN_BEST.get(f"{regime}_{L}_{op}", 0))


class CTFull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc):
        x = x.contiguous(); cond = cond.contiguous()
        wa = wa.contiguous(); wb = wb.contiguous(); ws = ws.contiguous()
        wsc = wsc.contiguous(); bsc = bsc.contiguous()
        ND = wa.shape[0]; d = x.shape[1]; L = x.shape[0]; reg = _reg(d)
        wcat = torch.cat([wa, wb], dim=0)               # (2ND, K)  [precomputed once per fwd]
        ab = nt(x, wcat, reg, L, "expand")              # (M, 2ND)
        a, b = ab[:, :ND], ab[:, ND:]                   # strided views, NO copy
        h = _swiglu(a, b)                               # fused triton SwiGLU (handles strides)
        out = nt(h, ws, reg, L, "squeeze")              # h@ws^T (ws:(D,ND))
        # gate-FUSED squeeze: ONE CUTLASS GEMM computes scale=cond@Wsc^T+bsc, fuses sigmoid-gate
        # (y=sigmoid(scale)*out) in its epilogue, and AuxStores scale for bwd. Cuts the separate
        # scale-GEMM + gate elementwise -> forward 5 launches -> 4.
        y, scale = T.fused_y(cond, wsc, bsc, out)
        ctx.save_for_backward(x, cond, a, b, h, out, scale, wcat, ws, wsc)
        ctx.ND = ND; ctx.reg = reg; ctx.L = L
        return y

    @staticmethod
    def backward(ctx, dy):
        x, cond, a, b, h, out, scale, wcat, ws, wsc = ctx.saved_tensors
        ND = ctx.ND; reg = ctx.reg; L = ctx.L
        dy = dy.contiguous()
        # gate-bwd fused elementwise -> dout, dscale (materialized; cuBLAS wgrad reads them)
        dout, dscale = _gate_bwd(out, scale, dy)
        dout = dout.contiguous()
        # dgrad: CUTLASS
        dcond = nt(dscale, wsc.t().contiguous(), reg, L, "dcond")   # dscale@wsc
        wsT = ws.t().contiguous()                                  # (ND, D)
        dab = torch.empty(L, 2 * ND, device=a.device, dtype=a.dtype)
        B.fused_dab_packed(dout, wsT, a, b, dab)                    # da,db packed (dh once, fused)
        dx = nt(dab, wcat.t().contiguous(), reg, L, "dx")          # dab@wcat
        # wgrad: cuBLAS (per user decision — CUTLASS TF32 TN huge-M wgrad lags cuBLAS 2.3-3.9x)
        dWsc = dscale.t() @ cond
        db_sc = dscale.sum(0)
        dWs = dout.t() @ h
        dWcat = dab.t() @ x
        dWa, dWb = dWcat[:ND].contiguous(), dWcat[ND:].contiguous()
        return dx, dcond, dWa, dWb, dWs, dWsc, db_sc


def cond_transition_train_full(x, cond, wa, wb, ws, wsc, bsc):
    return CTFull.apply(x, cond, wa, wb, ws, wsc, bsc)
