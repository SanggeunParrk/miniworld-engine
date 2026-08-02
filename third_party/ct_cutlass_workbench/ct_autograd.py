"""Python autograd Function wrapping the CUTLASS fused training path for the
post-AdaLN ConditionedTransition tail.

Forward mirrors the champion (cond_transition_train): cat-merged ab GEMM + elementwise
SwiGLU + out GEMM + sigmoid-gate. (No forward CUTLASS fusion: a AND b are both saved for
bwd, so the cat-merged single GEMM already matches the champion's GEMM count; a fused-epilogue
expand would force a redundant second expand GEMM.)

The CUTLASS WIN is the BACKWARD input-fusion: the dh-GEMM (dout@Ws) epilogue recomputes the
swiglu-bwd from saved a,b and emits da (D output) + db (AuxStore) in ONE pass (fused_dab), so
dh:(M,ND) and dab:(M,2ND) NEVER materialize in HBM and the swiglu-bwd elementwise launch is
gone — the thing the triton+cuBLAS champion cannot do.
"""
from __future__ import annotations

import torch

import ct_bwd_ext as bext        # fused_dab (dual-output input-fused swiglu-bwd)

_USE_DUAL = True


class CondTransitionCutlass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc):
        x = x.contiguous(); cond = cond.contiguous()
        wa = wa.contiguous(); wb = wb.contiguous(); ws = ws.contiguous()
        wsc = wsc.contiguous(); bsc = bsc.contiguous()
        ND = wa.shape[0]
        wcat = torch.cat([wa, wb], dim=0)            # (2ND, K)
        ab = x @ wcat.t()                            # (M, 2ND) one cat-merged GEMM
        a, b = ab[:, :ND], ab[:, ND:]
        h = torch.nn.functional.silu(a) * b          # SwiGLU elementwise
        out = h @ ws.t()                             # (M, D)
        scale = torch.addmm(bsc, cond, wsc.t())      # cond@Wsc^T + b_sc
        y = torch.sigmoid(scale) * out
        ctx.save_for_backward(x, cond, a.contiguous(), b.contiguous(), h, out, scale, wcat, ws, wsc)
        ctx.ND = ND
        return y

    @staticmethod
    def backward(ctx, dy):
        x, cond, a, b, h, out, scale, wcat, ws, wsc = ctx.saved_tensors
        ND = ctx.ND
        dy = dy.contiguous()
        # gate-bwd via the champion's fused triton kernel (dout,dscale in one pass)
        from miniworld_engine.kernels.conditioned_transition.triton.training import _gate_bwd
        dout, dscale = _gate_bwd(out, scale, dy)
        dout = dout.contiguous()
        dcond = dscale @ wsc
        dWsc = dscale.t() @ cond
        db_sc = dscale.sum(0)
        dWs = dout.t() @ h
        wsT = ws.t().contiguous()                    # (ND, D) so A@B^T = dout@ws = dh
        # INPUT-FUSED swiglu-bwd, PACKED: dh computed ONCE; da,db written straight into dab.
        M = a.shape[0]
        dab = torch.empty(M, 2 * ND, device=a.device, dtype=a.dtype)
        bext.fused_dab_packed(dout, wsT, a, b, dab)  # writes dab[:,:ND]=da, dab[:,ND:]=db
        dx = dab @ wcat
        dWcat = dab.t() @ x
        dWa, dWb = dWcat[:ND], dWcat[ND:]
        return dx, dcond, dWa.contiguous(), dWb.contiguous(), dWs, dWsc, db_sc


def cond_transition_train_cutlass(x, cond, wa, wb, ws, wsc, bsc):
    return CondTransitionCutlass.apply(x, cond, wa, wb, ws, wsc, bsc)
