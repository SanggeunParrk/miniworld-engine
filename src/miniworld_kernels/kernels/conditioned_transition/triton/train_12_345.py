"""Training autograd Function mirroring the 1+2 | 3+4+5 forward (cond_transition_fwd_12_345).

Forward groups (see composed.py):
    K1 = 1+2 : expand + SwiGLU            -> h
    K2 = 3+4+5 : squeeze + to_scale + gate -> y
Each forward kernel FUSES GEMM + elementwise. The backward MIRRORS that, in reverse order,
fusing the producing elementwise INTO the consuming dgrad GEMM prologue (the exact structure
to drop-in CUTLASS-ify later):

  Bwd of K2 (3+4+5)  -- given dy + saved(out, scale, h, Ws, Wsc, cond):
    gate-bwd (elementwise):  dout = sigmoid(scale)*dy ; dscale = out*sg*(1-sg)*dy
    squeeze-bwd:             dh   = dout  @ Ws         (dgrad) ; dWs  = dout^T  @ h    (wgrad)
    to_scale-bwd:            dcond = dscale @ Wsc       (dgrad) ; dWsc = dscale^T @ cond (wgrad)
                             db_sc = dscale.sum(0)
    -> the gate-bwd elementwise is FUSED into the dh dgrad GEMM prologue (_dh_gatebwd);
       it emits dout,dscale (materialized) for the cuBLAS wgrads + the dcond dgrad GEMM.

  Bwd of K1 (1+2)  -- given dh + saved(ab=[a|b], x, Wa, Wb):
    swiglu-bwd (elementwise): da = dh*b*silu'(a) ; db = dh*silu(a)  (silu'(a)=sa+silu(a)*(1-sa))
    expand-bwd:               dx = dab @ [Wa;Wb]   (dgrad, ONE cat-merged GEMM)
                              dWcat = dab^T @ x     (wgrad) -> dWa, dWb
    -> the swiglu-bwd elementwise is FUSED into the dx dgrad GEMM prologue (_dx_swiglubwd);
       it emits dab (materialized) for the cuBLAS dWa/dWb wgrad.

dgrad GEMMs: fused-elementwise triton (TF32), the mirrored CUTLASS-able structure.
wgrad GEMMs (dWs, dWsc, dWa, dWb): cuBLAS (reductions over M — cuBLAS's domain).
Forward saves ab, out, scale (NOT h: recompute h=silu(a)*b from ab in backward, avoiding the
[M,ND] h write). Works uniformly for atom (d=128) and token (d=768).
"""

from __future__ import annotations

import torch

# K1 (1+2) and K2 (3+4+5) forward kernels, emitting the saved-for-bwd tensors:
#   _fwd_expand_swiglu -> (h, ab=[a|b])   _fwd_squeeze_gate -> (y, out, scale)
from .train_fused import (
    _dgemm,
    _dh_gatebwd,
    _dx_swiglubwd,
    _fwd_expand_swiglu,
    _fwd_squeeze_gate,
    _gate_bwd,
    _swiglu_bwd_pack,
)
from .training import _swiglu  # h = silu(a)*b elementwise (recompute in bwd)


class ConditionedTransitionTail12345Function(torch.autograd.Function):
    """1+2 | 3+4+5 forward (cond_transition_fwd_12_345) + structurally-mirrored fused backward."""

    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc):
        x = x.contiguous(); cond = cond.contiguous()
        wa = wa.contiguous(); wb = wb.contiguous(); ws = ws.contiguous()
        wsc = wsc.contiguous(); bsc = bsc.contiguous()
        # K1 (1+2): expand + SwiGLU -> h ; ALSO emits ab=[a|b] (pre-activations for swiglu-bwd).
        h, ab = _fwd_expand_swiglu(x, wa, wb)
        # K2 (3+4+5): squeeze + to_scale + gate -> y ; emits out, scale (pre-gate, for gate-bwd).
        y, out, scale = _fwd_squeeze_gate(h, cond, ws, wsc, bsc)
        # Save ab,out,scale (NOT h: it is freed after K2; recomputed from ab in bwd).
        wcat = torch.cat([wa, wb], dim=0)          # (2*ND, K) for the cat-merged dx GEMM
        ctx.save_for_backward(x, cond, ab, out, scale, wa, wb, wcat, ws, wsc)
        ctx.ND = wa.shape[0]
        return y

    @staticmethod
    def backward(ctx, dy):
        x, cond, ab, out, scale, wa, wb, wcat, ws, wsc = ctx.saved_tensors
        ND = ctx.ND
        M, D = out.shape
        DC = cond.shape[1]
        dy = dy.contiguous()

        # ---- Bwd group 3+4+5 (mirror of fwd K2) ----
        # gate-bwd FUSED into the dh dgrad GEMM prologue; emits dout, dscale.
        dh, dout, dscale = _dh_gatebwd(out, scale, dy, ws, ND)      # dh=(M,ND); dout,dscale=(M,D)
        dcond = _dgemm(dscale, wsc, M, DC, D, wsc.stride(0), wsc.stride(1))  # dscale @ Wsc
        # wgrad (cuBLAS, reductions over M)
        dWs = dout.t() @ _recompute_h(ab, ND)                      # dout^T @ h
        dWsc = dscale.t() @ cond                                   # dscale^T @ cond
        db_sc = dscale.sum(0)

        # ---- Bwd group 1+2 (mirror of fwd K1) ----
        # swiglu-bwd FUSED into the dx dgrad GEMM prologue (one cat-merged GEMM); emits dab.
        dx, dab = _dx_swiglubwd(dh, ab, wcat)                      # dx=(M,K); dab=(M,2ND)
        dWcat = dab.t() @ x                                        # (2ND, K) wgrad (cuBLAS)
        dWa, dWb = dWcat[:ND].contiguous(), dWcat[ND:].contiguous()
        return dx, dcond, dWa, dWb, dWs, dWsc, db_sc


def _recompute_h(ab, ND):
    """h = silu(a)*b from saved ab=[a|b] (avoids saving the (M,ND) h in forward)."""
    a, b = ab[:, :ND], ab[:, ND:]
    return _swiglu(a, b)


def cond_transition_train_12_345(x, cond, wa, wb, ws, wsc, bsc):
    """Differentiable 1+2|3+4+5 ConditionedTransition tail (fwd saves ab,out,scale; mirrored bwd)."""
    return ConditionedTransitionTail12345Function.apply(x, cond, wa, wb, ws, wsc, bsc)


# ============================================================================
# CLEAN triton-dgrad variant (1+2 | 3+4+5 mirror), wgrad on cuBLAS.
#
# Same logical grouping as above, but WITHOUT the gate/swiglu elementwise fused into the
# dgrad-GEMM *prologue*. The fused-prologue (_dh_gatebwd/_dx_swiglubwd) recomputes the
# transcendental grad (sigmoid/silu/silu') once per N-output-tile (grid_n times) AND still
# has to materialize dout/dscale/dab for the cuBLAS wgrad anyway — so the fusion buys nothing
# but serializes ALU into the WGMMA pipeline. Since wgrad needs those operands materialized
# regardless, the optimal structure is: one CHEAP single-pass elementwise kernel (which also
# feeds wgrad, free) + a CLEAN autotuned TF32 GEMM (full WGMMA throughput, no inner-loop
# transcendentals). This is the triton-dgrad analogue of the cuBLAS champion (training.py),
# differing only in dh/dcond/dx going through triton tl.dot instead of torch.matmul.
#
#   group 3+4+5 bwd:  dout,dscale = gate_bwd(out,scale,dy)   [elementwise, 1 pass; -> wgrad]
#                     dh   = dout   @ Ws      (clean GEMM, K=D, N=ND)
#                     dcond= dscale @ Wsc     (clean GEMM, K=D, N=DC)
#   group 1+2 bwd:    dab = swiglu_bwd(dh, ab)               [elementwise, 1 pass; -> wgrad]
#                     dx  = dab @ Wcat        (one clean cat-merged GEMM, K=2ND, N=K)
#   wgrad (cuBLAS):   dWs=dout^T@h ; dWsc=dscale^T@cond ; dWcat=dab^T@x ; db_sc=dscale.sum(0)
# ============================================================================
class ConditionedTransitionTail12345CleanFunction(torch.autograd.Function):
    """1+2|3+4+5 forward + clean triton-dgrad backward (no prologue fusion); wgrad cuBLAS."""

    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc):
        x = x.contiguous(); cond = cond.contiguous()
        wa = wa.contiguous(); wb = wb.contiguous(); ws = ws.contiguous()
        wsc = wsc.contiguous(); bsc = bsc.contiguous()
        h, ab = _fwd_expand_swiglu(x, wa, wb)          # 1+2 -> h, ab=[a|b]
        y, out, scale = _fwd_squeeze_gate(h, cond, ws, wsc, bsc)  # 3+4+5 -> y, out, scale
        wcat = torch.cat([wa, wb], dim=0)              # (2ND, K) for the cat-merged dx GEMM
        ctx.save_for_backward(x, cond, ab, h, out, scale, wcat, ws, wsc)
        ctx.ND = wa.shape[0]
        return y

    @staticmethod
    def backward(ctx, dy):
        x, cond, ab, h, out, scale, wcat, ws, wsc = ctx.saved_tensors
        ND = ctx.ND
        M, D = out.shape
        DC = cond.shape[1]
        K = x.shape[1]
        dy = dy.contiguous()

        # ---- bwd group 3+4+5: gate-bwd (1 pass, also feeds wgrad) + clean dgrad GEMMs ----
        dout, dscale = _gate_bwd(out, scale, dy)                       # (M,D),(M,D)
        dh = _dgemm(dout, ws, M, ND, D, ws.stride(0), ws.stride(1))    # dout @ Ws  -> (M,ND)
        dcond = _dgemm(dscale, wsc, M, DC, D, wsc.stride(0), wsc.stride(1))  # dscale @ Wsc -> (M,DC)
        # ---- bwd group 1+2: swiglu-bwd (1 pass, also feeds wgrad) + one clean cat-merged GEMM ----
        dab = _swiglu_bwd_pack(dh, ab)                                 # (M,2ND) [da|db]
        dx = _dgemm(dab, wcat, M, K, 2 * ND, wcat.stride(0), wcat.stride(1))  # dab @ Wcat -> (M,K)

        # ---- wgrad: cuBLAS reductions over M (left on cuBLAS per directive) ----
        db_sc = dscale.sum(0)
        dWs = dout.t() @ h                                             # (D, ND)
        dWsc = dscale.t() @ cond                                       # (D, DC)
        dWcat = dab.t() @ x                                            # (2ND, K)
        dWa, dWb = dWcat[:ND].contiguous(), dWcat[ND:].contiguous()
        return dx, dcond, dWa, dWb, dWs, dWsc, db_sc


def cond_transition_train_12_345_clean(x, cond, wa, wb, ws, wsc, bsc):
    """1+2|3+4+5 tail: clean triton-dgrad (elementwise + autotuned GEMM) backward, cuBLAS wgrad."""
    return ConditionedTransitionTail12345CleanFunction.apply(x, cond, wa, wb, ws, wsc, bsc)
