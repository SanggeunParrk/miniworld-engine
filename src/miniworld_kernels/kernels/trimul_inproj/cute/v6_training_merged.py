"""Single-direction trimul TRAINING (fwd+bwd) — v6 split-back stack with the back-half MERGED
into ONE autograd.Function (`_SingleBackHalf`), mirroring `bidir_training.BidirBackHalf` but for
single direction (out_hidden=D, one contraction). Collapses front + contraction + LN_out + gate
into one Function so the backward FUSES the gate input-grad + the x_n-add into one cuBLAS addmm:

    dx_n = (d_concatᵀ @ W_stack) + (d_glogit @ Wgᵀ)
         = addmm(dconcᵀ @ W_stack, d_glogit, Wgᵀ)    # cuBLAS mm + addmm (the +add is the epilogue)

vs the old path (separate _FrontFn/_TriContract/GateElem Functions) this removes: the autograd
sum kernel (two x_n-grad contributions) and 3 autograd nodes of Python overhead. cuBLAS addmm
(not quack cute C-add: cute's 2 heavy interface launches lose in eager) and NO dispatch.pick
wrappers (added host overhead in the host-bound small-L regime). Measured win: eager small-L
+7-8% (real-harness L=384 dtv1 tie 1.04x -> 1.11x), cudagraph tie, never regresses. The deeper
d_glogit-into-d_concat fold (collapse to 1 GEMM + dWg) was tried & reverted — non-coalesced
relayout cost cancels small-L and regresses large-L. dW stays cuBLAS. B=1, bf16, square D.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm_linear.te_style import _te_backward, _te_forward
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch, _gate_mul_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_bwd_ew, gate_elem_triton,
)


class _SingleBackHalf(torch.autograd.Function):
    """Front (cute gated GEMM, out_hidden=D) -> 1 contraction (cuBLAS bmm, contiguous-grad) ->
    LN_out+@Wp (te) -> gate (triton), as ONE Function. Backward fuses the gate input-grad + the
    x_n-add into the front dxn GEMM (cute C-add epilogue). dW stays cuBLAS."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, b_lr, eps,
                direction_flag, residual=None, dropscale=None):
        # residual [B,L,L,D] (== module input pair) and dropscale [1,1,L,D] (== drop_row
        # mask/(1-p), broadcast over the i-index) fuse the pairformer residual+dropout into the
        # gate store: y = residual + dropscale ⊙ trimul(pair). Backward returns d_residual = gy.
        B, L, _, D = x_n.shape
        M = B * L * L
        left, right, preact = trimul_inproj_cute_forward(
            x_n, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False,
            b_lr=b_lr, return_preact=True)
        lf = left.reshape(D, L, L)
        rf = right.reshape(D, L, L)
        if direction_flag == 0:                       # outgoing: O = L @ Rᵀ
            tri = torch.bmm(lf, rf.transpose(1, 2))
        else:                                         # incoming: O = Lᵀ @ R
            tri = torch.bmm(lf.transpose(1, 2), rf)
        view = tri.reshape(D, M).t()                  # (M, D) m-major
        proj, te_xn, mean_out, rstd_out = _te_forward(view, ln_out_w, ln_out_b, Wp, None, eps)
        y, gate = gate_elem_triton(
            x_n.reshape(M, D), proj, Wg, return_gate=True,
            residual=residual, dropscale=dropscale, seq_len=L)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.direction_flag = eps, direction_flag
        ctx.dropscale, ctx.add_residual, ctx.seq_len = dropscale, residual is not None, L
        return y.reshape(B, L, L, D)

    @staticmethod
    def backward(ctx, gy):
        (x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
         preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj) = ctx.saved_tensors
        B, L, _, D = x_n.shape
        M = B * L * L
        df = ctx.direction_flag
        gy = gy.reshape(M, D).contiguous()  # guard: autograd may hand a non-contiguous / broadcast (.sum) grad; kernels assume contiguous

        # ② gate bwd (elementwise; dx_gate add is fused into the dxn addmm below). With dropout,
        # the drop scale is applied to gy inside (grad of y = dropscale ⊙ trimul); the residual
        # identity grad is returned separately as d_residual = gy.
        d_proj, d_glogit = gate_elem_bwd_ew(
            gy, proj, gate, dropscale=ctx.dropscale, seq_len=ctx.seq_len)
        dWg = x_n.reshape(M, D).t() @ d_glogit                     # (D,D) huge-K -> cuBLAS

        # ① LN_out + @Wp bwd
        view = tri.reshape(D, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        d_tri = d_view.t().reshape(D, L, L)

        # contraction bwd (contiguous-grad formulas), single direction
        if df == 0:
            d_lf = torch.bmm(d_tri, rf)                            # G @ R
            d_rf = torch.bmm(d_tri.transpose(1, 2), lf)            # Gᵀ @ L
        else:
            d_lf = torch.bmm(rf, d_tri.transpose(1, 2))           # R @ Gᵀ
            d_rf = torch.bmm(lf, d_tri)                            # L @ G
        d_left = d_lf.reshape(B, D, L, L)
        d_right = d_rf.reshape(B, D, L, L)

        # front bwd, then dxn = (dconcᵀ@W_stack) + (d_glogit@Wgᵀ) with the gate's dx_gate add
        # FUSED into one cuBLAS addmm epilogue. (The deeper d_glogit-into-d_concat fold — one GEMM
        # for both terms + dWg — was tried and REVERTED: its non-coalesced (M,D)->(D,M) relayout
        # cost cancels the saved launch at small L and regresses ~16% at large L. addmm = sweet
        # spot: fuse only the cheap add. dW stays cuBLAS.)
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
            d_left, d_right, preact, x_n, WL, WLg, WR, WRg)
        # dx_n = dconcᵀ@W_stack + d_glogit@Wgᵀ. Compute the gate term into a fresh (M,D) buffer,
        # then accumulate the front term IN-PLACE. Out-of-place torch.addmm(C, A, B) stages β·C by
        # copying C (a full (M,D)=268MB DtoD memcpy, ~175us/step at L=1024) into the output before
        # the GEMM; addmm_ accumulates into the buffer that already holds C, so that copy is gone.
        dx_n = torch.mm(d_glogit, Wg.t())                         # (M, D) gate term
        dx_n.addmm_(dconc.t(), W_stack)                           # += dconcᵀ@W_stack, in-place
        dx_n = dx_n.reshape(B, L, L, D)
        # residual identity: d(y=residual + dropscale⊙trimul)/d(residual) = 1 -> d_residual = gy.
        d_residual = gy.reshape(B, L, L, D) if ctx.add_residual else None
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None,
                d_residual, None)


@torch.compiler.disable
def v6_forward_merged(pair, WL, WLg, WR, WRg, Wg, Wp_nn, ln_in_w, ln_in_b,
                      ln_out_w, ln_out_b, eps, b_lr, direction="out", row_scale=None,
                      add_residual=False, dropscale=None):
    _bdll_patch.apply()
    _gate_mul_patch.apply()
    # AF pair-mask folded into LN_in as a row_scale (FREE — no separate (M,D) multiply): x_n =
    # LN(pair)*rs, so proj(0)=0 -> left/right=0 at masked positions (== AF's mask*projection); the
    # masked grad is folded into the LN backward. rs=None -> plain LN.
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps, row_scale=row_scale)
    flag = 0 if direction == "out" else 1
    # Fuse the pairformer residual+dropout: pair is passed BOTH as the LN input (x_n) and as the
    # residual, so autograd accumulates pair's grad from the trimul path AND the identity path.
    residual = pair if add_residual else None
    return _SingleBackHalf.apply(x_n, WL, WLg, WR, WRg, Wg, Wp_nn, ln_out_w, ln_out_b,
                                 b_lr, eps, flag, residual, dropscale)


class V6TriMulMerged(nn.Module):
    """Trainable single-direction trimul, v6 split back + MERGED back-half. Drop-in for V6TriMul."""

    def __init__(self, base, direction="out"):
        super().__init__()
        b = base
        self.direction = direction
        self.WL = nn.Parameter(b.to_left.weight.t().contiguous())
        self.WLg = nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = nn.Parameter(b.to_gate.weight.t().contiguous())
        self.Wp_nn = nn.Parameter(b.to_out.weight.detach().clone())
        self.ln_in_w = nn.Parameter(b.ln_pair.weight.detach().clone())
        self.ln_in_b = nn.Parameter(b.ln_pair.bias.detach().clone())
        self.ln_out_w = nn.Parameter(b.ln_out.weight.detach().clone())
        self.ln_out_b = nn.Parameter(b.ln_out.bias.detach().clone())
        self.eps = b.ln_pair.eps

    def forward(self, pair, mask=None, add_residual=False, dropscale=None):
        b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)
        row_scale = None
        if mask is not None:
            # mask: [B, L] bool -> per-row pair scale [M] (valid iff both endpoints valid)
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)        # [B, L, L]
            row_scale = m.reshape(-1).to(pair.dtype)           # [M]
        return v6_forward_merged(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp_nn,
                                 self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b,
                                 self.eps, b_lr, self.direction, row_scale,
                                 add_residual=add_residual, dropscale=dropscale)
