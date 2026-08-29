"""SM100 (B200) port of the H100 v6 single-direction trimul TRAINING path
(`v6_training_merged.V6TriMulMerged`), BYTE-FOR-BYTE FAITHFUL: identical autograd.Function
structure, op order, and layout contract. The ONLY difference from v6 is the FRONT: the sm90
`trimul_inproj_cute_forward(bdll_direct=True)` gated-GLU M-major store is replaced by the
sm100 equivalent `trimul_front_sm100_train` (non-gated m-major GEMM + Triton GLU), which
produces the SAME (left_bdll, right_bdll, preact_bdll) with 0 transposes. Everything else —
torch.bmm contractions (contiguous-grad formulas), `_te_forward/_te_backward` (LN_out+@Wp,
arch-agnostic cuBLAS+Triton, m-major in->m-major out), `gate_elem_*`, `front_bwd_dW`, and the
`torch.mm`+`addmm_` dxn — is v6 VERBATIM (all arch-agnostic; verified to run on sm100).

Physical transpose kernels: 0 (== H100 v6). See front_train_sm100.py for why the sm90 gated
bdll-direct store fails on sm100 but the non-gated store is bit-correct.
"""
from __future__ import annotations


import torch
import torch.nn as nn

from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _te_backward, _te_forward
from miniworld_engine.kernels.trimul_inproj.cute.front_train_sm100 import (
    prepack_lr_operand_sm100, trimul_front_sm100_train_sig,
)
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW_sig
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_bwd_ew, gate_elem_quack_fused,
)


class _SingleBackHalfSm100(torch.autograd.Function):
    """== v6 `_SingleBackHalf`, sm100 front. Front (sm100 gated GEMM, out_hidden=D) ->
    1 contraction (cuBLAS bmm, contiguous-grad) -> LN_out+@Wp (te) -> gate (triton), as ONE
    Function. Backward fuses the gate input-grad + the x_n-add into the front dxn addmm."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, b_lr, eps, direction_flag):
        B, L, _, D = x_n.shape
        M = B * L * L
        left, right, sg = trimul_front_sm100_train_sig(x_n, b_lr, D)  # bdll, 0 transposes; sg=σ(gate)
        lf = left.reshape(D, L, L)
        rf = right.reshape(D, L, L)
        if direction_flag == 0:                       # outgoing: O = L @ Rᵀ
            tri = torch.bmm(lf, rf.transpose(1, 2))
        else:                                         # incoming: O = Lᵀ @ R
            tri = torch.bmm(lf.transpose(1, 2), rf)
        view = tri.reshape(D, M).t()                  # (M, D) m-major
        proj, te_xn, mean_out, rstd_out = _te_forward(view, ln_out_w, ln_out_b, Wp, None, eps)
        # FUSED gate: y = sigmoid(x_n@Wg)⊙proj in ONE quack launch (no glogit HBM round-trip,
        # no separate _gate_mul). Saves the PREACT (glogit); bwd recomputes gate=σ(preact).
        y, gate_src = gate_elem_quack_fused(x_n.reshape(M, D), proj, Wg, return_preact=True)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              sg, lf, rf, tri, te_xn, mean_out, rstd_out, gate_src, proj)
        ctx.eps, ctx.direction_flag = eps, direction_flag
        return y.reshape(B, L, L, D)

    @staticmethod
    def backward(ctx, gy):
        (x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
         sg, lf, rf, tri, te_xn, mean_out, rstd_out, gate_src, proj) = ctx.saved_tensors
        B, L, _, D = x_n.shape
        M = B * L * L
        df = ctx.direction_flag
        gy = gy.reshape(M, D).contiguous()  # guard: autograd may hand a non-contiguous / broadcast (.sum) grad; kernels assume contiguous

        # ② gate bwd (elementwise; dx_gate add is fused into the dxn addmm below).
        # gate_src is the saved PREACT (glogit); recompute gate=σ(preact) in-kernel.
        d_proj, d_glogit = gate_elem_bwd_ew(gy, proj, gate_src, from_preact=True)
        # `del` after last use, inserted where no reference to the name remains anywhere below.
        # autograd frees an intermediate when its consumer node has run; this function holds every
        # local until it returns, and these are pair-shaped -- 144 MiB each at B=1 L=768 d=128
        # bf16. Measured on the triton bidirectional twin: 1,008 MiB off a 7,662 MiB peak.
        del gy
        dWg = x_n.reshape(M, D).t() @ d_glogit                     # (D,D) huge-K -> cuBLAS

        # ① LN_out + @Wp bwd
        view = tri.reshape(D, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        del d_proj, view
        d_tri = d_view.t().reshape(D, L, L)
        del d_view

        # contraction bwd (contiguous-grad formulas), single direction
        if df == 0:
            d_lf = torch.bmm(d_tri, rf)                            # G @ R
            d_rf = torch.bmm(d_tri.transpose(1, 2), lf)            # Gᵀ @ L
        else:
            d_lf = torch.bmm(rf, d_tri.transpose(1, 2))           # R @ Gᵀ
            d_rf = torch.bmm(lf, d_tri)                            # L @ G
        del d_tri
        d_left = d_lf.reshape(B, D, L, L)
        d_right = d_rf.reshape(B, D, L, L)

        # front bwd, then dxn = (dconcᵀ@W_stack) + (d_glogit@Wgᵀ) with the gate's dx_gate add
        # FUSED into one cuBLAS addmm epilogue (== v6). dW stays cuBLAS.
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW_sig(
            d_left, d_right, lf, rf, sg, x_n, WL, WLg, WR, WRg)
        del d_left, d_right
        dx_n = torch.mm(d_glogit, Wg.t())                         # (M, D) gate term
        del d_glogit
        dx_n.addmm_(dconc.t(), W_stack)                           # += dconcᵀ@W_stack, in-place
        del W_stack, dconc
        dx_n = dx_n.reshape(B, L, L, D)
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None)


# No wrapper: every launch reachable from here is an ``opaque`` op at its own definition,
# so Dynamo traces straight through this. It could never have BEEN an op itself -- see
# ``kernels._compile`` -- but it does not need to be.
def v6_forward_merged_sm100(pair, WL, WLg, WR, WRg, Wg, Wp_nn, ln_in_w, ln_in_b,
                            ln_out_w, ln_out_b, eps, b_lr, direction="out", row_scale=None):
    # AF pair-mask folded into LN_in as a row_scale (FREE), rs=None -> plain LN. (== v6)
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps, row_scale=row_scale)
    flag = 0 if direction == "out" else 1
    return _SingleBackHalfSm100.apply(x_n, WL, WLg, WR, WRg, Wg, Wp_nn, ln_out_w, ln_out_b,
                                      b_lr, eps, flag)


class V6TriMulMergedSm100(nn.Module):
    """SM100 v6-faithful single-direction trimul (fwd+bwd). Drop-in for V6TriMulMerged."""

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

    def forward(self, pair, mask=None):
        b_lr = prepack_lr_operand_sm100(self.WL, self.WLg, self.WR, self.WRg)
        row_scale = None
        if mask is not None:
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)        # [B, L, L]
            row_scale = m.reshape(-1).to(pair.dtype)           # [M]
        return v6_forward_merged_sm100(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg,
                                       self.Wp_nn, self.ln_in_w, self.ln_in_b, self.ln_out_w,
                                       self.ln_out_b, self.eps, b_lr, self.direction, row_scale)
