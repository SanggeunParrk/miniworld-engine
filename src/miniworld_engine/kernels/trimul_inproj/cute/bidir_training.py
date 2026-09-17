"""Bidirectional trimul TRAINING (fwd+bwd) — v6 split-back stack, bidirectional dims, with the
back-half MERGED into one autograd.Function (`BidirBackHalf`) so the backward can FUSE across
the gate/front boundary.

    x_n  = LayerNorm_in(pair)                  # triton_layernorm (autograd, repo kernel)
    y    = BidirBackHalf(x_n, ...)             # front + 2 contractions + LN_out + gate, fused bwd

Why merge: x_n feeds BOTH the front (start) and the gate (end); as separate autograd Functions
the engine sums their two x_n-grads with a standalone add kernel, and dx_gate is materialized.
Merged, the backward computes everything in order and FUSES the gate input-grad + the add into
the front's dxn GEMM:  dx_n = (d_concatᵀ @ W_stack) + (d_glogit @ Wgᵀ)  done as
  dxn_front = quack_gemm(d_concatᵀ, W_stack)            # cute
  dx_n      = gemm_act(d_glogit, Wgᵀ, C=dxn_front, act=None)   # cute, C-add epilogue
→ no separate dx_gate GEMM output, no separate add kernel. **dW (dWL/.../dWg/dWp) stay cuBLAS**
(huge-K reductions; quack is 2.6-5x slower there — measured). B=1, bf16, h=d_hidden per dir.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from . import output_training
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _te_backward, _te_forward
from miniworld_engine.kernels.trimul_inproj.cute import _bdll_patch, _gate_mul_patch, dispatch
from miniworld_engine.kernels.trimul_inproj.cute.contract import packed_backward, packed_forward
from miniworld_engine.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_bwd_ew, gate_elem_train, ones_dropscale,
)


class BidirBackHalf(torch.autograd.Function):
    """Front (cute gated GEMM, out_hidden=2h) → 2 contractions (outgoing [:h] / incoming [h:],
    packed bmm, contiguous-grad) → LN_out+@Wp (te) → gate (triton), as ONE Function. Backward
    fuses the gate input-grad + the x_n-add into the front dxn GEMM (cute); dW stays cuBLAS."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, b_lr, eps, h,
                residual, dropscale=None, pair_mask=None):
        # residual [B,L,L,D] (== module input pair) + dropscale [1,1,L,D] (drop_row mask/(1-p),
        # broadcast over i) fuse the pairformer residual+dropout into the gate; bwd returns d_residual=gy.
        B, L, _, D = x_n.shape
        M = B * L * L
        H = 2 * h
        left, right, preact = trimul_inproj_cute_forward(
            x_n, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False,
            b_lr=b_lr, out_hidden=H, return_preact=True, pair_mask=pair_mask)
        # Mask is fused into the front GEMM stores; saved preactivations stay raw.
        ctx.pair_mask = pair_mask
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        tri = packed_forward(lf, rf, h)
        view = tri.reshape(H, M).t()                            # (M, H) m-major
        ctx.native_output = output_training.supported(view, Wp)
        if ctx.native_output:
            proj, te_xn, mean_out, rstd_out = output_training.forward(view, ln_out_w, ln_out_b, Wp, eps)
        else:
            proj, te_xn, mean_out, rstd_out = _te_forward(view, ln_out_w, ln_out_b, Wp, None, eps)
        y, gate = gate_elem_train(
            x_n.reshape(M, D), proj, Wg, residual, dropscale, seq_len=L)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj, ln_out_b)
        ctx.eps, ctx.h = eps, h
        ctx.dropscale, ctx.seq_len = dropscale, L
        return y.reshape(B, L, L, D)

    @staticmethod
    def backward(ctx, gy):
        (x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
         preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj, ln_out_b) = ctx.saved_tensors
        from miniworld_engine.kernels._quack_compat import gemm as _quack_gemm
        from miniworld_engine.kernels._quack_compat import gemm_act as _quack_gemm_act
        B, L, _, D = x_n.shape
        M = B * L * L
        h = ctx.h
        H = 2 * h
        gy = gy.reshape(M, D)

        # ② gate bwd (elementwise only — dx_gate is fused into the dxn GEMM below). Dropout scale
        # is applied to gy inside; the residual identity grad is returned as d_residual = gy.
        d_proj, d_glogit = gate_elem_bwd_ew(gy, proj, gate, ctx.dropscale, ctx.seq_len)
        dWg = dispatch.mm("dWg", x_n.reshape(M, D).t(), d_glogit)   # (D, D) huge-K → cuBLAS

        # ① LN_out + @Wp bwd
        view = tri.reshape(H, M).t()
        if ctx.native_output:
            d_view, dLNo_w, dLNo_b, dWp = output_training.backward(
                d_proj, proj, te_xn, rstd_out, ln_out_w, ln_out_b, Wp)
        else:
            d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
                d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        # `del` after last use, inserted where no reference to the name remains anywhere below.
        # autograd frees an intermediate when its consumer node has run; this function holds every
        # local until it returns, and these are pair-shaped -- 144 MiB each at B=1 L=768 d=128
        # bf16. Measured on the triton bidirectional twin: 1,008 MiB off a 7,662 MiB peak.
        del d_proj, view
        d_tri = d_view.t().reshape(H, L, L)
        del d_view

        # contraction bwd (contiguous-grad formulas), split outgoing/incoming
        d_left, d_right = packed_backward(d_tri, lf, rf, h)
        d_left = d_left.reshape(B, H, L, L)
        d_right = d_right.reshape(B, H, L, L)
        del d_tri

        # front bwd: d_concat + dW (cuBLAS) + W_stack; dxn fused with the gate add.
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
            d_left, d_right, preact, x_n, WL, WLg, WR, WRg, pair_mask=ctx.pair_mask)
        del d_left, d_right
        dconcT = dconc.t()
        del dconc
        Wg_t = Wg.t()

        def _dxn_cute():   # quack: dxn_front + (d_glogit@Wgᵀ) fused via C-add epilogue (cute)
            dxn_front = _quack_gemm(dconcT, W_stack)
            return _quack_gemm_act(d_glogit, Wg_t, C=dxn_front, activation=None,
                                   store_preact=False)[1]

        def _dxn_cublas():  # cuBLAS: accumulate the dx_gate add IN-PLACE. Out-of-place
            # torch.addmm(C, A, B) stages β·C by copying C (a full (M,D)=268MB DtoD memcpy) into
            # the output before the GEMM; seeding the buffer with the gate term and accumulating
            # dconcᵀ@W_stack in-place removes that copy.
            dx = torch.mm(d_glogit, Wg_t)
            dx.addmm_(dconcT, W_stack)
            return dx

        # dispatch: cuBLAS wins small L (quack launch overhead), cute ≈/wins large L.
        dx_n = dispatch.pick("dxn", (M, 4 * H + D, D),
                             [("cute", _dxn_cute), ("cublas", _dxn_cublas)],
                             operands=(dconcT, W_stack, d_glogit, Wg_t),
                             case="triangle_multiplication_bidirectional").reshape(B, L, L, D)
        d_residual = gy.reshape(B, L, L, D)
        del gy
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None,
                d_residual, None, None)


# No wrapper: every launch reachable from here is an ``opaque`` op at its own definition,
# so Dynamo traces straight through this. It could never have BEEN an op itself -- see
# ``kernels._compile`` -- but it does not need to be.
def bidir_forward(pair, WL, WLg, WR, WRg, Wg, Wp_nn, ln_in_w, ln_in_b,
                  ln_out_w, ln_out_b, eps, b_lr, h, row_scale=None,
                  dropscale=None, eps_out=None):
    _bdll_patch.apply()
    _gate_mul_patch.apply()
    # Keep LN/output gate unmasked; mask front outputs and their gradients instead.
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps)
    # `pair` is passed BOTH as the LN input and as the residual (which is never absent) ->
    # autograd accumulates its grad from both paths.
    # A training entry always carries a drop scale, ones when p_drop is 0 -- that is what
    # keeps `gate_elem_train` and `gate_elem_bwd_ew` flagless. Measured cost 0.2%.
    if dropscale is None:
        dropscale = ones_dropscale(pair.shape[1], pair.shape[-1], pair)
    return BidirBackHalf.apply(x_n, WL, WLg, WR, WRg, Wg, Wp_nn, ln_out_w, ln_out_b, b_lr, eps if eps_out is None else eps_out, h,
                               pair, dropscale, row_scale)


class BidirV6TriMul(nn.Module):
    """Trainable bidirectional trimul (outgoing+incoming fused), v6 split back + merged BackHalf.
    Built from a base BidirectionalTriangleMultiplication's weights. bf16. Weights in x@W form."""

    def __init__(self, base):
        super().__init__()
        b = base
        self.h = b.d_hidden
        self.WL = nn.Parameter(b.to_left.weight.t().contiguous())        # (d, 2h)
        self.WLg = nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = nn.Parameter(b.to_gate.weight.t().contiguous())        # (d, d)
        self.Wp_nn = nn.Parameter(b.to_out.weight.detach().clone())      # (d, 2h) nn.Linear form
        self.ln_in_w = nn.Parameter(b.ln_pair.weight.detach().clone())
        self.ln_in_b = nn.Parameter(b.ln_pair.bias.detach().clone())
        self.ln_out_w = nn.Parameter(b.ln_out.weight.detach().clone())   # (2h,)
        self.ln_out_b = nn.Parameter(b.ln_out.bias.detach().clone())
        self.eps = b.ln_pair.eps
        self.eps_out = b.ln_out.eps

    def forward(self, pair, mask=None, dropscale=None):
        b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)
        row_scale = None
        if mask is not None:
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)        # [B, L, L]
            row_scale = m.reshape(-1).to(pair.dtype)           # [M]
        return bidir_forward(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp_nn,
                             self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b,
                             self.eps, b_lr, self.h, row_scale,
                             dropscale=dropscale, eps_out=self.eps_out)
