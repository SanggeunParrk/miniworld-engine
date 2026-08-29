"""SM100 (B200) port of the H100 v6 BIDIRECTIONAL trimul TRAINING path
(`bidir_training.BidirV6TriMul`), BYTE-FOR-BYTE FAITHFUL: identical autograd.Function structure,
op order, and layout contract. The ONLY difference from the H100 bidir path is the FRONT: the sm90
`trimul_inproj_cute_forward(bdll_direct=True, out_hidden=2h)` gated-GLU M-major store is replaced
by the sm100 equivalent `trimul_front_sm100_train` (non-gated m-major GEMM + Triton GLU), which
produces the SAME (left_bdll, right_bdll, preact_bdll) at out_hidden=H=2h with 0 transposes.

Everything else is the H100 bidir path VERBATIM (all arch-agnostic; each op verified to run on
sm100 by the single-dir v12 port):
  - TWO torch.bmm contractions (outgoing on [:h] = L@Rᵀ, incoming on [h:] = Lᵀ@R) via transpose
    VIEWS (zero-copy), then cat -> tri (H, L, L). bwd uses the CONTIGUOUS-grad formulas.
  - `_te_forward/_te_backward` (LN_out+@Wp, arch-agnostic cuBLAS+Triton, m-major in -> m-major out)
    makes `d_tri = d_view.t().reshape(H,L,L)` a FREE contiguous view.
  - `gate_elem_*` (triton), `front_bwd_dW` (triton, dW cuBLAS).
  - dxn = (dconcᵀ@W_stack) + (d_glogit@Wgᵀ) via one cuBLAS `mm`+`addmm_` (v12 style; the H100
    bidir `dispatch.pick`/quack-cute C-add wrapper is dropped — the addmm eager path never
    regresses and avoids the quack import in backward).

Physical transpose kernels: 0. The current bidir v2 (`bidir_training_b200.py`) has 8/iter
(front×2 gated store, tri->LN ×1, d_tri ×1, d_left/d_right ×4) via `_fast_T`; the v6-faithful
bmm+transpose-view + m-major LN + contiguous-grad structure eliminates all of them.

B=1, bf16 in / fp32 acc (fp32 LN stats) / bf16 out; h = d_hidden per direction, H = 2h.
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


class BidirBackHalfSm100(torch.autograd.Function):
    """== H100 `BidirBackHalf`, sm100 front. Front (sm100 non-gated GEMM+GLU, out_hidden=2h) ->
    2 contractions (outgoing [:h] / incoming [h:], cuBLAS bmm, contiguous-grad) -> LN_out+@Wp (te)
    -> gate (triton), as ONE Function. Backward fuses the gate input-grad + the x_n-add into the
    front dxn addmm (cuBLAS); dW stays cuBLAS."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, b_lr, eps, h):
        B, L, _, D = x_n.shape
        M = B * L * L
        H = 2 * h
        left, right, sg = trimul_front_sm100_train_sig(x_n, b_lr, H)  # bdll, 0 transposes; sg=σ(gate)
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        tri = torch.empty(H, L, L, dtype=lf.dtype, device=lf.device)  # (H, L, L)
        torch.bmm(lf[:h], rf[:h].transpose(1, 2), out=tri[:h])   # outgoing: O = L @ RT
        torch.bmm(lf[h:].transpose(1, 2), rf[h:], out=tri[h:])   # incoming: O = LT @ R
        view = tri.reshape(H, M).t()                        # (M, H) m-major
        proj, te_xn, mean_out, rstd_out = _te_forward(view, ln_out_w, ln_out_b, Wp, None, eps)
        # FUSED gate: y = sigmoid(x_n@Wg)⊙proj in ONE quack launch (no glogit HBM round-trip,
        # no separate _gate_mul). Saves the PREACT (glogit); bwd recomputes gate=σ(preact).
        y, gate_src = gate_elem_quack_fused(x_n.reshape(M, D), proj, Wg, return_preact=True)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              sg, lf, rf, tri, te_xn, mean_out, rstd_out, gate_src, proj)
        ctx.eps, ctx.h = eps, h
        return y.reshape(B, L, L, D)

    @staticmethod
    def backward(ctx, gy):
        (x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
         sg, lf, rf, tri, te_xn, mean_out, rstd_out, gate_src, proj) = ctx.saved_tensors
        B, L, _, D = x_n.shape
        M = B * L * L
        h = ctx.h
        H = 2 * h
        gy = gy.reshape(M, D)

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
        view = tri.reshape(H, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        del d_proj, view
        d_tri = d_view.t().reshape(H, L, L)
        del d_view

        # contraction bwd (contiguous-grad formulas), split outgoing/incoming
        d_o_out, d_o_in = d_tri[:h], d_tri[h:]
        del d_tri
        lo, ro, li, ri = lf[:h], rf[:h], lf[h:], rf[h:]
        d_left = torch.empty(H, L, L, dtype=lf.dtype, device=lf.device)
        d_right = torch.empty(H, L, L, dtype=lf.dtype, device=lf.device)
        torch.bmm(d_o_out, ro, out=d_left[:h])            # outgoing: O=lo@roT -> dl=G@R
        torch.bmm(ri, d_o_in.transpose(1, 2), out=d_left[h:])  # incoming: O=liT@ri -> dl=R@GT
        torch.bmm(d_o_out.transpose(1, 2), lo, out=d_right[:h]) #                       dr=GT@L
        torch.bmm(li, d_o_in, out=d_right[h:])            #                       dr=L@G
        d_left = d_left.reshape(B, H, L, L)
        d_right = d_right.reshape(B, H, L, L)

        # front bwd, then dxn = (dconcᵀ@W_stack) + (d_glogit@Wgᵀ) with the gate's dx_gate add
        # FUSED into one cuBLAS addmm epilogue (== v12 single-dir). dW stays cuBLAS.
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW_sig(
            d_left, d_right, lf, rf, sg, x_n, WL, WLg, WR, WRg)
        dx_n = torch.mm(d_glogit, Wg.t())                         # (M, D) gate term
        del d_glogit
        dx_n.addmm_(dconc.t(), W_stack)                           # += dconcᵀ@W_stack, in-place
        del W_stack, dconc
        dx_n = dx_n.reshape(B, L, L, D)
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None)


# No wrapper: every launch reachable from here is an ``opaque`` op at its own definition,
# so Dynamo traces straight through this. It could never have BEEN an op itself -- see
# ``kernels._compile`` -- but it does not need to be.
def bidir_forward_sm100(pair, WL, WLg, WR, WRg, Wg, Wp_nn, ln_in_w, ln_in_b,
                        ln_out_w, ln_out_b, eps, b_lr, h, row_scale=None):
    # AF pair-mask folded into LN_in as a row_scale (FREE), rs=None -> plain LN. (== H100 bidir)
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps, row_scale=row_scale)
    return BidirBackHalfSm100.apply(x_n, WL, WLg, WR, WRg, Wg, Wp_nn, ln_out_w, ln_out_b,
                                    b_lr, eps, h)


class BidirV6TriMulSm100(nn.Module):
    """SM100 v6-faithful bidirectional trimul (outgoing+incoming fused, fwd+bwd).
    Drop-in for `bidir_training.BidirV6TriMul`. Built from a base
    BidirectionalTriangleMultiplication's weights. bf16, B=1, h=d_hidden per direction."""

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

    def forward(self, pair, mask=None):
        b_lr = prepack_lr_operand_sm100(self.WL, self.WLg, self.WR, self.WRg)
        row_scale = None
        if mask is not None:
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)        # [B, L, L]
            row_scale = m.reshape(-1).to(pair.dtype)           # [M]
        return bidir_forward_sm100(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp_nn,
                                   self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b,
                                   self.eps, b_lr, self.h, row_scale)
