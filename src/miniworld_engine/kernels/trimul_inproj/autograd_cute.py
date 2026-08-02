"""B1 — autograd.Function wrapping the REAL cute forward, manual backward.

Forward uses the cute trimul_inproj kernel for the left+right gated GEMM (the
heavy front op); LN / bmm / gate / proj are torch for now (LN is cheap; bmm is
cuBLAS). Backward uses the B0-verified gradient formulas (recomputing the front
proj/gate logits pL,gL,pR,gR from x_n, since the cute kernel doesn't expose them).

This is the full-mode pipeline: trainable ours + first fwd+bwd timing. Backward
GEMMs are plain torch.mm (cuBLAS) here — the dt-v1-style concat + fused-EW
optimization comes next, validated against this (and B0).
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.trimul_inproj.autograd import _ln_bwd, _ln_fwd
from miniworld_engine.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)


class TriMulCuteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                ln_out_w, ln_out_b, eps, b_lr, mask2d):
        B, L, _, D = x.shape
        x_n, mean_in, rstd_in, xhat_in = _ln_fwd(x, ln_in_w, ln_in_b, eps)
        if mask2d is not None:
            x_n = x_n * mask2d
        # cute front: left/right in bdll (B,D,L,L)
        left_b, right_b, _ = trimul_inproj_cute_forward(
            x_n, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False, b_lr=b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)        # (B,D,L,L)
        tri_lld = tri.permute(0, 2, 3, 1).contiguous()               # (B,L,L,D)
        out_n, mean_out, rstd_out, xhat_out = _ln_fwd(tri_lld, ln_out_w, ln_out_b, eps)
        gate = torch.sigmoid(x_n @ Wg)
        proj = out_n @ Wp
        y = proj * gate

        ctx.save_for_backward(x_n, xhat_in, rstd_in, WL, WLg, WR, WRg, Wg, Wp,
                              left_b, right_b, xhat_out, rstd_out, ln_out_w, out_n, gate, proj)
        ctx.ln_in_w = ln_in_w
        ctx.has_mask = mask2d is not None
        ctx.mask2d = mask2d
        ctx.shape = (B, L, D)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x_n, xhat_in, rstd_in, WL, WLg, WR, WRg, Wg, Wp,
         left_b, right_b, xhat_out, rstd_out, ln_out_w, out_n, gate, proj) = ctx.saved_tensors
        ln_in_w = ctx.ln_in_w
        B, L, D = ctx.shape
        M = B * L * L

        def flat(t):
            return t.reshape(M, D)

        # ① back-half
        d_proj = dy * gate
        d_gate = dy * proj
        d_glog = d_gate * gate * (1 - gate)
        d_out_n = d_proj @ Wp.t()
        dWp = flat(out_n).t() @ flat(d_proj)
        dx_n = d_glog @ Wg.t()
        dWg = flat(x_n).t() @ flat(d_glog)
        d_tri_lld, dWln_out, dBln_out = _ln_bwd(d_out_n, xhat_out, rstd_out, ln_out_w)
        d_tri = d_tri_lld.permute(0, 3, 1, 2).contiguous()           # (B,D,L,L)

        # ② bmm bwd (bdll)
        d_left_b = torch.einsum("bdij,bdjk->bdik", d_tri, right_b)
        d_right_b = torch.einsum("bdij,bdik->bdjk", d_tri, left_b)
        d_left = d_left_b.permute(0, 2, 3, 1).contiguous()
        d_right = d_right_b.permute(0, 2, 3, 1).contiguous()

        # ③ front gated-GEMM bwd — recompute pL,gL,pR,gR from x_n
        def gated_bwd(d_out, Wp_proj, Wg_gate):
            p = x_n @ Wp_proj
            g = torch.sigmoid(x_n @ Wg_gate)
            d_p = d_out * g
            d_glogit = (d_out * p) * g * (1 - g)
            dxn = d_p @ Wp_proj.t() + d_glogit @ Wg_gate.t()
            dWproj = flat(x_n).t() @ flat(d_p)
            dWgate = flat(x_n).t() @ flat(d_glogit)
            return dxn, dWproj, dWgate

        dxn_L, dWL, dWLg = gated_bwd(d_left, WL, WLg)
        dxn_R, dWR, dWRg = gated_bwd(d_right, WR, WRg)
        dx_n = dx_n + dxn_L + dxn_R

        if ctx.has_mask:
            dx_n = dx_n * ctx.mask2d

        # ④ LN_in bwd
        dx, dWln_in, dBln_in = _ln_bwd(dx_n, xhat_in, rstd_in, ln_in_w)
        return (dx, dWL, dWLg, dWR, dWRg, dWg, dWp,
                dWln_in, dBln_in, dWln_out, dBln_out, None, None, None)


class TriMulCute(torch.nn.Module):
    """Trainable cute trimul (outgoing). Weights in nn.Linear (N,K) form internally;
    the Function takes x@W form (=.T). bf16."""

    def __init__(self, base):
        super().__init__()
        b = base
        self.WL, self.WLg = b.to_left.weight.t().contiguous(), b.to_left_gate.weight.t().contiguous()
        self.WR, self.WRg = b.to_right.weight.t().contiguous(), b.to_right_gate.weight.t().contiguous()
        self.Wg, self.Wp = b.to_gate.weight.t().contiguous(), b.to_out.weight.t().contiguous()
        self.ln_in_w, self.ln_in_b = b.ln_pair.weight, b.ln_pair.bias
        self.ln_out_w, self.ln_out_b = b.ln_out.weight, b.ln_out.bias
        self.eps = b.ln_pair.eps
        self.b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)

    def forward(self, pair, mask=None):
        m2 = None
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(pair.dtype)
        return TriMulCuteFn.apply(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp,
                                  self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b,
                                  self.eps, self.b_lr, m2)
