"""B200 BIDIRECTIONAL trimul TRAINING (fwd+bwd) — built on the v9 sm100 kernels.

Faithful bidirectional analog of ``training_b200.TriMulB200Fn`` (single-direction v9):
a save-heavy sm100 forward + manual backward with NO forward recompute, extended to
the repo's FUSED bidirectional formulation (2h left/right, outgoing on [:h], incoming
on [h:], SHARED back-half over the 2h concatenation).

Reused v9 sm100 kernels:
  - front: sm100 tcgen05 gated-GLU collective (trimul_front_sm100_fused), called per
    direction (outgoing/incoming), each square (d, h)=(128,128).
  - front bwd: FUSED sm100 front_gatebwd_sm100 (per direction).
  - LN fwd: fused triton _ln_fwd_fused (fp32 stats).
  - LN_out+proj bwd: FUSED dgrad_lnbwd_sm100 (LN-dim K=2h=256) when it compiles/validates,
    else safe fallback d_proj@Wp.T + arch-agnostic optimized LN bwd (_ln_opt_bwd).
  - LN_in bwd: arch-agnostic optimized LN bwd (_ln_opt_bwd).
  - LLD<->DLL transposes: coalesced tiled triton (_fast_T).
B=1, bf16 in / fp32 acc (fp32 LN stats) / bf16 out; d_hidden == d_pair == 128.
"""

from __future__ import annotations

import os

import torch

from miniworld_kernels.kernels.layernorm.compile_native import _dispatch_bwd as _ln_opt_bwd
from miniworld_kernels.kernels.layernorm_linear.cute.dgrad_lnbwd_sm100 import dgrad_lnbwd_sm100
from miniworld_kernels.kernels.trimul_inproj.cute.front_sm100_fused import (
    prepack_lr_operand_sm100,
    trimul_front_sm100_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute.gatebwd_sm100 import front_gatebwd_sm100
from miniworld_kernels.kernels.trimul_inproj.cute.training_b200 import (
    _add3, _colsum, _fast_T, _glu_bwd, _ln_fwd_fused,
)

# NOTE: the fused dgrad_lnbwd (LN_out+proj bwd folded into one tcgen05 epilogue) is
# UNSUPPORTED at the bidir LN dim K=2h=256 — its epilogue tile (tile_n must cover K)
# fails to compile at 256 (quack make_smem_layout_a -1 shape). The single-dir v9 path
# uses it at K=128. Default OFF for bidir: safe fallback = cuBLAS d_proj@Wp.T +
# arch-agnostic optimized LN bwd (_ln_opt_bwd). Set =1 to attempt the fused path.
_USE_FUSED_DGRAD = os.environ.get("MINIWORLD_BIDIR_DGRAD_FUSED", "0") != "0"


class BidirTriMulB200Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                ln_out_w, ln_out_b, eps, packed_o, packed_i, h, mask2d):
        B, L, _, D = pair.shape
        M = B * L * L
        H = 2 * h

        x_n, mean_in, rstd_in, _ = _ln_fwd_fused(pair, ln_in_w, ln_in_b, eps)
        if mask2d is not None:
            x_n = x_n * mask2d

        left_o, right_o = trimul_front_sm100_fused(x_n, packed=packed_o)   # (B,h,L,L)
        left_i, right_i = trimul_front_sm100_fused(x_n, packed=packed_i)
        tri_o = torch.einsum("bdik,bdjk->bdij", left_o, right_o)           # outgoing
        tri_i = torch.einsum("bdki,bdkj->bdij", left_i, right_i)           # incoming
        tri = torch.cat([tri_o, tri_i], dim=1)                            # (B, 2h, L, L)
        tri_lld = _fast_T(tri.reshape(H, M)).view(B, L, L, H)

        out_n, mean_out, rstd_out, xhat_out = _ln_fwd_fused(tri_lld, ln_out_w, ln_out_b, eps)
        xnf = x_n.reshape(M, D)
        gate = torch.sigmoid(xnf @ Wg)
        proj = out_n.reshape(M, H) @ Wp
        y = proj * gate

        ctx.save_for_backward(x_n, pair, mean_in, rstd_in, WL, WLg, WR, WRg, Wg, Wp,
                              left_o, right_o, left_i, right_i, xhat_out, rstd_out,
                              ln_out_w, out_n, gate, proj, tri_lld, mean_out)
        ctx.ln_in_w = ln_in_w
        ctx.has_mask = mask2d is not None
        ctx.mask2d = mask2d
        ctx.shape = (B, L, D, h)
        return y.view(B, L, L, D)

    @staticmethod
    def backward(ctx, dy):
        (x_n, pair, mean_in, rstd_in, WL, WLg, WR, WRg, Wg, Wp,
         left_o, right_o, left_i, right_i, xhat_out, rstd_out,
         ln_out_w, out_n, gate, proj, tri_lld, mean_out) = ctx.saved_tensors
        ln_in_w = ctx.ln_in_w
        B, L, D, h = ctx.shape
        M = B * L * L
        H = 2 * h
        dy = dy.reshape(M, D)
        xnf = x_n.reshape(M, D)

        # -- (1) gate/GLU bwd (no recompute) --
        d_proj, d_glog = _glu_bwd(dy, gate, proj)          # (M, D) each
        dWp = out_n.reshape(M, H).t() @ d_proj             # (H, D)
        dx_n = d_glog @ Wg.t()                             # (M, D)
        dWg = xnf.t() @ d_glog                             # (D, D)

        # -- (2) LN_out + @Wp bwd -> d_tri_lld (M, H) --
        if _USE_FUSED_DGRAD:
            Wpt = Wp.t().contiguous()                      # (D, H)
            xho = xhat_out.reshape(M, H)
            d_tri_lld = dgrad_lnbwd_sm100(
                d_proj, Wpt, xho, ln_out_w, rstd_out.reshape(-1)).view(M, H)
            T = d_proj.t() @ xho                            # (D, H)
            db_proj = _colsum(d_proj)                       # (D,)
            dBln_out = db_proj @ Wpt                        # (H,)
            dWln_out = (Wpt * T).sum(0)                     # (H,)
        else:
            d_out_n = d_proj @ Wp.t()                       # (M, H)
            d_tri_lld, dWln_out, dBln_out = _ln_opt_bwd(
                d_out_n.view(B, L, L, H), tri_lld, ln_out_w, mean_out, rstd_out)
            d_tri_lld = d_tri_lld.reshape(M, H)

        d_tri = _fast_T(d_tri_lld).view(B, H, L, L)         # (B, 2h, L, L)
        d_o, d_i = d_tri[:, :h], d_tri[:, h:]

        # -- (3) contraction bwd (outgoing / incoming) --
        d_left_o = torch.einsum("bdij,bdjk->bdik", d_o, right_o)
        d_right_o = torch.einsum("bdij,bdik->bdjk", d_o, left_o)
        d_left_i = torch.einsum("bdij,bdkj->bdki", d_i, right_i)
        d_right_i = torch.einsum("bdij,bdki->bdkj", d_i, left_i)
        d_lo = _fast_T(d_left_o.reshape(h, M))              # (M, h)
        d_ro = _fast_T(d_right_o.reshape(h, M))
        d_li = _fast_T(d_left_i.reshape(h, M))
        d_ri = _fast_T(d_right_i.reshape(h, M))

        # -- (4) front gated-GEMM bwd (fused sm100, per direction) --
        dxn_Lo, dWL_o, dWLg_o = front_gatebwd_sm100(xnf, d_lo, WL[:, :h], WLg[:, :h])
        dxn_Ro, dWR_o, dWRg_o = front_gatebwd_sm100(xnf, d_ro, WR[:, :h], WRg[:, :h])
        dxn_Li, dWL_i, dWLg_i = front_gatebwd_sm100(xnf, d_li, WL[:, h:], WLg[:, h:])
        dxn_Ri, dWR_i, dWRg_i = front_gatebwd_sm100(xnf, d_ri, WR[:, h:], WRg[:, h:])
        dWL = torch.cat([dWL_o, dWL_i], dim=1)             # (D, 2h)
        dWLg = torch.cat([dWLg_o, dWLg_i], dim=1)
        dWR = torch.cat([dWR_o, dWR_i], dim=1)
        dWRg = torch.cat([dWRg_o, dWRg_i], dim=1)

        dx_n = _add3(dx_n, dxn_Lo, dxn_Ro)
        dx_n = _add3(dx_n, dxn_Li, dxn_Ri)
        if ctx.has_mask:
            dx_n = dx_n * ctx.mask2d.reshape(M, D)

        # -- (5) LN_in bwd (arch-agnostic optimized LN bwd) --
        dx, dWln_in, dBln_in = _ln_opt_bwd(
            dx_n.view(B, L, L, D), pair, ln_in_w, mean_in, rstd_in)

        return (dx, dWL, dWLg, dWR, dWRg, dWg, dWp,
                dWln_in, dBln_in, dWln_out, dBln_out, None, None, None, None, None)


class BidirTriMulB200Train(torch.nn.Module):
    """Trainable bidirectional trimul (fused outgoing+incoming) on the v9 sm100 kernels.
    Built from a base BidirectionalTriangleMultiplication's weights. bf16, B=1, d=h=128."""

    def __init__(self, base):
        super().__init__()
        b = base
        self.h = b.d_hidden
        self.WL = torch.nn.Parameter(b.to_left.weight.t().contiguous())    # (d, 2h)
        self.WLg = torch.nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = torch.nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = torch.nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = torch.nn.Parameter(b.to_gate.weight.t().contiguous())    # (d, d)
        self.Wp = torch.nn.Parameter(b.to_out.weight.t().contiguous())     # (2h, d)
        self.ln_in_w = torch.nn.Parameter(b.ln_pair.weight.detach().clone())
        self.ln_in_b = torch.nn.Parameter(b.ln_pair.bias.detach().clone())
        self.ln_out_w = torch.nn.Parameter(b.ln_out.weight.detach().clone())   # (2h,)
        self.ln_out_b = torch.nn.Parameter(b.ln_out.bias.detach().clone())
        self.eps = b.ln_pair.eps

    def forward(self, pair, mask=None):
        h = self.h
        packed_o = prepack_lr_operand_sm100(
            self.WL[:, :h].detach(), self.WLg[:, :h].detach(),
            self.WR[:, :h].detach(), self.WRg[:, :h].detach())
        packed_i = prepack_lr_operand_sm100(
            self.WL[:, h:].detach(), self.WLg[:, h:].detach(),
            self.WR[:, h:].detach(), self.WRg[:, h:].detach())
        m2 = None
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(pair.dtype)
        return BidirTriMulB200Fn.apply(
            pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp,
            self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b,
            self.eps, packed_o, packed_i, h, m2)
