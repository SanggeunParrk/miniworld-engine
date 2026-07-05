"""B200 TRAINING (fwd+bwd) path for trimul (outgoing).

FAITHFUL PORT of the H100 training design (``cute/training.py`` = ``TriMulInprojFn``):
a *save-heavy* forward + a manual backward that consumes the saved tensors with
NO recompute (mirrors ``training.py`` stage-for-stage).  Only the GEMM backend of
the FORWARD front differs: the fast sm100 tcgen05 gated-GLU collective produces
``left_b/right_b`` (bdll) instead of quack/WGMMA.

vs the previous B200 path (v0, git eb41ca4): that one used the max-fusion inference
forward (fused triton back) which saved nothing, so its backward had to recompute
the WHOLE back-half (tri einsum + LN_out fwd + gate + proj) AND the LN_in stats.
Profiling showed that recompute + the stat-recompute were ~8 ms of the ~20 ms bwd
at L=1024 — pure waste that H100 ``training.py`` avoids by SAVING.  This module
restores the H100 save-heavy contract:

  saved: x_n, xhat_in, rstd_in, left_b, right_b, xhat_out, rstd_out, out_n, gate,
         proj  (+ weights)  ->  backward does ZERO forward recompute.

Backward GEMMs are plain torch/cuBLAS, EXACTLY as H100 ``training.py`` (its docstring:
"a correct manual backward with plain torch/cuBLAS GEMMs and a save-heavy forward").
The front pL/gL/pR/gR are recomputed from x_n in the backward (one concat GEMM each),
identical to ``training.py.gated_bwd``.

Mask: applied multiplicatively to x_n AFTER LN_in (== training.py).  dx_n accumulates
from front + gate + back-half, then the mask multiply, then LN_in backward.
LN stats are fp32 (``_ln_fwd``/``_ln_bwd``).  B == 1, square L, D == 128, bf16.
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.trimul_inproj.autograd import _ln_bwd, _ln_fwd
from miniworld_kernels.kernels.trimul_inproj.cute.front_sm100_fused import (
    prepack_lr_operand_sm100,
    trimul_front_sm100_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute.gatebwd_sm100 import front_gatebwd_sm100


class TriMulB200Fn(torch.autograd.Function):
    """Save-heavy sm100 forward + manual backward (torch/cuBLAS), == H100 training.py."""

    @staticmethod
    def forward(ctx, pair, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                ln_out_w, ln_out_b, eps, packed, mask2d):
        B, L, _, D = pair.shape

        # -- LN_in (fp32 stats saved, faithful to training.py _ln_fwd) --
        x_n, _, rstd_in, xhat_in = _ln_fwd(pair, ln_in_w, ln_in_b, eps)
        if mask2d is not None:
            x_n = x_n * mask2d

        # -- front: FAST sm100 tcgen05 gated-GLU GEMM -> bdll left/right (kept) --
        left_b, right_b = trimul_front_sm100_fused(x_n, packed=packed)

        # -- bmm (channel-wise outer over k) --
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)  # (B, D, L, L)
        tri_lld = tri.permute(0, 2, 3, 1).contiguous()          # (B, L, L, D)

        # -- back-half (computed + SAVED, == training.py) --
        out_n, _, rstd_out, xhat_out = _ln_fwd(tri_lld, ln_out_w, ln_out_b, eps)
        gate = torch.sigmoid(x_n @ Wg)
        proj = out_n @ Wp
        y = proj * gate

        ctx.save_for_backward(x_n, xhat_in, rstd_in, WL, WLg, WR, WRg, Wg, Wp,
                              left_b, right_b, xhat_out, rstd_out, ln_out_w,
                              out_n, gate, proj)
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

        # -- (1) back-half (NO recompute; uses saved out_n/gate/proj/stats) --
        d_proj = dy * gate
        d_gate = dy * proj
        d_glog = d_gate * gate * (1 - gate)
        d_out_n = d_proj @ Wp.t()
        dWp = flat(out_n).t() @ flat(d_proj)
        dx_n = d_glog @ Wg.t()
        dWg = flat(x_n).t() @ flat(d_glog)
        d_tri_lld, dWln_out, dBln_out = _ln_bwd(d_out_n, xhat_out, rstd_out, ln_out_w)
        d_tri = d_tri_lld.permute(0, 3, 1, 2).contiguous()      # (B, D, L, L)

        # -- (2) bmm bwd (bdll) --
        d_left_b = torch.einsum("bdij,bdjk->bdik", d_tri, right_b)
        d_right_b = torch.einsum("bdij,bdik->bdjk", d_tri, left_b)
        d_left = d_left_b.permute(0, 2, 3, 1).contiguous()      # (B, L, L, D)
        d_right = d_right_b.permute(0, 2, 3, 1).contiguous()

        # -- (3) front gated-GEMM bwd — FUSED sm100 port of H100 backward_gatebwd:
        #    one dual-accumulator GEMM recomputes pL/gL from x_n and applies the GLU
        #    gate-backward in the tcgen05 epilogue (emits [d_glogit|d_p] interleaved),
        #    then single 2N d_xn / dW GEMMs.  == gated_bwd math, ~1.8x faster. --
        xnf = flat(x_n)
        dxn_L, dWL, dWLg = front_gatebwd_sm100(xnf, flat(d_left), WL, WLg)
        dxn_R, dWR, dWRg = front_gatebwd_sm100(xnf, flat(d_right), WR, WRg)
        dx_n = dx_n + dxn_L.view_as(dx_n) + dxn_R.view_as(dx_n)

        if ctx.has_mask:
            dx_n = dx_n * ctx.mask2d

        # -- (4) LN_in bwd (saved stats, NO recompute) --
        dx, dWln_in, dBln_in = _ln_bwd(dx_n, xhat_in, rstd_in, ln_in_w)

        return (dx, dWL, dWLg, dWR, dWRg, dWg, dWp,
                dWln_in, dBln_in, dWln_out, dBln_out, None, None, None)


class TriMulB200Train(torch.nn.Module):
    """Trainable B200 trimul (outgoing).  Save-heavy sm100 forward + manual backward
    (== H100 training.py).  Weights stored in x@W form (= weight.T).  bf16, B == 1,
    D == 128."""

    def __init__(self, base):
        super().__init__()
        b = base
        self.WL = torch.nn.Parameter(b.to_left.weight.t().contiguous())
        self.WLg = torch.nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = torch.nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = torch.nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = torch.nn.Parameter(b.to_gate.weight.t().contiguous())
        self.Wp = torch.nn.Parameter(b.to_out.weight.t().contiguous())
        self.ln_in_w = b.ln_pair.weight
        self.ln_in_b = b.ln_pair.bias
        self.ln_out_w = b.ln_out.weight
        self.ln_out_b = b.ln_out.bias
        self.eps = b.ln_pair.eps
        self.D = self.WL.shape[0]

    def _packed(self):
        return prepack_lr_operand_sm100(
            self.WL.detach(), self.WLg.detach(), self.WR.detach(), self.WRg.detach()
        )

    def forward(self, pair, mask=None):
        m2 = None
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(pair.dtype)
        return TriMulB200Fn.apply(
            pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp,
            self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b,
            self.eps, self._packed(), m2,
        )
