"""trimul_inproj — TRAINING forward + backward (autograd.Function).

Unlike ``inference.py`` (saves nothing, max fusion), the TRAINING forward must
persist the tensors the backward consumes. *What* it saves is a design choice
that trades off against the backward's algorithm and speed:

  - save more (preact / saved sigmoid / LN stats / left,right) -> backward skips
    recompute (faster bwd) at the cost of memory + a heavier forward write;
  - save less -> backward recomputes (cheaper memory, slower bwd).

dt-v1's choice: save ``sig_m = sigmoid * mask`` and reuse the output ``ab`` so the
backward needs no projection/preact recompute (the ``ab*(1-sig)`` trick).

Current state = B1 baseline: a correct (B0-verified, bit-exact) manual backward
with plain torch/cuBLAS GEMMs and a save-heavy forward. This is the reference the
optimized save-design + fused backward will be validated against. The fast
inference path is reachable via ``TriMulInproj.inference``.

Mask: applied to x_n (== AF's mask*projection at every valid position, since
proj(0)=0); the output gate then sees masked x_n — invisible whenever padding
gradients are zero (standard masked training).
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.trimul_inproj.autograd import _ln_bwd, _ln_fwd
from miniworld_kernels.kernels.trimul_inproj.cute.inference import trimul_inproj_inference
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)


class TriMulInprojFn(torch.autograd.Function):
    """Training forward (save-heavy) + manual backward. B0-verified formulas."""

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

        # back-half
        d_proj = dy * gate
        d_gate = dy * proj
        d_glog = d_gate * gate * (1 - gate)
        d_out_n = d_proj @ Wp.t()
        dWp = flat(out_n).t() @ flat(d_proj)
        dx_n = d_glog @ Wg.t()
        dWg = flat(x_n).t() @ flat(d_glog)
        d_tri_lld, dWln_out, dBln_out = _ln_bwd(d_out_n, xhat_out, rstd_out, ln_out_w)
        d_tri = d_tri_lld.permute(0, 3, 1, 2).contiguous()           # (B,D,L,L)

        # bmm bwd (bdll)
        d_left_b = torch.einsum("bdij,bdjk->bdik", d_tri, right_b)
        d_right_b = torch.einsum("bdij,bdik->bdjk", d_tri, left_b)
        d_left = d_left_b.permute(0, 2, 3, 1).contiguous()
        d_right = d_right_b.permute(0, 2, 3, 1).contiguous()

        # front gated-GEMM bwd — recompute pL,gL,pR,gR from x_n
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

        # LN_in bwd
        dx, dWln_in, dBln_in = _ln_bwd(dx_n, xhat_in, rstd_in, ln_in_w)
        return (dx, dWL, dWLg, dWR, dWRg, dWg, dWp,
                dWln_in, dBln_in, dWln_out, dBln_out, None, None, None)


class TriMulInproj(torch.nn.Module):
    """Trainable cute trimul (outgoing). Weights kept in x@W form (= nn.Linear .T).
    ``forward`` is the training path (autograd); ``inference`` is the fwd-only
    max-fusion path (no saved tensors). bf16."""

    def __init__(self, base):
        super().__init__()
        b = base
        self.WL = torch.nn.Parameter(b.to_left.weight.t().contiguous())
        self.WLg = torch.nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = torch.nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = torch.nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = torch.nn.Parameter(b.to_gate.weight.t().contiguous())
        self.Wp = torch.nn.Parameter(b.to_out.weight.t().contiguous())
        # own ln weights (clone, don't alias base's params — else .to(dtype) mutates base)
        self.ln_in_w = torch.nn.Parameter(b.ln_pair.weight.detach().clone())
        self.ln_in_b = torch.nn.Parameter(b.ln_pair.bias.detach().clone())
        self.ln_out_w = torch.nn.Parameter(b.ln_out.weight.detach().clone())
        self.ln_out_b = torch.nn.Parameter(b.ln_out.bias.detach().clone())
        self.eps = b.ln_pair.eps

    def _b_lr(self):
        # built each call so weight grads flow through it via autograd (training)
        return prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)

    @staticmethod
    def _mask2d(mask, dtype):
        if mask is None:
            return None
        return (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(dtype)

    def forward(self, pair, mask=None):
        m2 = self._mask2d(mask, pair.dtype)
        return TriMulInprojFn.apply(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg,
                                    self.Wp, self.ln_in_w, self.ln_in_b, self.ln_out_w,
                                    self.ln_out_b, self.eps, self._b_lr(), m2)

    @torch.no_grad()
    def inference(self, pair, mask=None):
        rmask = None
        if mask is not None:
            rmask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).reshape(-1).to(pair.dtype)
        return trimul_inproj_inference(
            pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp,
            self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b, self.eps,
            self._b_lr(), rmask)
