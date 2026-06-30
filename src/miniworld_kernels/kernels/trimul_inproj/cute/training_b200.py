"""B200 TRAINING (fwd+bwd) path for trimul (outgoing).

A torch.autograd.Function (``TriMulB200Fn``) whose FORWARD calls the exact B200
fast inference kernels used by ``compile_native.TriMulCompile.forward``:

    x_n   = fused_ln_mask(pair, ln_in_w, ln_in_b, m2, eps)   # LN_in (+ pair mask)
            (or triton_layernorm when no mask)
    left_b, right_b = trimul_front_sm100_fused(x_n, packed)  # bdll gated GLU GEMM
    tri   = einsum("bdik,bdjk->bdij", left_b, right_b)        # (B, D, L, L)
    y     = trimul_back_triton(tri, x_n, Wp, Wg, ln_out_w, ln_out_b, eps)

and whose BACKWARD implements the manual gradient using the B0/B1-verified
formulas (``autograd.py`` / ``autograd_cute.py``).  The front pL/gL/pR/gR are
NOT exposed by the fused GLU kernel, so they are recomputed from x_n in the
backward (one concat GEMM each, as in ``TriMulCuteFn.gated_bwd``).  Backward
GEMMs are plain torch/cuBLAS (correctness over speed).

CRITICAL forward-consistency notes (must match the inference forward exactly):
  * The mask is applied multiplicatively to x_n AFTER LN_in.  The masked x_n is
    what feeds the front AND the back-half gate.  So d(x_n) accumulates from the
    front, the gate, AND the back-half's d_tri path; the mask multiply happens
    on the SUMMED dx_n before LN_in backward (chain rule through `x_n*mask`).
  * LN_in stats (mean/rstd/xhat) are recomputed in fp32 in the backward from the
    saved `pair` input, matching `_ln_fwd` in autograd.py.

This is a standalone module/Function; it edits no shared files.  B == 1, square
L, D == 128, bf16.
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.trimul_inproj.autograd import _ln_bwd, _ln_fwd
from miniworld_kernels.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_kernels.kernels.trimul_inproj.cute.front_sm100_fused import (
    prepack_lr_operand_sm100,
    trimul_front_sm100_fused,
)
from miniworld_kernels.kernels.fused_ln_mask.cute.fused_ln_mask import fused_ln_mask
from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm


class TriMulB200Fn(torch.autograd.Function):
    """Fast B200 fwd kernels + manual backward (cuBLAS GEMMs)."""

    @staticmethod
    def forward(ctx, pair, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                ln_out_w, ln_out_b, eps, packed, mask2d):
        # mask2d: (B, L, L, 1) pair mask broadcast over D, or None.
        B, L, _, D = pair.shape

        # ── LN_in (+ optional mask), exactly as the inference forward ──────────
        if mask2d is not None:
            m2 = mask2d.squeeze(-1).contiguous()  # (B, L, L) for the fused kernel
            x_n = fused_ln_mask(pair, ln_in_w, ln_in_b, m2, eps)
        else:
            x_n = triton_layernorm(
                pair.reshape(B * L * L, D), ln_in_w, ln_in_b, eps
            ).view(B, L, L, D)

        # ── front: fused sm100 gated GLU GEMM -> bdll left/right ───────────────
        left_b, right_b = trimul_front_sm100_fused(x_n, packed=packed)

        # ── bmm (channel-wise outer over k) ────────────────────────────────────
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)  # (B, D, L, L)

        # ── back-half: fused triton LN_out + proj + gate-mul ───────────────────
        y = trimul_back_triton(tri, x_n, Wp, Wg, ln_out_w, ln_out_b, eps)

        # Save only what the manual backward needs.  pL/gL/pR/gR are recomputed
        # from x_n (front kernel does not expose them), matching TriMulCuteFn.
        ctx.save_for_backward(pair, x_n, WL, WLg, WR, WRg, Wg, Wp,
                              left_b, right_b, ln_in_w, ln_in_b, ln_out_w, ln_out_b)
        ctx.eps = eps
        ctx.has_mask = mask2d is not None
        ctx.mask2d = mask2d
        ctx.shape = (B, L, D)
        return y

    @staticmethod
    def backward(ctx, dy):
        (pair, x_n, WL, WLg, WR, WRg, Wg, Wp,
         left_b, right_b, ln_in_w, ln_in_b, ln_out_w, ln_out_b) = ctx.saved_tensors
        eps = ctx.eps
        B, L, D = ctx.shape
        M = B * L * L

        def flat(t):
            return t.reshape(M, D)

        # Recompute the back-half forward intermediates (out_n, gate, proj) that
        # the fused triton kernel did not save: LN_out(tri) then gate/proj.
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)        # (B, D, L, L)
        tri_lld = tri.permute(0, 2, 3, 1).contiguous()               # (B, L, L, D)
        out_n, _, rstd_out, xhat_out = _ln_fwd(tri_lld, ln_out_w, ln_out_b, eps)
        gate = torch.sigmoid(x_n @ Wg)
        proj = out_n @ Wp

        # ── ① back-half ────────────────────────────────────────────────────────
        d_proj = dy * gate
        d_gate = dy * proj
        d_glog = d_gate * gate * (1 - gate)
        d_out_n = d_proj @ Wp.t()
        dWp = flat(out_n).t() @ flat(d_proj)
        dx_n = d_glog @ Wg.t()                                       # gate -> x_n
        dWg = flat(x_n).t() @ flat(d_glog)
        d_tri_lld, dWln_out, dBln_out = _ln_bwd(d_out_n, xhat_out, rstd_out, ln_out_w)
        d_tri = d_tri_lld.permute(0, 3, 1, 2).contiguous()          # (B, D, L, L)

        # ── ② bmm bwd (bdll) ─────────────────────────────────────────────────
        d_left_b = torch.einsum("bdij,bdjk->bdik", d_tri, right_b)
        d_right_b = torch.einsum("bdij,bdik->bdjk", d_tri, left_b)
        d_left = d_left_b.permute(0, 2, 3, 1).contiguous()          # (B, L, L, D)
        d_right = d_right_b.permute(0, 2, 3, 1).contiguous()

        # ── ③ front gated-GEMM bwd — recompute pL,gL,pR,gR from x_n ───────────
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

        # ── mask: x_n = LN(pair) * mask, so dx_n flows back through the mask ───
        if ctx.has_mask:
            dx_n = dx_n * ctx.mask2d

        # ── ④ LN_in bwd (recompute stats in fp32 from pair) ──────────────────
        _, _, rstd_in, xhat_in = _ln_fwd(pair, ln_in_w, ln_in_b, eps)
        dx, dWln_in, dBln_in = _ln_bwd(dx_n, xhat_in, rstd_in, ln_in_w)

        # grads match forward arg order:
        # pair, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b, ln_out_w, ln_out_b,
        # eps, packed, mask2d
        return (dx, dWL, dWLg, dWR, dWRg, dWg, dWp,
                dWln_in, dBln_in, dWln_out, dBln_out, None, None, None)


class TriMulB200Train(torch.nn.Module):
    """Trainable B200 trimul (outgoing).  Forward uses the fast B200 kernels;
    backward is the manual gradient.  Weights stored in x@W form (= weight.T).
    bf16, B == 1, D == 128."""

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
        # The fused GLU forward needs the interleaved [WLg|WL|WRg|WR] B-operand.
        # During training the weights change each step, so rebuild it from the
        # live (detached) weights every call.  Grads w.r.t. WL/WLg/WR/WRg flow
        # through the manual backward (gated_bwd), NOT through this buffer, so a
        # detached rebuild is correct.
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
