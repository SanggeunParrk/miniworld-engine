"""Single-direction trimul TRAINING (fwd+bwd) — v6 split back, composed from
autograd-enabled kernel pieces so torch chains the backward automatically:

    x_n  = LayerNorm_in(pair)                       # torch F.layer_norm (autograd)
    L,R  = Front(x_n)                               # _FrontFn: cute fwd, torch-recompute bwd
    tri  = einsum("bdik,bdjk->bdij", L, R)          # torch (cuBLAS), autograd
    proj = layernorm_linear_fn(tri, ln_out, Wp)     # ① cute LayerNormLinear (own bwd)
    y    = GateElem(x_n, proj, Wg)                   # ② triton GateElem (own bwd)

So the backward is the composition of: GateElem bwd (triton), LayerNormLinear bwd
(cute), einsum bwd (cuBLAS), Front bwd (torch recompute — front_bwd_fused is the
next kernelization), LN_in bwd (torch). No bespoke fused-back backward.

B=1, bf16, square (d_hidden=d_pair=D). Outgoing; pass direction='in' for incoming.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.layernorm_linear.te_style import layernorm_linear_te_fn
from miniworld_engine.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_fused
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import GateElem


class _TriContract(torch.autograd.Function):
    """Triangle contraction with CONTIGUOUS-output backward (dtv1's trick). Plain
    torch.bmm autograd makes grad_right non-contiguous (a transpose), which the
    downstream cat/reshape then .contiguous()-copies — profiled at ~8.6ms/iter @L=1024.
    Choosing equivalent grad formulas whose outputs are already contiguous kills it.
    left/right: (D, L, L). out: (D, L, L)."""

    @staticmethod
    def forward(ctx, left, right, direction_flag):
        if direction_flag == 0:   # outgoing: O = L @ Rᵀ
            out = torch.bmm(left, right.transpose(1, 2))
        else:                     # incoming: O = Lᵀ @ R
            out = torch.bmm(left.transpose(1, 2), right)
        ctx.save_for_backward(left, right)
        ctx.direction_flag = direction_flag
        return out

    @staticmethod
    def backward(ctx, grad_out):
        left, right = ctx.saved_tensors
        if ctx.direction_flag == 0:
            grad_left = torch.bmm(grad_out, right)                  # G @ R   (contiguous)
            grad_right = torch.bmm(grad_out.transpose(1, 2), left)  # Gᵀ @ L  (contiguous)
        else:
            grad_left = torch.bmm(right, grad_out.transpose(1, 2))  # R @ Gᵀ
            grad_right = torch.bmm(left, grad_out)                  # L @ G
        return grad_left, grad_right, None


class _FrontFn(torch.autograd.Function):
    """Front gated GEMM: L=(x_n@WL)⊙σ(x_n@WLg), R=(x_n@WR)⊙σ(x_n@WRg) → bdll.
    Forward = cute (fast, also emits the pre-glu preact). Backward = front_bwd_fused
    (triton, channel-major: NO bdll→blld transpose, fused elementwise — the torch
    recompute path was 24ms/call dominated by permute-clones + elementwise muls)."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, b_lr):
        left, right, preact = trimul_inproj_cute_forward(
            x_n, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False,
            b_lr=b_lr, return_preact=True)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, preact)
        return left, right

    @staticmethod
    def backward(ctx, d_left_b, d_right_b):
        x_n, WL, WLg, WR, WRg, preact = ctx.saved_tensors
        B, L, _, D = x_n.shape
        # d_left_b/d_right_b are already bdll (B,D,L,L) — pass separately (no d_lr cat).
        dxn, dWL, dWLg, dWR, dWRg = front_bwd_fused(
            d_left_b.reshape(B, D, L, L), d_right_b.reshape(B, D, L, L),
            preact, x_n, WL, WLg, WR, WRg)
        return dxn, dWL, dWLg, dWR, dWRg, None


@torch.compiler.disable   # opaque to torch.compile: keep our cuBLAS GEMMs (Inductor would
# re-codegen them into slower mm) and avoid graph-breaks at every cute/triton custom Function.
# Surrounding model (residual/dropout/other layers) still compiles & fuses around this call.
def v6_forward(pair, WL, WLg, WR, WRg, Wg, Wp_nn, ln_in_w, ln_in_b,
               ln_out_w, ln_out_b, eps, b_lr, direction="out"):
    B, L, _, D = pair.shape
    M = B * L * L
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps)            # LN_in (repo triton, autograd)
    left, right = _FrontFn.apply(x_n, WL, WLg, WR, WRg, b_lr)      # cute front
    # contraction via EXPLICIT torch.bmm (not torch.einsum) — torch.bmm's autograd is
    # efficient bmm-both-ways; einsum's autograd was ~5x slower (the L³ bwd blowup).
    lf = left.reshape(D, L, L)
    rf = right.reshape(D, L, L)
    tri = _TriContract.apply(lf, rf, 0 if direction == "out" else 1)   # (D,L,L), contiguous-grad bwd
    # ① LN_out + @Wp: stride-transparent TE-style LayerNormLinear. Reads the m-major tri
    # view (M,D, strides (1,M)) COPY-FREE and returns dx in the same m-major stride, so its
    # .t() back to (D,M) is contiguous → tri-grad reshape is free (no transpose-clone). Fuses
    # LN_out + @Wp in one path (replaces layer_norm_transpose + separate GEMM).
    view = tri.reshape(D, M).t()                                  # (M, D) m-major view (no copy)
    proj = layernorm_linear_te_fn(view, ln_out_w, ln_out_b, Wp_nn, None, eps)   # (M, N)
    y = GateElem.apply(x_n.reshape(M, D), proj, Wg)               # ② (M,D)
    return y.reshape(B, L, L, D)


class V6TriMul(nn.Module):
    """Trainable single-direction trimul, v6 split back. Built from a base
    TriangleMultiplication's weights. bf16. Weights kept in x@W form (= .T)."""

    def __init__(self, base, direction="out"):
        super().__init__()
        b = base
        self.direction = direction
        self.WL = nn.Parameter(b.to_left.weight.t().contiguous())
        self.WLg = nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = nn.Parameter(b.to_gate.weight.t().contiguous())   # (D,D) x@W form
        self.Wp_nn = nn.Parameter(b.to_out.weight.detach().clone())  # (N,K) nn.Linear form
        self.ln_in_w = nn.Parameter(b.ln_pair.weight.detach().clone())
        self.ln_in_b = nn.Parameter(b.ln_pair.bias.detach().clone())
        self.ln_out_w = nn.Parameter(b.ln_out.weight.detach().clone())
        self.ln_out_b = nn.Parameter(b.ln_out.bias.detach().clone())
        self.eps = b.ln_pair.eps

    def forward(self, pair):
        b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)
        return v6_forward(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp_nn,
                          self.ln_in_w, self.ln_in_b, self.ln_out_w, self.ln_out_b,
                          self.eps, b_lr, self.direction)
