"""B2 — compile-native trimul: cute front as a custom_op (opaque to Dynamo),
everything else (LN, bmm, gate, proj, and ALL their backwards) plain torch so
torch.compile + aot_autograd fuses & cudagraphs the whole fwd+bwd.

Only the fast cute front gated-GEMM is a graph break; its backward is registered
as plain torch (recompute pL/gL via one concat GEMM, then gated-GEMM bwd), so it
too gets compiled. This is the "compile everything consistently" path.
"""

from __future__ import annotations

import torch
from torch import Tensor
from miniworld_engine.kernels.layernorm.transpose import layer_norm_transpose

from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_engine.kernels.trimul_inproj.cute.launch import prepack_lr_operand
from miniworld_engine.kernels.trimul_inproj.triton.front import trimul_front_triton
from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_engine.kernels.fused_ln_mask.cute.fused_ln_mask import fused_ln_mask


def _ln(x, w, b, eps, layout):
    o = layer_norm_transpose(x, w, b, eps=eps, layout=layout)
    return o[0] if isinstance(o, tuple) else o


# ── cute front as a custom op ────────────────────────────────────────────────
# forward returns (lr=[B,2D,L,L] gated output, preact=[B,4D,L,L] pre-GLU logits),
# both channel-major (M-major views of bdll buffers). Saving preact lets the
# backward skip the recompute GEMM AND stay channel-major (no bdll->blld transpose).
# b_lr is built (differentiably) from WL/WLg/WR/WRg in the module, so weight grads
# flow through it via autograd — the op only returns d_x_n and d_b_lr.
@torch.library.custom_op("trimul_inproj::front", mutates_args=())
def _front(x_n: Tensor, b_lr: Tensor) -> tuple[Tensor, Tensor]:
    from quack.gemm_interface import gemm_act

    from miniworld_engine.kernels.trimul_inproj.cute import _bdll_patch
    _bdll_patch.apply()
    B, L, _, D = x_n.shape
    x_flat = x_n.reshape(B * L * L, D)
    lr = torch.empty(B, 2 * D, L, L, device=x_n.device, dtype=x_n.dtype)
    preact = torch.empty(B, 4 * D, L, L, device=x_n.device, dtype=x_n.dtype)
    lr_view = lr.view(2 * D, L * L).T        # (M, 2D)
    pre_view = preact.view(4 * D, L * L).T    # (M, 4D)  pre-GLU, interleaved [g|p]
    gemm_act(A=x_flat, B=b_lr, activation="glu", preact_out=pre_view,
             postact_out=lr_view, store_preact=True)
    return lr, preact


@_front.register_fake
def _(x_n, b_lr):
    B, L, _, D = x_n.shape
    return x_n.new_empty(B, 2 * D, L, L), x_n.new_empty(B, 4 * D, L, L)


def _front_setup(ctx, inputs, output):
    x_n, b_lr = inputs
    _, preact = output
    ctx.save_for_backward(x_n, b_lr, preact)


def _front_backward(ctx, d_lr, d_preact_unused):
    x_n, b_lr, preact = ctx.saved_tensors
    B, L, _, D = x_n.shape
    M = B * L * L
    D2 = 2 * D
    # channel-major slices (B,D,L,L). b_lr cols interleaved: even=gate, odd=proj,
    # first 2D = left, next 2D = right.
    gLlog, pLc = preact[:, 0:D2:2], preact[:, 1:D2:2]
    gRlog, pRc = preact[:, D2:4 * D:2], preact[:, D2 + 1:4 * D:2]
    gL, gR = torch.sigmoid(gLlog), torch.sigmoid(gRlog)
    dLc, dRc = d_lr[:, :D], d_lr[:, D:]                    # channel-major grads
    d_pL = dLc * gL
    d_gLlog = (dLc * pLc) * gL * (1 - gL)
    d_pR = dRc * gR
    d_gRlog = (dRc * pRc) * gR * (1 - gR)

    # d_preact in the SAME interleaved channel-major layout as b_lr's columns
    d_pre = torch.empty_like(preact)                       # (B,4D,L,L)
    d_pre[:, 0:D2:2] = d_gLlog
    d_pre[:, 1:D2:2] = d_pL
    d_pre[:, D2:4 * D:2] = d_gRlog
    d_pre[:, D2 + 1:4 * D:2] = d_pR

    d_pre2 = d_pre.reshape(4 * D, M)                       # (4D, M)  (B=1)
    dx_n = (b_lr @ d_pre2).t().reshape(B, L, L, D)         # b_lr(D,4D)@(4D,M)->(D,M)->blld
    d_b_lr = (d_pre2 @ x_n.reshape(M, D)).t()              # (4D,M)@(M,D)->(4D,D)->(D,4D)
    return dx_n, d_b_lr


_front.register_autograd(_front_backward, setup_context=_front_setup)


# ── back-half as a custom op: LN_out + @Wp + gate-mul fused (layernorm_linear) ──
# forward = one fused cute kernel (2x faster than cuequiv LN_out + matmul + mul at
# large L, since proj N=D=128 is the M2-fused win regime); backward = torch formulas.
@torch.library.custom_op("trimul_inproj::back", mutates_args=())
def _back(tri: Tensor, gate: Tensor, Wp_lin: Tensor, ln_w: Tensor, ln_b: Tensor,
          eps: float) -> Tensor:
    from miniworld_engine.kernels.trimul_inproj.cute import _bdll_patch
    _bdll_patch.apply()
    B, Dc, L, _ = tri.shape
    M = B * L * L
    tri_view = tri.reshape(Dc, M).t()                      # (M,D) strided view (no copy)
    y = layernorm_linear_cute_fused(tri_view, ln_w, ln_b, Wp_lin, None, eps=eps,
                                    gate=gate.reshape(M, Dc))
    return y.view(B, L, L, Dc)


@_back.register_fake
def _(tri, gate, Wp_lin, ln_w, ln_b, eps):
    B, Dc, L, _ = tri.shape
    return tri.new_empty(B, L, L, Dc)


def _back_setup(ctx, inputs, output):
    tri, gate, Wp_lin, ln_w, ln_b, eps = inputs
    ctx.save_for_backward(tri, gate, Wp_lin, ln_w, ln_b)
    ctx.eps = eps


def _back_backward(ctx, dy):
    # GEMMs in bf16 (cuBLAS), fp32 only for the LN-stat reductions. (mean/var over D.)
    tri, gate, Wp_lin, ln_w, ln_b = ctx.saved_tensors
    eps = ctx.eps
    B, Dc, L, _ = tri.shape
    M = B * L * L
    dt = tri.dtype
    trib = tri.permute(0, 2, 3, 1).reshape(M, Dc)          # bf16 (M,D)
    tf = trib.float()
    mean = tf.mean(-1, keepdim=True)
    rstd = torch.rsqrt(tf.var(-1, unbiased=False, keepdim=True) + eps)
    xhat = (tf - mean) * rstd                              # fp32 (M,D)
    out_n = (xhat * ln_w.float() + ln_b.float()).to(dt)    # bf16
    g = gate.reshape(M, Dc)
    dyr = dy.reshape(M, Dc)
    Wp_xw = Wp_lin.t()                                     # bf16 (D,D), x@W
    proj = out_n @ Wp_xw                                   # bf16
    d_proj = dyr * g                                       # bf16
    d_gate = dyr * proj                                    # bf16
    d_out_n = d_proj @ Wp_xw.t()                           # bf16
    dWp_lin = d_proj.t() @ out_n                           # bf16 (N,K)
    don = d_out_n.float()
    dxhat = don * ln_w.float()
    c1 = dxhat.mean(-1, keepdim=True)
    c2 = (dxhat * xhat).mean(-1, keepdim=True)
    d_trib = (rstd * (dxhat - c1 - xhat * c2)).to(dt)
    d_ln_w = (don * xhat).sum(0).to(ln_w.dtype)
    d_ln_b = don.sum(0).to(ln_b.dtype)
    d_tri = d_trib.reshape(B, L, L, Dc).permute(0, 3, 1, 2).contiguous()
    return (d_tri, d_gate.reshape(B, L, L, Dc), dWp_lin, d_ln_w, d_ln_b, None)


_back.register_autograd(_back_backward, setup_context=_back_setup)


# ── module: torch-native around the custom op ────────────────────────────────
class TriMulCompile(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        b = base
        self.WL = torch.nn.Parameter(b.to_left.weight.t().contiguous())
        self.WLg = torch.nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = torch.nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = torch.nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = torch.nn.Parameter(b.to_gate.weight.t().contiguous())
        self.Wp = torch.nn.Parameter(b.to_out.weight.t().contiguous())
        self.ln_in_w, self.ln_in_b = b.ln_pair.weight, b.ln_pair.bias
        self.ln_out_w, self.ln_out_b = b.ln_out.weight, b.ln_out.bias
        self.eps = b.ln_pair.eps
        self.D = self.WL.shape[0]
        # Prepack the sm_100 front B-operands ONCE (interleaved [WLᵀ|WRᵀ], [WLgᵀ|WRgᵀ])
        # so the hot path skips the per-call cat/transpose.
        from miniworld_engine.kernels.trimul_inproj.cute.front_sm100_fused import (
            prepack_lr_operand_sm100,
        )
        self._front_packed = prepack_lr_operand_sm100(self.WL, self.WLg, self.WR, self.WRg)

    def forward(self, pair, mask=None):
        B, L, _, D = pair.shape
        # B200 (sm_100): LN_in (+ optional pair-mask) fused in ONE triton kernel
        # (fused_ln_mask, ~0.09ms@L=1024, ~75% HBM peak) — replaces a separate
        # triton LN + a torch mask multiply. No cuequiv.
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(pair.dtype)  # (B,L,L)
            x_n = fused_ln_mask(pair, self.ln_in_w, self.ln_in_b, m2, self.eps)
        else:
            x_n = triton_layernorm(pair.reshape(B * L * L, D), self.ln_in_w, self.ln_in_b, self.eps).view(B, L, L, D)
        # front: custom sm_100 bdll-direct gated GEMM (M-major t2r + epilogue-fused
        # gate-mul, 2 quack tcgen05 GEMMs, no transpose) — beats the triton front
        # (0.448 vs 0.543ms@1024). left/right come out [B,D,L,L] bdll directly.
        from miniworld_engine.kernels.trimul_inproj.cute.front_sm100_fused import (
            trimul_front_sm100_fused,
        )
        left_b, right_b = trimul_front_sm100_fused(x_n, packed=self._front_packed)
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)       # (B,D,L,L)
        # back-half: existing triton kernel (LN_out + proj + gate-mul), no cuequiv.
        return trimul_back_triton(tri, x_n, self.Wp, self.Wg,
                                  self.ln_out_w, self.ln_out_b, self.eps)
