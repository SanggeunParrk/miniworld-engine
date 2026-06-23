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
from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose

from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand


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

    from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
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

    def forward(self, pair, mask=None):
        B, L, _, D = pair.shape
        # fused cuequiv LN (differentiable, fast bwd); LN_out fuses the bdij->bijd transpose
        x_n = _ln(pair, self.ln_in_w, self.ln_in_b, self.eps, "bijd->bijd")
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(x_n.dtype)
            x_n = x_n * m2
        # b_lr built here (differentiable) so weight grads flow via autograd
        b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)
        lr, _preact = _front(x_n, b_lr)  # (B,2D,L,L), (B,4D,L,L)
        left_b, right_b = lr[:, :self.D], lr[:, self.D:]
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)       # (B,D,L,L)
        out_n = _ln(tri, self.ln_out_w, self.ln_out_b, self.eps, "bdij->bijd")  # (B,L,L,D)
        gate = torch.sigmoid(x_n @ self.Wg)
        return (out_n @ self.Wp) * gate
