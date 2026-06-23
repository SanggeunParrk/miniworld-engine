"""B2 — compile-native trimul: cute front as a custom_op (opaque to Dynamo),
everything else (LN, bmm, gate, proj, and ALL their backwards) plain torch so
torch.compile + aot_autograd fuses & cudagraphs the whole fwd+bwd.

Only the fast cute front gated-GEMM is a graph break; its backward is registered
as plain torch (recompute pL/gL via one concat GEMM, then gated-GEMM bwd), so it
too gets compiled. This is the "compile everything consistently" path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand


# ── cute front as a custom op ────────────────────────────────────────────────
# Returns the ONE combined [B,2D,L,L] buffer (no clone, no aliasing); the module
# slices left=[:,:D]/right=[:,D:] as views (torch handles it under compile).
@torch.library.custom_op("trimul_inproj::front", mutates_args=())
def _front(x_n: Tensor, WL: Tensor, WLg: Tensor, WR: Tensor, WRg: Tensor,
           b_lr: Tensor) -> Tensor:
    from quack.gemm_interface import gemm_act

    from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
    _bdll_patch.apply()
    B, L, _, D = x_n.shape
    x_flat = x_n.reshape(B * L * L, D)
    lr = torch.empty(B, 2 * D, L, L, device=x_n.device, dtype=x_n.dtype)
    lr_view = lr.view(2 * D, L * L).T  # (M, 2D) strides (1, L*L)
    gemm_act(A=x_flat, B=b_lr, activation="glu", store_preact=False, postact_out=lr_view)
    return lr


@_front.register_fake
def _(x_n, WL, WLg, WR, WRg, b_lr):
    B, L, _, D = x_n.shape
    return x_n.new_empty(B, 2 * D, L, L)


def _front_setup(ctx, inputs, output):
    x_n, WL, WLg, WR, WRg, b_lr = inputs
    ctx.save_for_backward(x_n, WL, WLg, WR, WRg)


def _front_backward(ctx, d_lr):
    x_n, WL, WLg, WR, WRg = ctx.saved_tensors
    B, L, _, D = x_n.shape
    M = B * L * L
    xf = x_n.reshape(M, D)
    # recompute proj/gate logits for left+right in ONE GEMM: x_n @ [WL|WLg|WR|WRg]
    Wcat = torch.cat([WL, WLg, WR, WRg], dim=1)           # (D, 4D)
    pg = xf @ Wcat                                         # (M, 4D)
    pL, gLl, pR, gRl = pg[:, :D], pg[:, D:2 * D], pg[:, 2 * D:3 * D], pg[:, 3 * D:]
    gL, gR = torch.sigmoid(gLl), torch.sigmoid(gRl)

    dL = d_lr[:, :D].permute(0, 2, 3, 1).reshape(M, D)     # combined bdll -> (M,D)
    dR = d_lr[:, D:].permute(0, 2, 3, 1).reshape(M, D)
    d_pL = dL * gL
    d_gLlog = (dL * pL) * gL * (1 - gL)
    d_pR = dR * gR
    d_gRlog = (dR * pR) * gR * (1 - gR)

    # input-grad: [d_pL|d_gLlog|d_pR|d_gRlog] @ [WL|WLg|WR|WRg]^T  (one GEMM)
    DL = torch.cat([d_pL, d_gLlog, d_pR, d_gRlog], dim=1)  # (M, 4D)
    dx_n = (DL @ Wcat.t()).reshape(B, L, L, D)
    # weight-grad: x_n^T @ each
    dWL = xf.t() @ d_pL
    dWLg = xf.t() @ d_gLlog
    dWR = xf.t() @ d_pR
    dWRg = xf.t() @ d_gRlog
    return dx_n, dWL, dWLg, dWR, dWRg, None


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
        # b_lr prepacked once; registered as buffer so it moves with the module
        self.register_buffer("b_lr", prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg))

    def forward(self, pair, mask=None):
        B, L, _, D = pair.shape
        x_n = F.layer_norm(pair, (D,), self.ln_in_w, self.ln_in_b, self.eps)
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(x_n.dtype)
            x_n = x_n * m2
        lr = _front(x_n, self.WL, self.WLg, self.WR, self.WRg, self.b_lr)  # (B,2D,L,L)
        left_b, right_b = lr[:, :self.D], lr[:, self.D:]
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)       # (B,D,L,L)
        out_n = F.layer_norm(tri.permute(0, 2, 3, 1), (D,), self.ln_out_w, self.ln_out_b, self.eps)
        gate = torch.sigmoid(x_n @ self.Wg)
        return (out_n @ self.Wp) * gate
