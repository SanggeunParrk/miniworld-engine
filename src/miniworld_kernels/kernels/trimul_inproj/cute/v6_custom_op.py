"""v6 single-dir trimul as a torch.library CUSTOM OP — opaque to torch.compile WITHOUT a
graph break (Dynamo keeps it as one node), so a compiled model fuses around it and can
CUDA-graph the whole step, while our cuBLAS/cute/triton kernels run untouched.

⚠️ MEASURED & REJECTED — kept as a worked reference, NOT used in production. It is correct
(cos 0.99996 on all grads) and achieves 0 dynamo graph breaks + compiles fwd+bwd, but it runs
~1.9x SLOWER than the `@torch.compiler.disable` autograd.Function path (`v6_training.py`) in
the SAME harness (L=512 fwd+bwd: 4.57/4.30 vs 2.42/2.41 ms @D128; 8.26/7.78 vs 4.22/4.22 @D256,
eager/compiled). Root cause is structural: a custom op can't stash intermediates in ctx, so the
fwd op must RETURN the 10 bwd-needed tensors (preact (4D,L,L), x_n, tri, …) as outputs →
functionalization materializes/clones them; plus the alias-avoiding `lr=cat` copy and the extra
v6_bwd dispatch. autograd.Function saves those in ctx for free. The graph-break-0 win is real
but far smaller than the save-as-output cost. → use `@compiler.disable` (v6_training.py).

Note: BOTH fwd and bwd must be custom ops (`trimul::v6` + `trimul::v6_bwd`) — otherwise compile
traces into the backward's triton/cute kernels and hits the FakeTensor data-pointer error.

B=1, bf16, square (d_pair=d_hidden=D). Outgoing or incoming.
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.layernorm_linear.te_style import _te_backward, _te_forward
from miniworld_kernels.kernels.trimul_inproj.autograd import _ln_bwd, _ln_fwd
from miniworld_kernels.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_kernels.kernels.trimul_inproj.triton.back_fused import front_bwd_fused
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import gate_elem_bwd, gate_elem_triton


@torch.library.custom_op("trimul::v6", mutates_args=())
def v6(pair: torch.Tensor, WL: torch.Tensor, WLg: torch.Tensor, WR: torch.Tensor,
       WRg: torch.Tensor, Wg: torch.Tensor, Wp: torch.Tensor, lin_w: torch.Tensor,
       lin_b: torch.Tensor, lout_w: torch.Tensor, lout_b: torch.Tensor, b_lr: torch.Tensor,
       eps: float, outgoing: bool) -> list[torch.Tensor]:
    """Returns [y, x_n, preact, lr, tri, te_xn, mean_out, rstd_out, gate, proj]; only y is
    used downstream — the rest are saved for backward (see module docstring)."""
    B, L, _, D = pair.shape
    M = B * L * L
    _, mean_in, rstd_in, _ = _ln_fwd(pair, lin_w, lin_b, eps)
    x_n = ((pair - mean_in[..., None]) * rstd_in[..., None]) * lin_w + lin_b   # (B,L,L,D)
    left, right, preact = trimul_inproj_cute_forward(
        x_n, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False,
        b_lr=b_lr, return_preact=True)
    lr = torch.cat([left.reshape(1, D, L, L), right.reshape(1, D, L, L)], dim=1)  # (1,2D,L,L) non-alias
    lf, rf = left.reshape(D, L, L), right.reshape(D, L, L)
    tri = torch.bmm(lf, rf.transpose(1, 2)) if outgoing else torch.bmm(lf.transpose(1, 2), rf)
    view = tri.reshape(D, M).t()                                              # (M,D) m-major
    proj, te_xn, mean_out, rstd_out = _te_forward(view, lout_w, lout_b, Wp, None, eps)
    y, gate = gate_elem_triton(x_n.reshape(M, D), proj, Wg, return_gate=True)
    return [y.reshape(B, L, L, D), x_n, preact, lr, tri, te_xn, mean_out, rstd_out, gate, proj]


@v6.register_fake
def _(pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w, lout_b, b_lr, eps, outgoing):
    B, L, _, D = pair.shape
    M = B * L * L
    e = pair.new_empty
    return [e(B, L, L, D), e(B, L, L, D), e(B, 4 * D, L, L), e(B, 2 * D, L, L), e(D, L, L),
            e(M, D), e(M), e(M), e(M, D), e(M, D)]


def _v6_setup(ctx, inputs, output):
    pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w, lout_b, b_lr, eps, outgoing = inputs
    _, x_n, preact, lr, tri, te_xn, mean_out, rstd_out, gate, proj = output
    ctx.save_for_backward(pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w,
                          x_n, preact, lr, tri, te_xn, mean_out, rstd_out, gate, proj)
    ctx.eps = eps
    ctx.outgoing = outgoing


@torch.library.custom_op("trimul::v6_bwd", mutates_args=())
def v6_bwd(gy: torch.Tensor, pair: torch.Tensor, WL: torch.Tensor, WLg: torch.Tensor,
           WR: torch.Tensor, WRg: torch.Tensor, Wg: torch.Tensor, Wp: torch.Tensor,
           lin_w: torch.Tensor, lin_b: torch.Tensor, lout_w: torch.Tensor, x_n: torch.Tensor,
           preact: torch.Tensor, lr: torch.Tensor, tri: torch.Tensor, te_xn: torch.Tensor,
           mean_out: torch.Tensor, rstd_out: torch.Tensor, gate: torch.Tensor,
           proj: torch.Tensor, eps: float, outgoing: bool) -> list[torch.Tensor]:
    """The v6 backward as a custom op (opaque to compile too — else compile traces into the
    triton/cute kernels and hits the FakeTensor data-pointer error).
    Returns [dx, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNi_w, dLNi_b, dLNo_w, dLNo_b]."""
    B, L, _, D = pair.shape
    M = B * L * L
    gy = gy.reshape(M, D)
    d_proj, dx_gate, dWg = gate_elem_bwd(gy, x_n.reshape(M, D), proj, gate, Wg)
    view = tri.reshape(D, M).t()
    d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
        d_proj, te_xn, view, mean_out, rstd_out, lout_w, Wp, has_bias=False)
    d_tri = d_view.t().reshape(D, L, L)
    lf, rf = lr[0, :D], lr[0, D:]
    if outgoing:
        d_left = torch.bmm(d_tri, rf)
        d_right = torch.bmm(d_tri.transpose(1, 2), lf)
    else:
        d_left = torch.bmm(rf, d_tri.transpose(1, 2))
        d_right = torch.bmm(lf, d_tri)
    dxn_front, dWL, dWLg, dWR, dWRg = front_bwd_fused(
        d_left.reshape(B, D, L, L), d_right.reshape(B, D, L, L), preact, x_n, WL, WLg, WR, WRg)
    dx_n = dxn_front + dx_gate.reshape(B, L, L, D)
    _, mean_in, rstd_in, xhat_in = _ln_fwd(pair, lin_w, lin_b, eps)
    dx, dLNi_w, dLNi_b = _ln_bwd(dx_n, xhat_in, rstd_in, lin_w)
    return [dx, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNi_w, dLNi_b, dLNo_w, dLNo_b]


@v6_bwd.register_fake
def _(gy, pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w, x_n, preact, lr, tri,
      te_xn, mean_out, rstd_out, gate, proj, eps, outgoing):
    return [torch.empty_like(pair), torch.empty_like(WL), torch.empty_like(WLg),
            torch.empty_like(WR), torch.empty_like(WRg), torch.empty_like(Wg),
            torch.empty_like(Wp), torch.empty_like(lin_w), torch.empty_like(lin_b),
            torch.empty_like(lout_w), torch.empty_like(lin_b)]   # dLNo_b ~ lin_b shape (D,)


def _v6_backward(ctx, grads):
    # list-output custom op → `grads` is the list of per-output grads; only y's matters.
    (pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w,
     x_n, preact, lr, tri, te_xn, mean_out, rstd_out, gate, proj) = ctx.saved_tensors
    g = v6_bwd(grads[0], pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w, x_n, preact,
               lr, tri, te_xn, mean_out, rstd_out, gate, proj, ctx.eps, ctx.outgoing)
    # order: pair,WL,WLg,WR,WRg,Wg,Wp,lin_w,lin_b,lout_w,lout_b,b_lr,eps,outgoing
    return (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], None, None, None)


v6.register_autograd(_v6_backward, setup_context=_v6_setup)


def v6_custom(pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w, lout_b, b_lr, eps, direction="out"):
    """Differentiable call returning y. Wraps the custom op (which returns y + saved)."""
    return v6(pair, WL, WLg, WR, WRg, Wg, Wp, lin_w, lin_b, lout_w, lout_b, b_lr, eps,
              direction == "out")[0]
