"""B0 — backward correctness scaffold for trimul (outgoing).

A single ``autograd.Function`` whose forward mirrors the trimul math (in torch,
incl. the bdll layout flip) and whose backward implements the gradient by hand
(stages ①-④ below). This is the *oracle*: its manual backward is checked against
torch autograd of the same forward. Once it is bit-correct, individual stages get
swapped for the cute/triton kernels while this stays as the reference.

Forward (outgoing):
    x_n  = LN_in(x)
    left = (x_n@WL) ⊙ σ(x_n@WLg) ;  right = (x_n@WR) ⊙ σ(x_n@WRg)   (→ bdll)
    tri  = einsum("bdik,bdjk->bdij", left, right)
    out_n= LN_out(tri)                                              (LN over D)
    gate = σ(x_n@Wg) ;  y = (out_n@Wp) ⊙ gate

Weights are in x@W form (= nn.Linear.weight.T): WL,WLg,WR,WRg,Wg,Wp all (D,D).
B=1 oriented but B is carried symbolically. dtype: compute in fp32 for the
oracle; the kernel path will be bf16.
"""

from __future__ import annotations

import torch


def _ln_fwd(x, w, b, eps):
    """LayerNorm over the last dim. Returns (y, mean, rstd, xhat)."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    xhat = (x - mean) * rstd
    y = xhat * w + b
    return y, mean.squeeze(-1), rstd.squeeze(-1), xhat


def _ln_bwd(dy, xhat, rstd, w):
    """Backward of LN over last dim D. dy,(xhat):(...,D), rstd:(...). w:(D,)
    Returns dx:(...,D), dw:(D,), db:(D,)."""
    D = dy.shape[-1]
    dw = (dy * xhat).reshape(-1, D).sum(0)
    db = dy.reshape(-1, D).sum(0)
    dxhat = dy * w
    rstd_ = rstd.unsqueeze(-1)
    mean1 = dxhat.mean(dim=-1, keepdim=True)
    mean2 = (dxhat * xhat).mean(dim=-1, keepdim=True)
    dx = rstd_ * (dxhat - mean1 - xhat * mean2)
    return dx, dw, db


class TriMulManualBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                ln_out_w, ln_out_b, eps):
        B, L, _, D = x.shape
        x_n, _mean_in, rstd_in, xhat_in = _ln_fwd(x, ln_in_w, ln_in_b, eps)

        pL = x_n @ WL
        gL = torch.sigmoid(x_n @ WLg)
        left = pL * gL                       # (B,L,L,D)
        pR = x_n @ WR
        gR = torch.sigmoid(x_n @ WRg)
        right = pR * gR

        left_b = left.permute(0, 3, 1, 2)    # (B,D,L,L)
        right_b = right.permute(0, 3, 1, 2)
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)   # (B,D,L,L)
        tri_lld = tri.permute(0, 2, 3, 1).contiguous()           # (B,L,L,D)

        out_n, _mean_out, rstd_out, xhat_out = _ln_fwd(tri_lld, ln_out_w, ln_out_b, eps)
        g_logit = x_n @ Wg
        gate = torch.sigmoid(g_logit)
        proj = out_n @ Wp
        y = proj * gate

        ctx.save_for_backward(x, x_n, xhat_in, rstd_in, WL, WLg, WR, WRg, Wg, Wp,
                              pL, gL, pR, gR, left_b, right_b,
                              xhat_out, rstd_out, ln_out_w, out_n, gate, proj)
        ctx.ln_in_w = ln_in_w
        ctx.shape = (B, L, D)
        return y

    @staticmethod
    def backward(ctx, dy):
        (_x, x_n, xhat_in, rstd_in, WL, WLg, WR, WRg, Wg, Wp,
         pL, gL, pR, gR, left_b, right_b,
         xhat_out, rstd_out, ln_out_w, out_n, gate, proj) = ctx.saved_tensors
        ln_in_w = ctx.ln_in_w
        B, L, D = ctx.shape
        M = B * L * L

        def flat(t):
            return t.reshape(M, D)

        # ── ① back-half ──────────────────────────────────────────────────
        d_proj = dy * gate                                  # (B,L,L,D)
        d_gate = dy * proj
        d_glog = d_gate * gate * (1 - gate)                 # σ'
        # proj GEMM: out_n@Wp
        d_out_n = d_proj @ Wp.t()                           # (B,L,L,D)
        dWp = flat(out_n).t() @ flat(d_proj)                # (D,D)
        # gate GEMM: x_n@Wg
        dx_n = d_glog @ Wg.t()                              # gate→x_n
        dWg = flat(x_n).t() @ flat(d_glog)
        # LN_out bwd → d_tri (B,L,L,D), then to bdll
        d_tri_lld, dWln_out, dBln_out = _ln_bwd(d_out_n, xhat_out, rstd_out, ln_out_w)
        d_tri = d_tri_lld.permute(0, 3, 1, 2).contiguous()  # (B,D,L,L)

        # ── ② bmm bwd ────────────────────────────────────────────────────
        # tri = einsum("bdik,bdjk->bdij", left, right)
        d_left_b = torch.einsum("bdij,bdjk->bdik", d_tri, right_b)
        d_right_b = torch.einsum("bdij,bdik->bdjk", d_tri, left_b)
        d_left = d_left_b.permute(0, 2, 3, 1).contiguous()  # (B,L,L,D)
        d_right = d_right_b.permute(0, 2, 3, 1).contiguous()

        # ── ③ front gated-GEMM bwd (left & right) ────────────────────────
        def gated_bwd(d_out, p, g, Wp_proj, Wg_gate):
            d_p = d_out * g
            d_g = d_out * p
            d_glogit = d_g * g * (1 - g)
            dxn = d_p @ Wp_proj.t() + d_glogit @ Wg_gate.t()
            dWproj = flat(x_n).t() @ flat(d_p)
            dWgate = flat(x_n).t() @ flat(d_glogit)
            return dxn, dWproj, dWgate

        dxn_L, dWL, dWLg = gated_bwd(d_left, pL, gL, WL, WLg)
        dxn_R, dWR, dWRg = gated_bwd(d_right, pR, gR, WR, WRg)
        dx_n = dx_n + dxn_L + dxn_R

        # ── ④ LN_in bwd ──────────────────────────────────────────────────
        dx, dWln_in, dBln_in = _ln_bwd(dx_n, xhat_in, rstd_in, ln_in_w)

        # grads match forward arg order:
        # x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps
        return (dx, dWL, dWLg, dWR, dWRg, dWg, dWp,
                dWln_in, dBln_in, dWln_out, dBln_out, None)


def trimul_manual(x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                  ln_out_w, ln_out_b, eps=1e-5):
    return TriMulManualBwd.apply(x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                                 ln_out_w, ln_out_b, eps)
