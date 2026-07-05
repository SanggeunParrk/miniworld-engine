"""B200 TRAINING (fwd+bwd) path for trimul (outgoing).

FAITHFUL PORT of the H100 training design (``cute/training.py`` = ``TriMulInprojFn``):
a *save-heavy* forward + a manual backward that consumes the saved tensors with
NO recompute (mirrors ``training.py`` stage-for-stage).  Only the GEMM backend of
the FORWARD front differs: the fast sm100 tcgen05 gated-GLU collective produces
``left_b/right_b`` (bdll) instead of quack/WGMMA.

vs the previous B200 path (v0, git eb41ca4): that one used the max-fusion inference
forward (fused triton back) which saved nothing, so its backward had to recompute
the WHOLE back-half.  This module restores the H100 save-heavy contract:

  saved: x_n, xhat_in, rstd_in, left_b, right_b, xhat_out, rstd_out, out_n, gate,
         proj  (+ weights)  ->  backward does ZERO forward recompute.

Backward GEMMs are plain torch/cuBLAS, EXCEPT:
  - front gated-GEMM bwd -> FUSED sm100 backward_gatebwd port (v2, gatebwd_sm100).
  - LN_out backward (d_tri_lld = LNbwd(d_proj @ Wp.t())) -> FUSED sm100 dgrad_lnbwd
    port (v4/v5, dgrad_lnbwd_sm100): the projection-backward GEMM + the LN-normalize
    backward are folded into ONE tcgen05 epilogue, removing the d_out_n (M,D) HBM
    round-trip and the separate torch _ln_bwd memory pass.  LN_out affine grads
    (dgamma/dbeta) derived from T = d_proj^T @ xhat_out (dgrad emits only dx).
    Toggle: MINIWORLD_TRAIN_DGRAD_FUSED=0.
  - the LLD<->DLL permutes (d_tri, d_left, d_right) use a COALESCED tiled triton
    transpose (v6) instead of torch .permute().contiguous() (an uncoalesced generic
    copy, ~2.3 ms each at L=1024 vs ~0.16 ms tiled).  Same values, ~5 ms/iter less
    memory traffic.  Toggle: MINIWORLD_TRAIN_FAST_PERMUTE=0.

Mask: applied multiplicatively to x_n AFTER LN_in (== training.py).  dx_n accumulates
from front + gate + back-half, then the mask multiply, then LN_in backward.
LN stats are fp32 (``_ln_fwd``/``_ln_bwd``).  B == 1, square L, D == 128, bf16.
"""

from __future__ import annotations

import os

import torch
import triton

from miniworld_kernels.kernels.trimul_inproj.autograd import _ln_bwd, _ln_fwd
from miniworld_kernels.kernels.trimul_inproj.cute.front_sm100_fused import (
    prepack_lr_operand_sm100,
    trimul_front_sm100_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute.gatebwd_sm100 import front_gatebwd_sm100
from miniworld_kernels.kernels.layernorm_linear.cute.dgrad_lnbwd_sm100 import dgrad_lnbwd_sm100
from miniworld_kernels.kernels.trimul_inproj.cute.front_sm100 import (
    _transpose_kernel, get_seq_group,
)

_USE_FUSED_DGRAD = os.environ.get("MINIWORLD_TRAIN_DGRAD_FUSED", "1") != "0"
_USE_FAST_PERMUTE = os.environ.get("MINIWORLD_TRAIN_FAST_PERMUTE", "1") != "0"


def _fast_T(src):
    """Coalesced tiled transpose: src (M,N) row-major contiguous -> (N,M) row-major.
    Replaces torch .permute().contiguous() (uncoalesced generic copy, ~2.3 ms per
    L=1024 transpose) with the front tiled triton transpose (~0.16 ms). Same values."""
    M, N = src.shape
    dst = torch.empty(N, M, device=src.device, dtype=src.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
    _transpose_kernel[grid](src, dst, M, N, GROUP_M=get_seq_group(M))
    return dst


import triton.language as tl


@triton.jit
def _glu_bwd_kernel(dy_ptr, gate_ptr, proj_ptr, dproj_ptr, dglog_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    dy = tl.load(dy_ptr + off, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + off, mask=mask).to(tl.float32)
    p = tl.load(proj_ptr + off, mask=mask).to(tl.float32)
    tl.store(dproj_ptr + off, (dy * g).to(tl.bfloat16), mask=mask)
    tl.store(dglog_ptr + off, (dy * p * g * (1.0 - g)).to(tl.bfloat16), mask=mask)


def _glu_bwd(dy, gate, proj):
    d_proj = torch.empty_like(dy)
    d_glog = torch.empty_like(dy)
    n = dy.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _glu_bwd_kernel[grid](dy, gate, proj, d_proj, d_glog, n, BLOCK=2048)
    return d_proj, d_glog


_USE_FUSE_GLU = os.environ.get("MINIWORLD_TRAIN_FUSE_GLU", "1") != "0"


@triton.jit
def _add3_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    a = tl.load(a_ptr + off, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + off, mask=mask).to(tl.float32)
    c = tl.load(c_ptr + off, mask=mask).to(tl.float32)
    tl.store(out_ptr + off, (a + b + c).to(tl.bfloat16), mask=mask)


def _add3(a, b, c):
    out = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _add3_kernel[grid](a, b, c, out, n, BLOCK=2048)
    return out


_USE_FUSE_LN = os.environ.get("MINIWORLD_TRAIN_FUSE_LN", "1") != "0"

_LN_CFGS = [
    triton.Config({"BLOCK_M": bm}, num_warps=nw)
    for bm in (1, 2, 4, 8, 16)
    for nw in (1, 2, 4)
]


@triton.autotune(configs=_LN_CFGS, key=["D"])
@triton.jit
def _ln_fwd_kernel(X, W, Bs, Y, Rstd, Xhat, M, D: tl.constexpr, eps,
                   BLOCK_M: tl.constexpr):
    # One program does BLOCK_M rows, full D columns; LN over D. fp32 stats.
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    mmask = rm < M
    off = rm[:, None] * D + rd[None, :]
    x = tl.load(X + off, mask=mmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd[:, None]
    w = tl.load(W + rd).to(tl.float32)
    b = tl.load(Bs + rd).to(tl.float32)
    y = xhat * w[None, :] + b[None, :]
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mmask[:, None])
    tl.store(Xhat + off, xhat.to(Xhat.dtype.element_ty), mask=mmask[:, None])
    tl.store(Rstd + rm, rstd, mask=mmask)


def _ln_fwd_fused(x, w, b, eps):
    """Fused LayerNorm fwd over last dim D. Returns (y, None, rstd_fp32, xhat_bf16).
    Drop-in for autograd._ln_fwd (mean unused by callers). fp32 stats (constraint)."""
    xf = x.reshape(-1, x.shape[-1])
    M, D = xf.shape
    y = torch.empty_like(xf)
    xhat = torch.empty_like(xf)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _ln_fwd_kernel[grid](xf, w, b, y, rstd, xhat, M, D=D, eps=float(eps))
    shp = x.shape
    return (y.view(shp), None, rstd.view(shp[:-1]), xhat.view(shp))


@triton.autotune(configs=_LN_CFGS, key=["D"])
@triton.jit
def _ln_bwd_dx_kernel(DY, Xhat, Rstd, W, DX, M, D: tl.constexpr,
                      BLOCK_M: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    mmask = rm < M
    off = rm[:, None] * D + rd[None, :]
    dy = tl.load(DY + off, mask=mmask[:, None], other=0.0).to(tl.float32)
    xh = tl.load(Xhat + off, mask=mmask[:, None], other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + rm, mask=mmask, other=0.0).to(tl.float32)
    w = tl.load(W + rd).to(tl.float32)
    dxhat = dy * w[None, :]
    mean1 = tl.sum(dxhat, axis=1) / D
    mean2 = tl.sum(dxhat * xh, axis=1) / D
    dx = rstd[:, None] * (dxhat - mean1[:, None] - xh * mean2[:, None])
    tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=mmask[:, None])


def _ln_bwd_fused(dy, xhat, rstd, w):
    """Fused LayerNorm bwd over last dim D. Returns (dx, dw, db). dx in one fused
    pass; dw/db are cheap (M,D)->(D,) reductions (torch). Drop-in for autograd._ln_bwd."""
    D = dy.shape[-1]
    dyf = dy.reshape(-1, D)
    xhf = xhat.reshape(-1, D)
    M = dyf.shape[0]
    dx = torch.empty_like(dyf)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _ln_bwd_dx_kernel[grid](dyf, xhf, rstd.reshape(-1).contiguous(), w, dx, M, D=D)
    dw = (dyf * xhf).sum(0)
    db = dyf.sum(0)
    return dx.view(dy.shape), dw, db


class TriMulB200Fn(torch.autograd.Function):
    """Save-heavy sm100 forward + manual backward (torch/cuBLAS + fused gatebwd/dgrad)."""

    @staticmethod
    def forward(ctx, pair, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b,
                ln_out_w, ln_out_b, eps, packed, mask2d):
        B, L, _, D = pair.shape

        # -- LN_in (fp32 stats saved, faithful to training.py _ln_fwd) --
        _lnf = _ln_fwd_fused if _USE_FUSE_LN else _ln_fwd
        x_n, _, rstd_in, xhat_in = _lnf(pair, ln_in_w, ln_in_b, eps)
        if mask2d is not None:
            x_n = x_n * mask2d

        # -- front: FAST sm100 tcgen05 gated-GLU GEMM -> bdll left/right (kept) --
        left_b, right_b = trimul_front_sm100_fused(x_n, packed=packed)

        # -- bmm (channel-wise outer over k) --
        tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)  # (B, D, L, L)
        if _USE_FAST_PERMUTE:
            tri_lld = _fast_T(tri.reshape(D, L * L)).view(B, L, L, D)  # DLL->LLD (tiled)
        else:
            tri_lld = tri.permute(0, 2, 3, 1).contiguous()          # (B, L, L, D)

        # -- back-half (computed + SAVED, == training.py) --
        out_n, _, rstd_out, xhat_out = _lnf(tri_lld, ln_out_w, ln_out_b, eps)
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
        if _USE_FUSE_GLU:
            d_proj, d_glog = _glu_bwd(dy, gate, proj)   # 1 fused pass (was 5 elementwise)
        else:
            d_proj = dy * gate
            d_gate = dy * proj
            d_glog = d_gate * gate * (1 - gate)
        dWp = flat(out_n).t() @ flat(d_proj)
        dx_n = d_glog @ Wg.t()
        dWg = flat(x_n).t() @ flat(d_glog)

        if _USE_FUSED_DGRAD:
            # FUSED: d_tri_lld = LNbwd(d_proj @ Wp.t()) in the dgrad tcgen05 epilogue.
            dpf = flat(d_proj)
            xho = flat(xhat_out)
            Wpt = Wp.t().contiguous()                              # (D, D)
            d_tri_lld_flat = dgrad_lnbwd_sm100(
                dpf, Wpt, xho, ln_out_w, rstd_out.reshape(-1))
            d_tri_lld = d_tri_lld_flat.view(B, L, L, D)
            # LN_out affine grads from T (== _ln_bwd dw/db, no d_out_n materialize):
            T = dpf.t() @ xho                                       # (D, D)
            db_proj = dpf.sum(0)                                    # (D,)
            dBln_out = db_proj @ Wpt                                # (D,)
            dWln_out = (Wpt * T).sum(0)                             # (D,)
        else:
            d_out_n = d_proj @ Wp.t()
            d_tri_lld, dWln_out, dBln_out = _ln_bwd(d_out_n, xhat_out, rstd_out, ln_out_w)

        if _USE_FAST_PERMUTE:
            d_tri = _fast_T(d_tri_lld.reshape(M, D)).view(B, D, L, L)  # LLD->DLL (tiled)
        else:
            d_tri = d_tri_lld.permute(0, 3, 1, 2).contiguous()      # (B, D, L, L)

        # -- (2) bmm bwd (bdll) --
        d_left_b = torch.einsum("bdij,bdjk->bdik", d_tri, right_b)
        d_right_b = torch.einsum("bdij,bdik->bdjk", d_tri, left_b)
        if _USE_FAST_PERMUTE:
            d_left = _fast_T(d_left_b.reshape(D, M)).view(B, L, L, D)   # DLL->LLD (tiled)
            d_right = _fast_T(d_right_b.reshape(D, M)).view(B, L, L, D)
        else:
            d_left = d_left_b.permute(0, 2, 3, 1).contiguous()      # (B, L, L, D)
            d_right = d_right_b.permute(0, 2, 3, 1).contiguous()

        # -- (3) front gated-GEMM bwd — FUSED sm100 port of H100 backward_gatebwd --
        xnf = flat(x_n)
        dxn_L, dWL, dWLg = front_gatebwd_sm100(xnf, flat(d_left), WL, WLg)
        dxn_R, dWR, dWRg = front_gatebwd_sm100(xnf, flat(d_right), WR, WRg)
        if _USE_FUSE_GLU:
            dx_n = _add3(dx_n, dxn_L.view_as(dx_n), dxn_R.view_as(dx_n))  # 1 fused add (was 2)
        else:
            dx_n = dx_n + dxn_L.view_as(dx_n) + dxn_R.view_as(dx_n)

        if ctx.has_mask:
            dx_n = dx_n * ctx.mask2d

        # -- (4) LN_in bwd (saved stats, NO recompute; pure LN-bwd, no GEMM to fuse) --
        _lnb = _ln_bwd_fused if _USE_FUSE_LN else _ln_bwd
        dx, dWln_in, dBln_in = _lnb(dx_n, xhat_in, rstd_in, ln_in_w)

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
