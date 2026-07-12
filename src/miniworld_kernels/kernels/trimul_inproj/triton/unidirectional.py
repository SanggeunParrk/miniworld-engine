"""Single-direction trimul in TRITON — the SAME fused BDLL pipeline the bidir path
uses (``triton/bidirectional.py``), specialised to one direction (outgoing OR
incoming). Mirrors the CUTE single-direction dispatch stage-for-stage; only the
backend differs (triton/cuBLAS, no quack).

The pipeline is:

    x_n   = triton_layernorm(pair, ...)              # LN_in
    left,right,preact = FRONT(x_n, WL,WLg,WR,WRg)    # gated in-proj, width=d_hidden, bdll
    tri   = bmm(lf, rfᵀ)      (outgoing)  |  bmm(lfᵀ, rf)  (incoming)   # ONE contraction
    proj  = _te_forward(tri_view, ln_out, Wp)        # LN_out + @Wp  (te_style: triton LN + cuBLAS)
    y     = gate_elem(x_n, proj, Wg)                 # sigmoid output-gate  (triton)

This is exactly the bidir back-half with h→full width and a SINGLE contraction:
the FRONT (``bidir_front_triton``) is direction-agnostic (a wide gated GEMM that
emits left/right in channel-major BDLL directly via a transposed store — NO
permute), so it is reused verbatim; the only per-direction logic is the one
``torch.bmm`` (and its transpose in the backward). All the heavy machinery
(``_te_forward/_te_backward``, ``front_bwd_dW``, ``gate_elem_*``) is shared with
bidir. B=1, bf16 / fp32. Requires d_hidden == d_pair.
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm_linear.te_style import (
    _te_backward,
    _te_forward,
)
from miniworld_kernels.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_kernels.kernels.trimul_inproj.triton.bidirectional import (
    bidir_front_triton,
)
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_bwd_ew,
    gate_elem_triton,
)


def _contract(lf, rf, outgoing):
    """The single triangle contraction on the BDLL tensors (channel-major bmm =
    cuBLAS, exactly cute's dispatch.bmm). lf/rf:(D,L,L).
      outgoing  O[d,i,j] = Σ_k lf[d,i,k]·rf[d,j,k]  = bmm(lf, rfᵀ)
      incoming  O[d,i,j] = Σ_k lf[d,k,i]·rf[d,k,j]  = bmm(lfᵀ, rf)
    """
    if outgoing:
        return torch.bmm(lf, rf.transpose(1, 2))
    return torch.bmm(lf.transpose(1, 2), rf)


# ── merged back-half (mirror of the bidir _BidirBackHalfTriton), fwd + manual bwd ──
class _UniBackHalfTriton(torch.autograd.Function):
    """front → 1 contraction (outgoing OR incoming) → LN_out+@Wp → gate, as ONE
    Function so the backward matches the fused structure (gate dx_n add folded into
    the front dxn GEMM). Weights x@W form; Wp is nn.Linear (N,K) form."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, eps, outgoing, mask=None):
        B, L, _, D = x_n.shape
        M = B * L * L
        H = WL.shape[1]                                          # per-side hidden = d_hidden
        left, right, preact = bidir_front_triton(x_n, WL, WLg, WR, WRg)
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        # Mask applies to the contraction inputs (left/right) ONLY — NOT to x_n, so
        # the output gate sigmoid(x_n@Wg) stays unmasked (matches the pytorch/cuequiv
        # reference). mask is (B=1,L,L) -> broadcast over the H channel axis.
        mm = None
        if mask is not None:
            mm = mask.reshape(L, L).to(lf.dtype)
            lf = lf * mm
            rf = rf * mm
        tri = _contract(lf, rf, outgoing)                        # (H, L, L)
        view = tri.reshape(H, M).t()                             # (M, H) m-major
        proj, te_xn, mean_out, rstd_out = _te_forward(
            view, ln_out_w, ln_out_b, Wp, None, eps)             # (M, D)
        y, gate = gate_elem_triton(x_n.reshape(M, D), proj, Wg, return_gate=True)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.outgoing, ctx.mm = eps, outgoing, mm
        return y.reshape(B, L, L, D)

    @staticmethod
    def backward(ctx, gy):
        (x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
         preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj) = ctx.saved_tensors
        B, L, _, D = x_n.shape
        M = B * L * L
        H = WL.shape[1]
        outgoing = ctx.outgoing
        gy = gy.reshape(M, D)

        # ② gate bwd (elementwise; dx_gate folded into the dxn GEMM below)
        d_proj, d_glogit = gate_elem_bwd_ew(gy.contiguous(), proj.contiguous(), gate.contiguous())
        dWg = torch.mm(x_n.reshape(M, D).t(), d_glogit)          # (D, D) cuBLAS

        # ① LN_out + @Wp bwd (te_style)
        view = tri.reshape(H, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        d_tri = d_view.t().reshape(H, L, L)

        # contraction bwd (single direction), cuBLAS bmm. lf/rf are the MASKED
        # inputs (saved post-mask), so these grads are w.r.t. the masked tensors.
        if outgoing:                                             # O = lf @ rfᵀ
            d_left = torch.bmm(d_tri, rf)
            d_right = torch.bmm(d_tri.transpose(1, 2), lf)
        else:                                                    # O = lfᵀ @ rf
            d_left = torch.bmm(rf, d_tri.transpose(1, 2))
            d_right = torch.bmm(lf, d_tri)
        # chain back through the elementwise mask (left_masked = left*mm)
        if ctx.mm is not None:
            d_left = d_left * ctx.mm
            d_right = d_right * ctx.mm
        d_left = d_left.reshape(B, H, L, L)
        d_right = d_right.reshape(B, H, L, L)

        # front bwd: d_concat (triton) + dW (cuBLAS) + W_stack; dxn fuses the gate add
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
            d_left, d_right, preact, x_n, WL, WLg, WR, WRg)
        dx = torch.mm(d_glogit, Wg.t())                         # dx_gate  (M, D)
        dx.addmm_(dconc.t(), W_stack)                           # + dconcᵀ@W_stack (in-place)
        dx_n = dx.reshape(B, L, L, D)
        # trailing Nones: eps, outgoing, mask (non-differentiable)
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None)


@torch.no_grad()
def _uni_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wp, ln_out_w, ln_out_b, eps, outgoing, mask=None):
    """Forward-only single-direction back-half — SAME kernel structure as the merged
    Function but NO autograd.Function / saved tensors and NO preact side output, so
    it cudagraphs at cute's speed (the merged Function's saves are used only under
    grad). Mirrors the bidir ``_bidir_infer``."""
    B, L, _, D = x_n.shape
    M = B * L * L
    left, right, _ = bidir_front_triton(x_n, WLt, WLgt, WRt, WRgt, save_preact=False)
    H = left.shape[1]
    lf = left.reshape(H, L, L)
    rf = right.reshape(H, L, L)
    if mask is not None:                                        # mask left/right only
        mm = mask.reshape(L, L).to(lf.dtype)
        lf = lf * mm
        rf = rf * mm
    tri = _contract(lf, rf, outgoing)                           # (H, L, L)
    proj = _te_forward(tri.reshape(H, M).t(), ln_out_w, ln_out_b, Wp, None, eps)[0]
    y = gate_elem_triton(x_n.reshape(M, D), proj, Wgt)          # gate GEMM + sigmoid·mul
    return y.view(B, L, L, D)


def trimul_triton(
    pair,                        # (B, L, L, d_pair)
    WL, WLg, WR, WRg,            # to_{left,left_gate,right,right_gate}.weight  (d_hidden, d_pair)
    Wg,                          # to_gate.weight   (d_pair, d_pair)
    Wout,                        # to_out.weight    (d_pair, d_hidden)  (nn.Linear form)
    ln_in_w, ln_in_b,            # (d_pair,)
    ln_out_w, ln_out_b,          # (d_hidden,)
    eps_in, eps_out, d_hidden,
    outgoing,                    # bool: outgoing (True) or incoming (False)
    mask=None,                   # (B,L) residue OR (B,L,L) pair mask, optional (folded into LN_in)
):
    """Faithful triton mirror of the single-direction cute trimul. Returns
    (B, L, L, d_pair). Mirrors cute's dispatch exactly: LN_in (triton, row_scale
    mask), then a forward-only path for inference (``_uni_infer``) and the merged
    autograd Function for training (``_UniBackHalfTriton``). All-triton/cuBLAS;
    requires d_hidden == d_pair (the front produces per-side width d_hidden)."""
    d = pair.shape[-1]
    if d_hidden != d:
        raise ValueError(
            f"TRITON single-direction trimul requires d_hidden == d_pair "
            f"(got d_hidden={d_hidden}, d_pair={d})."
        )
    # The mask is applied to the contraction inputs (left/right), NOT folded into
    # LN_in: x_n must stay unmasked so the output gate sigmoid(x_n@Wg) is unmasked
    # (matches the pytorch/cuequiv reference). Build the (B,L,L) pair mask here.
    m2d = None
    if mask is not None:
        if mask.dim() == 2:                                    # (B, L) residue mask
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)        # (B, L, L)
        else:                                                  # (B, L, L) pair mask (cuequiv form)
            m = mask
        m2d = m.to(pair.dtype)                                  # (B, L, L)
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps_in)
    WLt, WLgt = WL.t().contiguous(), WLg.t().contiguous()
    WRt, WRgt, Wgt = WR.t().contiguous(), WRg.t().contiguous(), Wg.t().contiguous()
    if not torch.is_grad_enabled():
        # INFERENCE: forward-only (no saved tensors) — cudagraphs at cute's speed.
        return _uni_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout,
                          ln_out_w, ln_out_b, eps_out, outgoing, mask=m2d)
    # TRAINING: merged autograd Function (weights x@W; autograd flows the transpose).
    return _UniBackHalfTriton.apply(
        x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout, ln_out_w, ln_out_b, eps_out, outgoing, m2d,
    )
