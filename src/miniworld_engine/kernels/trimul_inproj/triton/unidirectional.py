"""Single-direction Triton TriMul with the bidirectional training fusions.

The shared front emits masked, gated projections in channel-major layout, then
one cuBLAS contraction computes outgoing or incoming triangles. BF16 training
uses output LN followed by F567 (projection, gate, dropout and residual), and
fuses the two input-gradient GEMMs. Input LN backward adds the residual gradient
in its store. FP32 retains the split GEMMs. Inference keeps the existing fused
LN/projection/gate output kernel. B=1; d_hidden == d_pair.
"""

from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
    _te_backward,
    _te_forward,
    _ln_materialize,
)
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import (
    input_dual_bwd,
    input_ln_residual,
)
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
    bidir_front_triton,
)
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_bwd_ew,
    gate_elem_train,
    ones_dropscale,
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
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, eps, outgoing,
                mask=None, residual=None, dropscale=None):
        B, L, _, D = x_n.shape
        M = B * L * L
        H = WL.shape[1]                                          # per-side hidden = d_hidden
        left, right, preact = bidir_front_triton(x_n, WL, WLg, WR, WRg, pair_mask=mask)
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        # Mask applies to the contraction inputs (left/right) ONLY — NOT to x_n, so
        # the output gate sigmoid(x_n@Wg) stays unmasked (matches the pytorch/cuequiv
        # reference). mask is (B=1,L,L) -> broadcast over the H channel axis.
        mm = mask  # Applied inside the front stores; x_n/output gate stay unmasked.
        tri = _contract(lf, rf, outgoing)                        # (H, L, L)
        view = tri.reshape(H, M).t()                             # (M, H) m-major
        if x_n.dtype == torch.bfloat16:
            te_xn, mean_out, rstd_out = _ln_materialize(
                view, ln_out_w, ln_out_b, eps, shape_key=both_key(M))
            y, proj, gate = output_f567_train(
                te_xn, x_n.reshape(M, D), Wp, Wg, residual, dropscale, L)
        else:
            proj, te_xn, mean_out, rstd_out = _te_forward(
                view, ln_out_w, ln_out_b, Wp, None, eps, shape_key=both_key(M))
            y, gate = gate_elem_train(
                x_n.reshape(M, D), proj, Wg, residual, dropscale, seq_len=L)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.outgoing, ctx.mm = eps, outgoing, mm
        ctx.dropscale, ctx.seq_len = dropscale, L
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

        # residual grad passes straight through (d/d_residual [residual + drop⊙op] = 1); the op
        # branch grad is scaled by the same drop_row mask inside gate_elem_bwd_ew.
        d_residual = gy.reshape(M, D)              # match the residual input [M,D]
        # ② gate bwd (elementwise; dx_gate folded into the dxn GEMM below); dropout-scale dy
        d_proj, d_glogit = gate_elem_bwd_ew(gy.contiguous(), proj.contiguous(), gate.contiguous(),
                                            ctx.dropscale, ctx.seq_len)
        # `del` after last use, inserted where no reference to the name remains anywhere below.
        # autograd frees an intermediate when its consumer node has run; this function holds every
        # local until it returns, and these are pair-shaped -- 144 MiB each at B=1 L=768 d=128
        # bf16. Measured on the triton bidirectional twin: 1,008 MiB off a 7,662 MiB peak.
        del gy
        dWg = torch.mm(x_n.reshape(M, D).t(), d_glogit)          # (D, D) cuBLAS

        # ① LN_out + @Wp bwd (te_style)
        view = tri.reshape(H, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False,
            shape_key=both_key(M))
        del d_proj, view
        d_tri = d_view.t().reshape(H, L, L)
        del d_view

        # contraction bwd (single direction), cuBLAS bmm. lf/rf are the MASKED
        # inputs (saved post-mask), so these grads are w.r.t. the masked tensors.
        if outgoing:                                             # O = lf @ rfᵀ
            d_left = torch.bmm(d_tri, rf)
            d_right = torch.bmm(d_tri.transpose(1, 2), lf)
        else:                                                    # O = lfᵀ @ rf
            d_left = torch.bmm(rf, d_tri.transpose(1, 2))
            d_right = torch.bmm(lf, d_tri)
        del d_tri
        d_left = d_left.reshape(B, H, L, L)
        d_right = d_right.reshape(B, H, L, L)

        # front bwd: d_concat (triton) + dW (cuBLAS) + W_stack; dxn fuses the gate add
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
            d_left, d_right, preact, x_n, WL, WLg, WR, WRg, pair_mask=ctx.mm)
        del d_left, d_right
        if x_n.dtype == torch.bfloat16:
            dx = input_dual_bwd(d_glogit, dconc.t(), Wg.t(), W_stack, L)
        else:
            dx = torch.mm(d_glogit, Wg.t())
            dx.addmm_(dconc.t(), W_stack)
        del d_glogit
        del W_stack, dconc
        dx_n = dx.reshape(B, L, L, D)
        del dx
        # trailing Nones: eps, outgoing, mask; then d_residual (fused residual input), dropscale
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None,
                d_residual, None)


@torch.no_grad()
def _uni_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wp, ln_out_w, ln_out_b, eps, outgoing, mask=None,
               residual=None):
    """Forward-only single-direction back-half — SAME kernel structure as the merged
    Function but NO autograd.Function / saved tensors and NO preact side output, so
    it cudagraphs at cute's speed (the merged Function's saves are used only under
    grad). Mirrors the bidir ``_bidir_infer``."""
    B, L, _, D = x_n.shape
    left, right, _ = bidir_front_triton(
        x_n, WLt, WLgt, WRt, WRgt, save_preact=False, pair_mask=mask)
    H = left.shape[1]
    lf = left.reshape(H, L, L)
    rf = right.reshape(H, L, L)
    tri = _contract(lf, rf, outgoing)                           # (H, L, L)
    # Fused back-half: LN_out + proj-gemm + gate-gemm + mul in ONE kernel (``trimul_back_triton``),
    # the exact kernel the H100 sm90 cute path already uses (module ``_forward_cute_free``). It
    # replaces the 2-kernel ``_te_forward`` (LN_out+proj) + ``gate_elem_triton`` (gate+mul) split:
    # one fewer launch and one fewer HBM round-trip of the [L,L,D] proj tensor. All triton, so it
    # runs on A100/sm86 as well as the Hopper path it came from.
    #
    # MEASURED, against a tuned cache, both paths checked for correctness before any timing:
    #
    #   L=384   fused 0.355 ms   split 0.444 ms   1.25x   rel_err 2.7e-03 vs 3.9e-03
    #   L=768   fused 1.028 ms   split 1.360 ms   1.32x   rel_err 3.1e-03 vs 4.1e-03
    #
    # Faster AND closer to the fp32 reference, because the fused kernel keeps proj and the gate
    # logit in its fp32 accumulator where the split path materialises each as bf16 first. NOT
    # bit-identical to the split path -- that earlier claim was an artefact, see below.
    #
    # It took five attempts to get a number worth trusting, and each failure is worth keeping:
    #   1-3. `to_out` is a zero-initialised Linear, so `y = pair + 0` and every variant was
    #        trivially equal -- three probes reported rel_err 0.000e+00 against an fp32 reference,
    #        impossible for a bf16 kernel. Randomising to_out/to_gate is what made the comparison
    #        mean anything, and is what showed the two paths are NOT bit-identical.
    #   4.   the fused path was timed while this kernel's cache missed on the DTYPE axis at every
    #        launch -- production keys `bfloat16+float32` (the LN_out affine is fp32, pinned by
    #        `primitives._Fp32ParamsMixin`) against a `bfloat16`-only cache -- so it ran on the
    #        bounded heuristic subset against a split path that had tuned configs. That is where
    #        "1.03x, no win" came from.
    #   5.   the two paths were handed the same weight form, when `trimul_back_triton` takes
    #        `to_out.weight.T` and `_te_forward` takes it as-is. The split path then scored
    #        rel_err 1.03 -- unrelated to the answer -- and its "1.35x" was timing a wrong result.
    #
    # `.bench/probe/fused_back_ab.py` carries all five guards, and refuses to print a speed if
    # either path is off the fp32 reference by more than 5e-2.
    #
    # Training retains output LN as a separate stage and uses F567 for BF16 so
    # its saved activations retain the original rounding points.
    # Weight forms: trimul_back wants ``.T`` weights, so Wp (to_out, nn.Linear form) -> Wp.T; Wgt is
    # already to_gate.weight.T. residual comes in flat [M,D] and is reshaped to [B,L,L,D].
    return trimul_back_triton(tri.unsqueeze(0), x_n, Wp.T.contiguous(), Wgt,
                              ln_out_w, ln_out_b, eps, residual=residual.view(B, L, L, D))


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
    dropscale=None,              # drop_row scale [B,1,L,D] (== mask/(1-p)); training only
):
    """Faithful triton mirror of the single-direction cute trimul. Returns
    (B, L, L, d_pair). Mirrors cute's dispatch exactly: LN_in (triton, row_scale
    mask), then a forward-only path for inference (``_uni_infer``) and the merged
    autograd Function for training (``_UniBackHalfTriton``). All-triton/cuBLAS;
    requires d_hidden == d_pair (the front produces per-side width d_hidden).

    B==1 by design (inherited from the shared bidir front + bdll layout). B>1 was implemented
    + verified correct but is slower than looping this path per batch (L2 thrashing of large
    bdll intermediates); see notes/trimul_batch_generalization."""
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
    B, L = pair.shape[0], pair.shape[1]
    M = B * L * L
    # The residual is the ORIGINAL (pre-LN_in) input pair, and it is not optional: this op is
    # ``y = pair + drop_row(trimul(pair))``. dropscale [B,1,L,D] -> [L,D] (B==1) for the gate
    # store's row-broadcast indexing. Both fold into the gate_elem epilogue (no external add).
    x_n, residual = input_ln_residual(pair, ln_in_w, ln_in_b, eps_in)
    residual_flat = residual.reshape(M, d)
    ds_2d = dropscale.reshape(L, d) if dropscale is not None else None
    WLt, WLgt = WL.t().contiguous(), WLg.t().contiguous()
    WRt, WRgt, Wgt = WR.t().contiguous(), WRg.t().contiguous(), Wg.t().contiguous()
    if not torch.is_grad_enabled() and ds_2d is None:
        # INFERENCE: forward-only (no saved tensors) — cudagraphs at cute's speed. Also gate on
        # ds_2d is None: a live dropout scale (train() under no_grad, p_drop>0) must take the
        # TRAINING apply below (which folds dropout into the gate epilogue) — the inference path
        # has no dropscale and would silently skip dropout. Mirrors the cute dispatch's guard.
        return _uni_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout,
                          ln_out_w, ln_out_b, eps_out, outgoing, mask=m2d, residual=residual_flat)
    # The training path ALWAYS carries a drop scale, ones when the model's p_drop is 0. That is
    # what lets the training kernel be flagless: `gate_elem_train` and `gate_elem_bwd_ew` apply it
    # unconditionally instead of branching on a keyed constexpr, at a measured 0.2% (the scale is
    # [L, N] and stays in L2). Building it here, at the one place that decides "this is training",
    # keeps that decision in a single spot.
    if ds_2d is None:
        ds_2d = ones_dropscale(L, d, pair)
    # TRAINING: merged autograd Function (weights x@W; autograd flows the transpose).
    return _UniBackHalfTriton.apply(
        x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout, ln_out_w, ln_out_b, eps_out, outgoing, m2d,
        residual_flat, ds_2d,
    )
