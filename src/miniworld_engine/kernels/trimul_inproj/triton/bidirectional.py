"""Bidirectional trimul in TRITON — a faithful 1:1 mirror of the CUTE bidir
(``cute/bidir_training.py`` ``BidirBackHalf`` + ``bidir_forward``), same algorithm
and same fusion boundaries, ONLY the backend differs (triton/cuBLAS, no quack).

Stage-for-stage the cute bidir is:

    x_n   = triton_layernorm(pair, ...)              # LN_in  (already triton in cute)
    left,right,preact = FRONT(x_n, WL,WLg,WR,WRg)    # gated in-proj, out_hidden=2h, bdll
    o_out = bmm(lf[:h], rf[:h]ᵀ) ;  o_in = bmm(lf[h:]ᵀ, rf[h:])   # 2 triangle contractions
    tri   = packed_forward(lf, rf, h)              # training: GEMMs write final (2h,L,L) buffer
    proj  = _te_forward(tri_view, ln_out, Wp)        # LN_out + @Wp   (te_style: triton LN + cuBLAS)
    y     = gate_elem(x_n, proj, Wg)                 # sigmoid output-gate  (triton)

and its backward is the merged BackHalf (gate-ew → dWg → te-bwd → contraction-bwd →
front-bwd, dxn fused with the gate add). We reuse the EXACT same helpers cute uses:
``_te_forward/_te_backward`` (layernorm_linear/te_style), ``front_bwd_dW``
(trimul_inproj/triton/back_fused), ``gate_elem_triton/gate_elem_bwd_ew``
(trimul_inproj/triton/gate_elem), ``triton_layernorm``. The two triangle
training contractions use ``torch.bmm(..., out=...)`` on the BDLL tensors,
removing one forward and two backward concatenations. The CuTe path uses the
same packed layout with its measured cuBLAS/Quack policy. The big GEMMs
(dWg, dxn) remain cuBLAS.

The ONLY new kernel here is the FRONT forward: cute's front is a quack gated
M-major GEMM; we write the equivalent in triton, producing left/right in BDLL
(channel-major) AND the interleaved ``preact`` (=[gLlog,pL] per channel, left then
right) that ``front_bwd_dW`` consumes. It reuses the ``front.py`` ``_lr_kernel``
design (half-accumulator, transposed bdll store) generalised to per-side width 2h
and extended to also store ``preact``. B=1, bf16 / fp32.
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl


from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
    _te_backward,
    _te_forward,
)
from miniworld_engine.autotune.shape_key import pack, token_key
from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward, packed_backward
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_bwd_ew,
    gate_elem_infer,
    gate_elem_train,
    ones_dropscale,
)






# SAVE_PREACT IS in the key: training keeps it on and the kernel emits two extra
# (H2 x BLOCK_M1) transposed stores per channel chunk per side -- the whole (4*H2, M) preact
# tensor, four of this kernel's six stores -- while inference writes only left/right. The tiles
# are already in registers, so the BODY reads as a pure store, but the swing in store traffic is
# what picks the tile here. See gate_elem.py for the same distinction, and adaln/triton/ln_strided.py
# for the GEMM where the opposite reading was the measured one.
@triton.autotune(configs=configs_for("trimul_gemm_gate_mmajor_triton"),
                 key=['shape_key', 'SAVE_PREACT'])
@triton.jit
def _bidir_front_kernel(
    x_ptr, w_ptr,
    left_ptr, right_ptr,
    preact_ptr, pair_mask_ptr,
    M, LL,
    K: tl.constexpr, H2: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K_D: tl.constexpr, BLOCK_K_H2: tl.constexpr, shape_key,
    SAVE_PREACT: tl.constexpr = True,
):
    # int64 program id -> rm and all rm-derived store offsets are int64. Combined with the
    # per-store int64 cast of the channel base below, this stops the preact store offset
    # (row_base up to 4*H2, times M) from overflowing int32 at large M -- e.g. 4096 * 768^2
    # ~ 2.4e9 > 2^31, which faulted with an illegal memory access at d=512, L>=768.
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rk = tl.arange(0, BLOCK_K_D)
    c2 = tl.arange(0, 2 * BLOCK_K_H2)                # contiguous interleaved (g,p) cols within a chunk
    mmask = rm < M
    smask = rm[None, :] < M
    W4 = 4 * H2
    et = left_ptr.dtype.element_ty
    if pair_mask_ptr is not None:
        pair_scale = tl.load(pair_mask_ptr + rm, mask=mmask, other=0).to(tl.float32)
    # The K loop walks the contraction axis in BLOCK_K_D steps and `rk` is never re-bounded, so a K
    # that is not a multiple of the tuned BLOCK_K_D made the last trip read columns K..ceil-1 --
    # the next row's leading channels for x, and past the end of the (K, 4*H2) weight entirely
    # (memcheck: invalid 8-byte global read 2001-4241 bytes past a 250000-byte allocation at K=125).
    # The row/column masks cannot help: the out-of-range index is on the contraction axis. Both
    # K and BLOCK_K_D are constexpr, so this folds at compile time and the aligned path keeps the
    # unmasked loads -- the same EVEN_* dispatch triangle_attention/triton/atomic.py:156 uses.
    EVEN_K = K % BLOCK_K_D == 0

    # ---- LEFT half: weight cols [0 : 2*H2), out -> left_ptr, preact rows [0 : 2*H2) ----
    for c0 in range(0, H2, BLOCK_K_H2):
        ch = c0 + tl.arange(0, BLOCK_K_H2)                       # channel indices this chunk
        chmask = ch < H2
        cols = 2 * c0 + c2                               # contiguous cols [2c0 : 2c0+2BN)
        colmask = cols < 2 * H2
        x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
        w_ptrs = w_ptr + rk[:, None] * W4 + cols[None, :]
        acc = tl.zeros((BLOCK_M1, 2 * BLOCK_K_H2), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K_D):
            if EVEN_K:
                a = tl.load(x_ptrs, mask=mmask[:, None], other=0.0)
                w = tl.load(w_ptrs, mask=colmask[None, :], other=0.0)
            else:
                kmask = k0 + rk < K
                a = tl.load(x_ptrs, mask=mmask[:, None] & kmask[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=kmask[:, None] & colmask[None, :], other=0.0)
            acc = tl.dot(a, w, acc)
            x_ptrs += BLOCK_K_D
            w_ptrs += BLOCK_K_D * W4
        g, p = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_K_H2, 2)))    # (BLOCK_M1, BLOCK_K_H2) each: gate-logit, proj
        smsk = chmask[:, None] & smask
        if SAVE_PREACT:
            tl.store(preact_ptr + (2 * ch).to(tl.int64)[:, None] * M + rm[None, :], tl.trans(g).to(et), mask=smsk)
            tl.store(preact_ptr + (2 * ch + 1).to(tl.int64)[:, None] * M + rm[None, :], tl.trans(p).to(et), mask=smsk)
        outl = tl.sigmoid(g) * p
        if pair_mask_ptr is not None:
            outl = outl.to(et).to(tl.float32) * pair_scale[:, None]
        tl.store(left_ptr + ch.to(tl.int64)[:, None] * LL + rm[None, :], tl.trans(outl).to(et), mask=smsk)

    # ---- RIGHT half: weight cols [2*H2 : 4*H2), out -> right_ptr, preact rows [2*H2 : 4*H2) ----
    for c0 in range(0, H2, BLOCK_K_H2):
        ch = c0 + tl.arange(0, BLOCK_K_H2)
        chmask = ch < H2
        cols = 2 * H2 + 2 * c0 + c2
        colmask = (2 * c0 + c2) < 2 * H2
        x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
        w_ptrs = w_ptr + rk[:, None] * W4 + cols[None, :]
        acc = tl.zeros((BLOCK_M1, 2 * BLOCK_K_H2), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K_D):
            if EVEN_K:
                a = tl.load(x_ptrs, mask=mmask[:, None], other=0.0)
                w = tl.load(w_ptrs, mask=colmask[None, :], other=0.0)
            else:
                kmask = k0 + rk < K
                a = tl.load(x_ptrs, mask=mmask[:, None] & kmask[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=kmask[:, None] & colmask[None, :], other=0.0)
            acc = tl.dot(a, w, acc)
            x_ptrs += BLOCK_K_D
            w_ptrs += BLOCK_K_D * W4
        g, p = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_K_H2, 2)))
        smsk = chmask[:, None] & smask
        if SAVE_PREACT:
            tl.store(preact_ptr + (2 * H2 + 2 * ch).to(tl.int64)[:, None] * M + rm[None, :], tl.trans(g).to(et), mask=smsk)
            tl.store(preact_ptr + (2 * H2 + 2 * ch + 1).to(tl.int64)[:, None] * M + rm[None, :], tl.trans(p).to(et), mask=smsk)
        outr = tl.sigmoid(g) * p
        if pair_mask_ptr is not None:
            outr = outr.to(et).to(tl.float32) * pair_scale[:, None]
        tl.store(right_ptr + ch.to(tl.int64)[:, None] * LL + rm[None, :], tl.trans(outr).to(et), mask=smsk)


def bidir_front_triton(x_n, WL, WLg, WR, WRg, *, save_preact=True, pair_mask=None):
    """x_n:(B,L,L,K); WL/WLg/WR/WRg:(K, 2h) x@W form. Returns
    left,right:(B,2h,L,L) bdll and preact:(4*2h, M) interleaved (front_bwd_dW layout).
    ``save_preact=False`` (inference) skips the preact tensor + its stores — the
    backward-only side output cute's forward-only front also omits."""
    # B==1 by design: bdll intermediates put batch OUTSIDE the channel dim. B>1 was implemented
    # (batched grid axis + einsum channel-last contraction) + verified correct, but is SLOWER
    # than looping this B==1 path per batch — the large bdll intermediates (~300 MB at B=8,L=384)
    # thrash L2 (40 MB) when chained, so a per-batch loop (working set ~L2-sized) wins. Loop over
    # B at the caller if needed. See notes/trimul_batch_generalization.
    B, L, L2, K = x_n.shape
    assert B == 1 and L == L2
    H2 = WL.shape[1]                       # per-side hidden = 2*d_hidden
    M = L * L
    x_flat = x_n.reshape(M, K)
    # interleave (gate-logit, proj) columns per side: col 2c=Wg[:,c], 2c+1=W[:,c]
    left_w = torch.stack([WLg, WL], dim=2).reshape(K, 2 * H2)
    right_w = torch.stack([WRg, WR], dim=2).reshape(K, 2 * H2)
    Wlr = torch.cat([left_w, right_w], dim=1).contiguous()      # (K, 4*H2)
    # Cast outside the opaque launch so the compiler can absorb mask construction.
    pair_mask = None if pair_mask is None else pair_mask.to(x_n.dtype).reshape(M).contiguous()
    left, right, preact = _bidir_front_launch(
        x_flat, Wlr, H2, L, save_preact, token_key(L), pair_mask)
    return left, right, (preact if save_preact else None)


def _bidir_front_launch_fake(x_flat, Wlr, H2, L, save_preact, shape_key, pair_mask=None):
    """``left`` and ``right`` as (1, H2, L, L) bdll, plus ``preact`` (4*H2, L*L) interleaved.

    ``preact`` is a 0-element (0, 0) placeholder when ``save_preact`` is False: a schema has one
    fixed return arity and cannot return None, and ``bidir_front_triton`` turns it back into one.
    """
    m = L * L
    return (
        x_flat.new_empty((1, H2, L, L)),
        x_flat.new_empty((1, H2, L, L)),
        x_flat.new_empty((4 * H2, m)) if save_preact else x_flat.new_empty((0, 0)),
    )


@opaque(fake=_bidir_front_launch_fake, name="trimul_bidir_front")
def _bidir_front_launch(
    x_flat: torch.Tensor,
    Wlr: torch.Tensor,
    H2: int,
    L: int,
    save_preact: bool,
    shape_key: int,
    pair_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gated in-projection launch -> ``(left, right, preact)``, left/right in bdll.

    ``preact`` is a 0-element placeholder when ``save_preact`` is False. The eager code passed
    ``left`` as the dummy pointer there, which an op may not do -- returning the same tensor twice
    makes two outputs alias. ``bidir_front_triton`` above turns the placeholder back into ``None``.
    The weight interleave (``stack``/``cat``) stays outside, in the graph.
    """
    m = L * L
    left = torch.empty(1, H2, L, L, device=x_flat.device, dtype=x_flat.dtype)
    right = torch.empty(1, H2, L, L, device=x_flat.device, dtype=x_flat.dtype)
    preact = (torch.empty(4 * H2, m, device=x_flat.device, dtype=x_flat.dtype)
              if save_preact else left)   # dummy ptr when not saving (stores guarded)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)          # noqa: E731
    _bidir_front_kernel[grid](
        x_flat, Wlr, left, right, preact, pair_mask, m, m,
        K=x_flat.shape[1], H2=H2,
        shape_key=pack(shape_key, H2=H2, K=x_flat.shape[1]), SAVE_PREACT=save_preact,
    )
    if not save_preact:
        preact = x_flat.new_empty((0, 0))
    return left, right, preact


# ── merged back-half (mirror of cute BidirBackHalf), fwd + manual bwd ─────────
class _BidirBackHalfTriton(torch.autograd.Function):
    """front → 2 contractions (outgoing [:h] / incoming [h:]) → LN_out+@Wp → gate,
    as ONE Function so the backward matches cute's fused structure (gate dx_n add
    folded into the front dxn GEMM). Weights x@W form; Wp is nn.Linear (N,K) form."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, eps, h, mask,
                residual, dropscale=None):
        B, L, _, D = x_n.shape
        M = B * L * L
        H = 2 * h                                                 # = WL.shape[1]
        left, right, preact = bidir_front_triton(x_n, WL, WLg, WR, WRg, pair_mask=mask)
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        # Mask applies to the contraction inputs (left/right) ONLY — NOT to x_n, so the
        # output gate sigmoid(x_n@Wg) stays unmasked (matches the pytorch reference).
        mm = mask  # Applied inside the front stores; x_n/output gate stay unmasked.
        tri = packed_forward(lf, rf, h)
        view = tri.reshape(H, M).t()                              # (M, H) m-major
        proj, te_xn, mean_out, rstd_out = _te_forward(
            view, ln_out_w, ln_out_b, Wp, None, eps)              # (M, D)
        # fuse the pairformer residual (== module input pair [M,D]) + row-broadcast dropout
        # into the gate store epilogue (same path the cute dispatch uses).
        y, gate = gate_elem_train(x_n.reshape(M, D), proj, Wg, residual, dropscale, seq_len=L)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.h, ctx.mm = eps, h, mm
        ctx.dropscale, ctx.seq_len = dropscale, L
        return y.reshape(B, L, L, D)

    @staticmethod
    def backward(ctx, gy):
        (x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
         preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj) = ctx.saved_tensors
        B, L, _, D = x_n.shape
        M = B * L * L
        h = ctx.h
        H = 2 * h
        gy = gy.reshape(M, D).contiguous()  # guard: autograd may hand a non-contiguous / broadcast (.sum) grad
        # residual grad passes straight through; op-branch grad is drop_row-scaled in gate_elem_bwd_ew
        d_residual = gy.reshape(M, D)

        # ② gate bwd (elementwise; dx_gate folded into the dxn GEMM below); dropout-scale dy
        d_proj, d_glogit = gate_elem_bwd_ew(gy, proj, gate, ctx.dropscale, ctx.seq_len)
        dWg = torch.mm(x_n.reshape(M, D).t(), d_glogit)           # (D, D) cuBLAS

        # ① LN_out + @Wp bwd (te_style)
        view = tri.reshape(H, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        # `_te_backward` writes dx at x's strides and x here is `view` ((1, M)), so d_view is
        # m-major, `.t()` is contiguous and this reshape is a FREE VIEW -- d_tri ALIASES
        # d_view. Deleting the name frees nothing on its own; the storage goes at the `del`
        # below, which names d_tri and both of its slices. (bidir_training_sm100.py's header
        # states the same aliasing; an earlier comment here claimed a copy and was wrong.)
        d_tri = d_view.t().reshape(H, L, L)
        del d_view, d_proj, view

        # contraction bwd (split outgoing/incoming), cuBLAS bmm
        d_left, d_right = packed_backward(d_tri, lf, rf, h)
        d_left = d_left.reshape(B, H, L, L)
        d_right = d_right.reshape(B, H, L, L)
        del d_tri
        # front bwd: d_concat (triton) + dW (cuBLAS) + W_stack; dxn fuses the gate add
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
            d_left, d_right, preact, x_n, WL, WLg, WR, WRg, pair_mask=ctx.mm)
        dx = torch.mm(d_glogit, Wg.t())                          # dx_gate  (M, D)
        dx.addmm_(dconc.t(), W_stack)                            # + dconcᵀ@W_stack (in-place)
        dx_n = dx.reshape(B, L, L, D)
        # trailing Nones: eps, h, mask; then d_residual (fused residual input), dropscale
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None,
                d_residual, None)


@torch.no_grad()
def _bidir_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wp, ln_out_w, ln_out_b, eps, h, mask,
                 residual):
    """Forward-only bidir back-half — the SAME kernel structure as cute's inference
    ``bidirectional_trimul_sm100`` (front → 2 bmm → LN_out+@Wp → gate), but NO
    autograd.Function / saved tensors and NO preact side output. This is why the
    inference path cudagraphs at cute's speed; the merged Function (with its saves)
    is used only under grad."""
    B, L, _, D = x_n.shape
    M = B * L * L
    H = 2 * h
    left, right, _ = bidir_front_triton(
        x_n, WLt, WLgt, WRt, WRgt, save_preact=False, pair_mask=mask)
    lf = left.reshape(H, L, L)
    rf = right.reshape(H, L, L)
    o_out = torch.bmm(lf[:h], rf[:h].transpose(1, 2))            # outgoing
    o_in = torch.bmm(lf[h:].transpose(1, 2), rf[h:])            # incoming
    tri = torch.cat([o_out, o_in], dim=0)                        # (H, L, L)
    # ONE pass: LN_out(H) + proj GEMM (H -> D) + gate GEMM (D -> D) + residual. This used to be
    # `_te_forward` (LN+GEMM) followed by `gate_elem_infer`, because `trimul_back_triton` gated
    # over the same axis it normalised and so refused H != D. It takes the gate's width separately
    # now. Measured on an A6000 at L=1024, d_pair=128: the split pair cost 5.06 ms (2.79 + 2.27)
    # and materialised an (M, D) proj between the two, while this module sat 1.2 ms behind
    # cuequivariance for the whole forward.
    return trimul_back_triton(
        tri.reshape(1, H, L, L), x_n, Wp.t().contiguous(), Wgt,
        ln_out_w, ln_out_b, eps, residual.view(B, L, L, D),
    )


def bidirectional_trimul_triton(
    pair,                        # (B, L, L, d_pair)
    WL, WLg, WR, WRg,            # to_{left,left_gate,right,right_gate}.weight  (2h, d_pair)
    Wg,                          # to_gate.weight   (d_pair, d_pair)
    Wout,                        # to_out.weight    (d_pair, 2h)  (nn.Linear form)
    ln_in_w, ln_in_b,            # (d_pair,)
    ln_out_w, ln_out_b,          # (2h,)
    eps_in, eps_out, d_hidden,
    mask=None,                   # (B, L) residue mask, optional (folded into LN_in like cute)
    dropscale=None,              # drop_row scale [B,1,L,D] (== mask/(1-p)); training only
):
    """Faithful triton mirror of the cute bidir. Returns (B, L, L, d_pair).
    Mirrors cute's dispatch exactly: LN_in (triton, row_scale mask), then — as cute
    does — a forward-only path for inference (``_bidir_infer``) and the merged
    autograd Function for training (``_BidirBackHalfTriton``). All-triton/cuBLAS;
    requires d_hidden == d_pair (the front produces per-side width 2*d_hidden)."""
    d = pair.shape[-1]
    if d_hidden != d:
        raise ValueError(
            f"TRITON bidirectional trimul requires d_hidden == d_pair "
            f"(got d_hidden={d_hidden}, d_pair={d})."
        )
    # Mask applies to the contraction inputs (left/right), NOT folded into LN_in — so
    # the output gate sigmoid(x_n@Wg) stays unmasked (matches the pytorch reference).
    m2d = None
    if mask is not None:
        if mask.dim() == 2:                                     # (B, L) residue mask
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)         # (B, L, L)
        else:                                                   # (B, L, L) pair mask
            m = mask
        m2d = m.to(pair.dtype)
    B, L = pair.shape[0], pair.shape[1]
    M = B * L * L
    # The residual is the ORIGINAL (pre-LN_in) input pair, and is not optional: this op is
    # ``y = pair + drop_row(bidir_trimul(pair))``. dropscale [B,1,L,D] -> [L,D] (B==1) for the
    # gate store's row-broadcast indexing. Both fold into the gate_elem epilogue (no external add).
    residual_flat = pair.reshape(M, d)
    ds_2d = dropscale.reshape(L, d) if dropscale is not None else None
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps_in)
    WLt, WLgt = WL.t().contiguous(), WLg.t().contiguous()
    WRt, WRgt, Wgt = WR.t().contiguous(), WRg.t().contiguous(), Wg.t().contiguous()
    if not torch.is_grad_enabled() and ds_2d is None:
        # INFERENCE: forward-only (no saved tensors) — cudagraphs at cute's speed. Also gate on
        # ds_2d is None: a live dropout scale (train() under no_grad, p_drop>0) must take the
        # TRAINING apply below (which folds dropout into the gate epilogue) — the inference path
        # has no dropscale and would silently skip dropout. Mirrors the cute dispatch's guard.
        return _bidir_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout,
                            ln_out_w, ln_out_b, eps_out, d_hidden, mask=m2d, residual=residual_flat)
    # The training path ALWAYS carries a drop scale, ones when the model's p_drop is 0. That is
    # what lets the training kernel be flagless: `gate_elem_train` and `gate_elem_bwd_ew` apply it
    # unconditionally instead of branching on a keyed constexpr, at a measured 0.2% (the scale is
    # [L, N] and stays in L2). Building it here, at the one place that decides "this is training",
    # keeps that decision in a single spot.
    if ds_2d is None:
        ds_2d = ones_dropscale(L, d, pair)
    # TRAINING: merged autograd Function (weights x@W; autograd flows the transpose).
    return _BidirBackHalfTriton.apply(
        x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout, ln_out_w, ln_out_b, eps_out, d_hidden, m2d,
        residual_flat, ds_2d,
    )
