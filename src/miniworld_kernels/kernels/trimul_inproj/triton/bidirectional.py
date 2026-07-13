"""Bidirectional trimul in TRITON — a faithful 1:1 mirror of the CUTE bidir
(``cute/bidir_training.py`` ``BidirBackHalf`` + ``bidir_forward``), same algorithm
and same fusion boundaries, ONLY the backend differs (triton/cuBLAS, no quack).

Stage-for-stage the cute bidir is:

    x_n   = triton_layernorm(pair, ...)              # LN_in  (already triton in cute)
    left,right,preact = FRONT(x_n, WL,WLg,WR,WRg)    # gated in-proj, out_hidden=2h, bdll
    o_out = bmm(lf[:h], rf[:h]ᵀ) ;  o_in = bmm(lf[h:]ᵀ, rf[h:])   # 2 triangle contractions
    tri   = cat([o_out, o_in])                       # (2h, L, L)
    proj  = _te_forward(tri_view, ln_out, Wp)        # LN_out + @Wp   (te_style: triton LN + cuBLAS)
    y     = gate_elem(x_n, proj, Wg)                 # sigmoid output-gate  (triton)

and its backward is the merged BackHalf (gate-ew → dWg → te-bwd → contraction-bwd →
front-bwd, dxn fused with the gate add). We reuse the EXACT same helpers cute uses:
``_te_forward/_te_backward`` (layernorm_linear/te_style), ``front_bwd_dW``
(trimul_inproj/triton/back_fused), ``gate_elem_triton/gate_elem_bwd_ew``
(trimul_inproj/triton/gate_elem), ``triton_layernorm``. The two triangle
contractions are ``torch.bmm`` on the BDLL tensors — exactly what cute's
``dispatch.bmm`` is (cuBLAS). The big GEMMs (dWg, dxn) are cuBLAS, as in cute.

The ONLY new kernel here is the FRONT forward: cute's front is a quack gated
M-major GEMM; we write the equivalent in triton, producing left/right in BDLL
(channel-major) AND the interleaved ``preact`` (=[gLlog,pL] per channel, left then
right) that ``front_bwd_dW`` consumes. It reuses the ``front.py`` ``_lr_kernel``
design (half-accumulator, transposed bdll store) generalised to per-side width 2h
and extended to also store ``preact``. B=1, bf16 / fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm_linear.te_style import (
    _te_backward,
    _te_forward,
)
from miniworld_kernels.kernels.trimul_inproj.triton._autotune import get_seq_group
from miniworld_kernels.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_bwd_ew,
    gate_elem_triton,
)


# ── FRONT forward (the one new triton kernel) ────────────────────────────────
# ONE kernel, N-tiled over the 2*H2 output width in BN-channel chunks (contiguous
# [BK, 2*BN] weight slice + reshape/split per chunk). BN is autotuned WITH a device-aware
# smem prune: at small d a BN that spans the whole width in a single chunk survives and is
# picked — identical to the original single wide GEMM — while at large d (where that would
# OOM A100's 163KB) it is pruned and a smaller BN (more chunks, bounded smem) is used. So
# the algorithm degrades gracefully with d instead of needing a separate kernel.
def _front_smem_prune(configs, named_args, **kwargs):  # noqa: ARG001
    """Drop configs whose per-chunk weight tile [BK, 2*BN] (+ x tile) exceeds device smem."""
    import triton as _triton

    try:
        limit = _triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device(),
        )["max_shared_mem"]
    except Exception:  # noqa: BLE001 -- conservative sm100 budget
        limit = 227 * 1024

    # Peak smem is dominated by the [BK, 2*BN] weight tile; measured ~132 B per output col
    # for this gated-GEMM front (the full-width form needed 270336 B at 2*H2=2048, and fit at
    # 2*H2=1024). Estimate by width (+~6% margin) — matches the observed OOM boundary, whereas
    # a naive stages*tile product over-counts ~2x and would drop the still-fitting configs.
    def _smem(c):
        return int(2 * c.kwargs["BN"]) * 140

    kept = [c for c in configs if _smem(c) <= limit]
    return kept or [min(configs, key=_smem)]


@triton.autotune(
    # BN = CHANNELS per chunk. Each chunk loads the CONTIGUOUS interleaved [BK, 2*BN] weight
    # slice (gate+proj for BN channels) in ONE coalesced GEMM, then reshape (BM, BN, 2) + split.
    # BN=512 spans the whole width in a SINGLE chunk at d<=256 (H2<=512) — then this is exactly
    # the original wide GEMM. At d=512 (H2=1024) that single-chunk form OOMs, so the smem prune
    # drops the too-wide configs and autotune falls to BN=256/128 (more, bounded chunks).
    configs=[
        triton.Config({"BM": 64, "BK": 64, "BN": 512}, num_warps=8, num_stages=2),
        triton.Config({"BM": 64, "BK": 32, "BN": 512}, num_warps=8, num_stages=3),
        triton.Config({"BM": 32, "BK": 32, "BN": 512}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BK": 64, "BN": 256}, num_warps=8, num_stages=2),
        triton.Config({"BM": 64, "BK": 32, "BN": 256}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BK": 32, "BN": 128}, num_warps=4, num_stages=3),
        triton.Config({"BM": 32, "BK": 32, "BN": 128}, num_warps=4, num_stages=3),
    ],
    key=["GROUP_M", "H2"],
    prune_configs_by={"early_config_prune": _front_smem_prune},
)
@triton.jit
def _bidir_front_kernel(
    x_ptr, w_ptr,
    left_ptr, right_ptr,
    preact_ptr,
    M, LL,
    K: tl.constexpr, H2: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr,
    SAVE_PREACT: tl.constexpr = True,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    c2 = tl.arange(0, 2 * BN)                # contiguous interleaved (g,p) cols within a chunk
    mmask = rm < M
    smask = rm[None, :] < M
    W4 = 4 * H2
    et = left_ptr.dtype.element_ty

    # ---- LEFT half: weight cols [0 : 2*H2), out -> left_ptr, preact rows [0 : 2*H2) ----
    for c0 in range(0, H2, BN):
        ch = c0 + tl.arange(0, BN)                       # channel indices this chunk
        chmask = ch < H2
        cols = 2 * c0 + c2                               # contiguous cols [2c0 : 2c0+2BN)
        colmask = cols < 2 * H2
        x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
        w_ptrs = w_ptr + rk[:, None] * W4 + cols[None, :]
        acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(x_ptrs, mask=mmask[:, None], other=0.0)
            acc = tl.dot(a, tl.load(w_ptrs, mask=colmask[None, :], other=0.0), acc)
            x_ptrs += BK
            w_ptrs += BK * W4
        g, p = tl.split(tl.reshape(acc, (BM, BN, 2)))    # (BM, BN) each: gate-logit, proj
        smsk = chmask[:, None] & smask
        if SAVE_PREACT:
            tl.store(preact_ptr + (2 * ch)[:, None] * M + rm[None, :], tl.trans(g).to(et), mask=smsk)
            tl.store(preact_ptr + (2 * ch + 1)[:, None] * M + rm[None, :], tl.trans(p).to(et), mask=smsk)
        outl = tl.sigmoid(g) * p
        tl.store(left_ptr + ch[:, None] * LL + rm[None, :], tl.trans(outl).to(et), mask=smsk)

    # ---- RIGHT half: weight cols [2*H2 : 4*H2), out -> right_ptr, preact rows [2*H2 : 4*H2) ----
    for c0 in range(0, H2, BN):
        ch = c0 + tl.arange(0, BN)
        chmask = ch < H2
        cols = 2 * H2 + 2 * c0 + c2
        colmask = (2 * c0 + c2) < 2 * H2
        x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
        w_ptrs = w_ptr + rk[:, None] * W4 + cols[None, :]
        acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(x_ptrs, mask=mmask[:, None], other=0.0)
            acc = tl.dot(a, tl.load(w_ptrs, mask=colmask[None, :], other=0.0), acc)
            x_ptrs += BK
            w_ptrs += BK * W4
        g, p = tl.split(tl.reshape(acc, (BM, BN, 2)))
        smsk = chmask[:, None] & smask
        if SAVE_PREACT:
            tl.store(preact_ptr + (2 * H2 + 2 * ch)[:, None] * M + rm[None, :], tl.trans(g).to(et), mask=smsk)
            tl.store(preact_ptr + (2 * H2 + 2 * ch + 1)[:, None] * M + rm[None, :], tl.trans(p).to(et), mask=smsk)
        outr = tl.sigmoid(g) * p
        tl.store(right_ptr + ch[:, None] * LL + rm[None, :], tl.trans(outr).to(et), mask=smsk)


def bidir_front_triton(x_n, WL, WLg, WR, WRg, *, save_preact=True):
    """x_n:(B,L,L,K); WL/WLg/WR/WRg:(K, 2h) x@W form. Returns
    left,right:(B,2h,L,L) bdll and preact:(4*2h, M) interleaved (front_bwd_dW layout).
    ``save_preact=False`` (inference) skips the preact tensor + its stores — the
    backward-only side output cute's forward-only front also omits."""
    B, L, L2, K = x_n.shape
    assert B == 1 and L == L2
    H2 = WL.shape[1]                       # per-side hidden = 2*d_hidden
    M = L * L
    x_flat = x_n.reshape(M, K)
    # interleave (gate-logit, proj) columns per side: col 2c=Wg[:,c], 2c+1=W[:,c]
    left_w = torch.stack([WLg, WL], dim=2).reshape(K, 2 * H2)
    right_w = torch.stack([WRg, WR], dim=2).reshape(K, 2 * H2)
    Wlr = torch.cat([left_w, right_w], dim=1).contiguous()      # (K, 4*H2)
    left = torch.empty(B, H2, L, L, device=x_n.device, dtype=x_n.dtype)
    right = torch.empty(B, H2, L, L, device=x_n.device, dtype=x_n.dtype)
    preact = (torch.empty(4 * H2, M, device=x_n.device, dtype=x_n.dtype)
              if save_preact else left)   # dummy ptr when not saving (stores guarded)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)          # noqa: E731
    _bidir_front_kernel[grid](
        x_flat, Wlr, left, right, preact, M, M,
        K=K, H2=H2, GROUP_M=get_seq_group(M), SAVE_PREACT=save_preact,
    )
    return left, right, (preact if save_preact else None)


# ── merged back-half (mirror of cute BidirBackHalf), fwd + manual bwd ─────────
class _BidirBackHalfTriton(torch.autograd.Function):
    """front → 2 contractions (outgoing [:h] / incoming [h:]) → LN_out+@Wp → gate,
    as ONE Function so the backward matches cute's fused structure (gate dx_n add
    folded into the front dxn GEMM). Weights x@W form; Wp is nn.Linear (N,K) form."""

    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, eps, h):
        B, L, _, D = x_n.shape
        M = B * L * L
        H = 2 * h                                                 # = WL.shape[1]
        left, right, preact = bidir_front_triton(x_n, WL, WLg, WR, WRg)
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        o_out = torch.bmm(lf[:h], rf[:h].transpose(1, 2))         # outgoing  lo @ roᵀ
        o_in = torch.bmm(lf[h:].transpose(1, 2), rf[h:])          # incoming  liᵀ @ ri
        tri = torch.cat([o_out, o_in], dim=0)                     # (H, L, L)
        view = tri.reshape(H, M).t()                              # (M, H) m-major
        proj, te_xn, mean_out, rstd_out = _te_forward(
            view, ln_out_w, ln_out_b, Wp, None, eps)              # (M, D)
        y, gate = gate_elem_triton(x_n.reshape(M, D), proj, Wg, return_gate=True)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.h = eps, h
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

        # ② gate bwd (elementwise; dx_gate folded into the dxn GEMM below)
        d_proj, d_glogit = gate_elem_bwd_ew(gy, proj, gate)
        dWg = torch.mm(x_n.reshape(M, D).t(), d_glogit)           # (D, D) cuBLAS

        # ① LN_out + @Wp bwd (te_style)
        view = tri.reshape(H, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        d_tri = d_view.t().reshape(H, L, L)

        # contraction bwd (split outgoing/incoming), cuBLAS bmm
        d_o_out, d_o_in = d_tri[:h], d_tri[h:]
        lo, ro, li, ri = lf[:h], rf[:h], lf[h:], rf[h:]
        d_lo = torch.bmm(d_o_out, ro)                             # outgoing O=lo@roᵀ
        d_ro = torch.bmm(d_o_out.transpose(1, 2), lo)
        d_li = torch.bmm(ri, d_o_in.transpose(1, 2))             # incoming O=liᵀ@ri
        d_ri = torch.bmm(li, d_o_in)
        d_left = torch.cat([d_lo, d_li], dim=0).reshape(B, H, L, L)
        d_right = torch.cat([d_ro, d_ri], dim=0).reshape(B, H, L, L)

        # front bwd: d_concat (triton) + dW (cuBLAS) + W_stack; dxn fuses the gate add
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
            d_left, d_right, preact, x_n, WL, WLg, WR, WRg)
        dx = torch.mm(d_glogit, Wg.t())                          # dx_gate  (M, D)
        dx.addmm_(dconc.t(), W_stack)                            # + dconcᵀ@W_stack (in-place)
        dx_n = dx.reshape(B, L, L, D)
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None)


@torch.no_grad()
def _bidir_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wp, ln_out_w, ln_out_b, eps, h):
    """Forward-only bidir back-half — the SAME kernel structure as cute's inference
    ``bidirectional_trimul_sm100`` (front → 2 bmm → LN_out+@Wp → gate), but NO
    autograd.Function / saved tensors and NO preact side output. This is why the
    inference path cudagraphs at cute's speed; the merged Function (with its saves)
    is used only under grad."""
    B, L, _, D = x_n.shape
    M = B * L * L
    H = 2 * h
    left, right, _ = bidir_front_triton(x_n, WLt, WLgt, WRt, WRgt, save_preact=False)
    lf = left.reshape(H, L, L)
    rf = right.reshape(H, L, L)
    o_out = torch.bmm(lf[:h], rf[:h].transpose(1, 2))            # outgoing
    o_in = torch.bmm(lf[h:].transpose(1, 2), rf[h:])            # incoming
    tri = torch.cat([o_out, o_in], dim=0)                        # (H, L, L)
    proj = _te_forward(tri.reshape(H, M).t(), ln_out_w, ln_out_b, Wp, None, eps)[0]
    y = gate_elem_triton(x_n.reshape(M, D), proj, Wgt)           # gate GEMM + sigmoid·mul
    return y.view(B, L, L, D)


def bidirectional_trimul_triton(
    pair,                        # (B, L, L, d_pair)
    WL, WLg, WR, WRg,            # to_{left,left_gate,right,right_gate}.weight  (2h, d_pair)
    Wg,                          # to_gate.weight   (d_pair, d_pair)
    Wout,                        # to_out.weight    (d_pair, 2h)  (nn.Linear form)
    ln_in_w, ln_in_b,            # (d_pair,)
    ln_out_w, ln_out_b,          # (2h,)
    eps_in, eps_out, d_hidden,
    mask=None,                   # (B, L) residue mask, optional (folded into LN_in like cute)
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
    row_scale = None
    if mask is not None:
        m = mask.unsqueeze(-1) & mask.unsqueeze(-2)              # (B, L, L)
        row_scale = m.reshape(-1).to(pair.dtype)                # (M,)  folded into LN_in
    x_n = triton_layernorm(pair, ln_in_w, ln_in_b, eps_in, row_scale=row_scale)
    WLt, WLgt = WL.t().contiguous(), WLg.t().contiguous()
    WRt, WRgt, Wgt = WR.t().contiguous(), WRg.t().contiguous(), Wg.t().contiguous()
    if not torch.is_grad_enabled():
        # INFERENCE: forward-only (no saved tensors) — cudagraphs at cute's speed.
        return _bidir_infer(x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout,
                            ln_out_w, ln_out_b, eps_out, d_hidden)
    # TRAINING: merged autograd Function (weights x@W; autograd flows the transpose).
    return _BidirBackHalfTriton.apply(
        x_n, WLt, WLgt, WRt, WRgt, Wgt, Wout, ln_out_w, ln_out_b, eps_out, d_hidden,
    )
