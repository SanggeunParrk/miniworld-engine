"""Fused front-backward: EW (gated-GEMM bwd) fused INTO the two grad GEMMs so the
(M,4D) d_preact intermediate never hits HBM (dt-v1 materializes it; we don't).

Inputs are channel-major (matches the cute front's bdll output + saved preact):
  d_lr   : (2D, M)  grads of [left|right]            (= d_lr.reshape(2D,M))
  preact : (4D, M)  pre-GLU logits, interleaved [g,p] per col, left then right
  x_n    : (M, D)
  W*     : (D, D)   x@W form
EW (per element):  gL=σ(gLlog); d_pL=dL*gL; d_gLlog=dL*pL*gL(1-gL)  (and R).
Outputs: dx_n (M,D), dWL/dWLg/dWR/dWRg (D,D).
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of
from miniworld_engine.kernels.trimul_inproj.triton._autotune import get_seq_group


# `_dx_kernel` and `_dw_kernel` were removed here. Both were @triton.jit with NO autotune and no
# caller anywhere in src/, benchmarks/ or tests/ -- the only surviving mention was the comment in
# `front_bwd_dW` recording that cuBLAS beats the hand-written split-K wgrad. They also held the
# last untiled shape axis in the kernels: `rd = tl.arange(0, D)` took the whole channel axis in
# one tile (and `_dw_kernel` carried four [D, D] fp32 accumulators, i.e. 4 x D^2 registers).
# Tiling dead code would have added an unexercised code path; deleting it removes the violation
# and the maintenance surface at once.



@triton.autotune(configs=configs_for("trimul_bwd_gate_packed_triton"), key=['GROUP_M', 'D'])
@triton.jit
def _dconcat_kernel(dL_ptr, dR_ptr, preact, out, M, DM, D: tl.constexpr, BLOCK_E: tl.constexpr,
                    GROUP_M):
    """1D channel-major elementwise: out (4D,M) = [d_gLlog; d_pL; d_gRlog; d_pR].
    Iterate over DM=D*M positions (d,m); dL=dL_ptr[idx], dR=dR_ptr[idx] (separate left/right
    buffers — no d_lr cat in the caller); preact is interleaved so needs (2d)*M+m indexing.
    1D (no full-D tile) → D-general, no reg blowup."""
    # int64 offsets: at large d·L (e.g. d_pair=512, L=1024) the max flat offset 4·DM = 2^32
    # overflows int32/uint32 → illegal memory access. Promote idx, M, DM to int64 so every
    # derived offset (k·DM+idx, (2d±)·M+m) is computed in int64.
    Mi = M.to(tl.int64)
    DMi = DM.to(tl.int64)
    idx = tl.program_id(0).to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E).to(tl.int64)
    mask = idx < DMi
    d = idx // Mi
    m = idx - d * Mi
    D2 = 2 * D
    dL = tl.load(dL_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dR = tl.load(dR_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    gLlog = tl.load(preact + (2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pL = tl.load(preact + (2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gRlog = tl.load(preact + (D2 + 2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pR = tl.load(preact + (D2 + 2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gL, gR = tl.sigmoid(gLlog), tl.sigmoid(gRlog)
    et = out.dtype.element_ty
    tl.store(out + idx, (dL * pL * gL * (1 - gL)).to(et), mask=mask)            # d_gLlog
    tl.store(out + DMi + idx, (dL * gL).to(et), mask=mask)                      # d_pL
    tl.store(out + 2 * DMi + idx, (dR * pR * gR * (1 - gR)).to(et), mask=mask)  # d_gRlog
    tl.store(out + 3 * DMi + idx, (dR * gR).to(et), mask=mask)                  # d_pR




@triton.autotune(configs=configs_for("trimul_bwd_gate_transpose_packed_triton"), key=['GROUP_M', 'D'])
@triton.jit
def _dconcat5_kernel(dL_ptr, dR_ptr, preact, dglog_ptr, out, M, DM, D: tl.constexpr,
                     BLOCK_E: tl.constexpr, GROUP_M):
    """Like `_dconcat_kernel` but builds a 5-block d_concat (5D,M):
       [d_gLlog; d_pL; d_gRlog; d_pR; d_glogit].
    Block 4 relayouts the gate input-grad d_glogit (M,D) row-major into channel-major (D,M).
    Folding d_glogit in lets ONE GEMM give all 5 weight grads (incl dWg) AND ONE GEMM give
    dx_n = dconcᵀ@W_all = (front dxn) + (d_glogit@Wgᵀ) — i.e. 6a+6b+dWg collapse. Single-dir
    only (gate width == per-side hidden D)."""
    Mi = M.to(tl.int64)
    DMi = DM.to(tl.int64)
    idx = tl.program_id(0).to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E).to(tl.int64)
    mask = idx < DMi
    d = idx // Mi
    m = idx - d * Mi
    D2 = 2 * D
    dL = tl.load(dL_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dR = tl.load(dR_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    gLlog = tl.load(preact + (2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pL = tl.load(preact + (2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gRlog = tl.load(preact + (D2 + 2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pR = tl.load(preact + (D2 + 2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gL, gR = tl.sigmoid(gLlog), tl.sigmoid(gRlog)
    dglog = tl.load(dglog_ptr + m * D + d, mask=mask, other=0.0).to(tl.float32)  # (M,D)->(D,M)
    et = out.dtype.element_ty
    tl.store(out + idx, (dL * pL * gL * (1 - gL)).to(et), mask=mask)            # d_gLlog
    tl.store(out + DMi + idx, (dL * gL).to(et), mask=mask)                      # d_pL
    tl.store(out + 2 * DMi + idx, (dR * pR * gR * (1 - gR)).to(et), mask=mask)  # d_gRlog
    tl.store(out + 3 * DMi + idx, (dR * gR).to(et), mask=mask)                  # d_pR
    tl.store(out + 4 * DMi + idx, dglog.to(et), mask=mask)                      # d_glogit


def front_bwd_dW_glogit(d_left, d_right, preact, x_n, WL, WLg, WR, WRg, d_glogit, Wg):
    """NEGATIVE RESULT — tried & not adopted (kept as reference). Collapses the single-dir back
    half to TWO cuBLAS GEMMs by folding d_glogit in as the 5th d_concat block:
      dWs5 = dconc5 @ x_n   (5D,D) -> dWLg/dWL/dWRg/dWR + dWg          (all weight grads, 1 GEMM)
      dx_n = dconc5ᵀ @ W_all (M,D) = dconcᵀ@W_stack + d_glogit@Wgᵀ     (6a+6b in 1 GEMM, add free)
    Replaces the old 4 GEMMs (dWg, dWs, dxn_front, addmm). BUT folding the (M,D) d_glogit into the
    channel-major (D,M) 5th block needs a non-coalesced (stride-D) relayout in `_dconcat5_kernel`,
    plus a larger (5D,M) dconc materialization. Measured: small-L gain cancels (relayout cost ≈
    saved launch), large-L regresses ~16% (the relayout/materialization GPU cost dominates when
    GPU-bound). The shipped path fuses ONLY the cheap add via cuBLAS addmm. square single-dir."""
    B, H, L, _ = d_left.shape
    Din = WL.shape[0]
    M = B * L * L
    dt = x_n.dtype
    dL2 = d_left.reshape(H * M)
    dR2 = d_right.reshape(H * M)
    preact2 = preact.reshape(4 * H, M)
    xf = x_n.reshape(M, Din)
    dglog = d_glogit.reshape(M, H)

    dconc5 = torch.empty(5 * H, M, device=x_n.device, dtype=dt)
    DM = H * M
    _dconcat5_kernel[lambda meta: (triton.cdiv(DM, meta["BLOCK_E"]),)](
        dL2, dR2, preact2, dglog, dconc5, M, DM, D=H, GROUP_M=get_seq_group(M))

    dWs = dconc5 @ xf                                            # (5H, Din) cuBLAS huge-K
    dWLg = dWs[:H].t().contiguous()
    dWL = dWs[H:2 * H].t().contiguous()
    dWRg = dWs[2 * H:3 * H].t().contiguous()
    dWR = dWs[3 * H:4 * H].t().contiguous()
    dWg = dWs[4 * H:].t().contiguous()
    W_all = torch.cat([WLg.t(), WL.t(), WRg.t(), WR.t(), Wg.t()], dim=0)   # (5H, Din)
    dxn = dconc5.t() @ W_all                                     # (M, Din) cuBLAS — dxn+gate fused
    return dxn, dWL, dWLg, dWR, dWRg, dWg


def front_bwd_fused(d_left, d_right, preact, x_n, WL, WLg, WR, WRg):
    """d_left/d_right:(B,H,L,L) preact:(B,4H,L,L) x_n:(B,L,L,Din) W*:(Din,H). dx_n + 4 wgrads.

    d_left/d_right are passed SEPARATELY (not cat'd into a (B,2H,L,L) d_lr) — the elementwise
    kernel reads them from two pointers, killing the per-backward (B,2H,L,L) cat copy.

    Dimension-general: the per-side HIDDEN width H is INDEPENDENT of the input width Din
    (from the weights / x_n). They coincide for square trimul (H=Din=D) but NOT for
    bidirectional (H=2·d_hidden, Din=d_pair) — keep them separate or the x_n reshape/dxn break.

    Kernel-optimized path: a light channel-major elementwise (d_concat) feeds TWO cuBLAS
    GEMMs (dW = d_concatᵀ-stacked; dxn). cuBLAS hits far higher efficiency on these
    (esp. the huge-K wgrad reduction) than the hand-written split-K _dw_kernel (~30 TFLOPS)."""
    dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
        d_left, d_right, preact, x_n, WL, WLg, WR, WRg)
    B, _, L, _ = d_left.shape
    Din = WL.shape[0]
    dxn = (dconc.t() @ W_stack).reshape(B, L, L, Din)   # cuBLAS (non-merged path)
    return dxn, dWL, dWLg, dWR, dWRg


def front_bwd_dW(d_left, d_right, preact, x_n, WL, WLg, WR, WRg):
    """The front bwd EXCEPT the final dxn GEMM: builds d_concat (elementwise), the 4 weight
    grads (cuBLAS huge-K, STAYS cuBLAS), and the stacked W operand. Returns
    (dconc (4H,M), dWL, dWLg, dWR, dWRg, W_stack (4H,Din)). The caller forms dxn = dconcᵀ@W_stack
    (and `BidirBackHalf` fuses the gate's dx_gate add into that GEMM in cute)."""
    B, H, L, _ = d_left.shape   # per-side hidden width
    Din = WL.shape[0]           # input width (= d_pair); may differ from H (bidirectional)
    M = B * L * L
    dt = x_n.dtype
    dL2 = d_left.reshape(H * M)
    dR2 = d_right.reshape(H * M)
    preact2 = preact.reshape(4 * H, M)
    xf = x_n.reshape(M, Din)

    dconc = torch.empty(4 * H, M, device=x_n.device, dtype=dt)   # [d_gLlog;d_pL;d_gRlog;d_pR]
    DM = H * M
    _dconcat_kernel[lambda meta: (triton.cdiv(DM, meta["BLOCK_E"]),)](
        dL2, dR2, preact2, dconc, M, DM, D=H, GROUP_M=get_seq_group(M))

    # dW: (4H,M)@(M,Din) — dispatched (huge-K reduction reliably picks cuBLAS; quack 2.6-5x
    # slower there — measured. dispatch confirms + self-documents).
    from miniworld_engine.kernels.trimul_inproj.cute import dispatch
    dWs = dispatch.mm("dWs", dconc, xf)
    dWLg = dWs[:H].t().contiguous()
    dWL = dWs[H:2 * H].t().contiguous()
    dWRg = dWs[2 * H:3 * H].t().contiguous()
    dWR = dWs[3 * H:].t().contiguous()
    W_stack = torch.cat([WLg.t(), WL.t(), WRg.t(), WR.t()], dim=0)   # (4H, Din)
    return dconc, dWL, dWLg, dWR, dWRg, W_stack


# ── σ(gate) backward: reconstruct GLU grads from lr + sg (no preact) ──────────────────────────


@triton.autotune(configs=configs_for("trimul_bwd_gate_packed_recompute_triton"), key=['GROUP_M', 'D'])
@triton.jit
def _dconcat_sig_kernel(dL_ptr, dR_ptr, lrL_ptr, lrR_ptr, sg_ptr, out, M, DM,
                        D: tl.constexpr, BLOCK_E: tl.constexpr, GROUP_M):
    """out (4D,M) = [d_gLlog; d_pL; d_gRlog; d_pR], built from the forward outputs lr (=left,
    right) and sg (=σ(gate), [2D,M]) instead of raw preact logits:
        d_glog = d_out·lr·(1-sg) ;  d_proj = d_out·sg    (proj = lr/sg is never needed).
    lr split as two [D,M] buffers (left/right); sg is [2D,M] (left rows 0:D, right rows D:2D)."""
    Mi = M.to(tl.int64)
    DMi = DM.to(tl.int64)
    idx = tl.program_id(0).to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E).to(tl.int64)
    mask = idx < DMi
    dL = tl.load(dL_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dR = tl.load(dR_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    lrL = tl.load(lrL_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    lrR = tl.load(lrR_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    sgL = tl.load(sg_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    sgR = tl.load(sg_ptr + DMi + idx, mask=mask, other=0.0).to(tl.float32)   # sg[D:2D] rows
    et = out.dtype.element_ty
    tl.store(out + idx, (dL * lrL * (1.0 - sgL)).to(et), mask=mask)            # d_gLlog
    tl.store(out + DMi + idx, (dL * sgL).to(et), mask=mask)                    # d_pL
    tl.store(out + 2 * DMi + idx, (dR * lrR * (1.0 - sgR)).to(et), mask=mask)  # d_gRlog
    tl.store(out + 3 * DMi + idx, (dR * sgR).to(et), mask=mask)                # d_pR


def front_bwd_dW_sig(d_left, d_right, left, right, sg, x_n, WL, WLg, WR, WRg):
    """σ(gate) variant of front_bwd_dW: reconstructs d_concat from the forward outputs
    (left, right, sg=σ(gate)) instead of preact. Same returns/layout as front_bwd_dW
    (dconc (4H,M) = [d_gLlog;d_pL;d_gRlog;d_pR], + the 4 dW and W_stack). dW stays cuBLAS."""
    B, H, L, _ = d_left.shape
    Din = WL.shape[0]
    M = B * L * L
    dt = x_n.dtype
    dL2 = d_left.reshape(H * M)
    dR2 = d_right.reshape(H * M)
    lrL = left.reshape(H * M)
    lrR = right.reshape(H * M)
    sg2 = sg.reshape(2 * H, M)
    xf = x_n.reshape(M, Din)

    dconc = torch.empty(4 * H, M, device=x_n.device, dtype=dt)
    DM = H * M
    _dconcat_sig_kernel[lambda meta: (triton.cdiv(DM, meta["BLOCK_E"]),)](
        dL2, dR2, lrL, lrR, sg2, dconc, M, DM, D=H, GROUP_M=get_seq_group(M))

    from miniworld_engine.kernels.trimul_inproj.cute import dispatch
    dWs = dispatch.mm("dWs", dconc, xf)
    dWLg = dWs[:H].t().contiguous()
    dWL = dWs[H:2 * H].t().contiguous()
    dWRg = dWs[2 * H:3 * H].t().contiguous()
    dWR = dWs[3 * H:].t().contiguous()
    W_stack = torch.cat([WLg.t(), WL.t(), WRg.t(), WR.t()], dim=0)
    return dconc, dWL, dWLg, dWR, dWRg, W_stack
