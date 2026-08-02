"""Single-op left+right+gate front, in Triton — gate gets a PROPER sigmoid.

Three logical dots over x(M,K), K=D=128, B=1, bf16:
  acc_g = x @ [WLg|WRg]   (M,2D)   gate logits for left,right
  acc_p = x @ [WL |WR ]   (M,2D)   projections
  acc_t = x @ Wg          (M, D)   the output-gate logits
  left  = sigmoid(acc_g[:,:D]) * acc_p[:,:D]   -> bdll [B,D,L,L]
  right = sigmoid(acc_g[:,D:]) * acc_p[:,D:]   -> bdll
  gate  = sigmoid(acc_t)                       -> blld [B,L,L,D]

No glu-pair constraint, no zero columns, no bias trick: the gate is a plain
unary sigmoid alongside left/right.

Schedule (sm_100 / B200)
------------------------
Profiling showed this op is NOT tensor-core bound: the GEMM is ~0.014 ms while the
epilogue stores dominate (lr bdll = 512 MB, gate = 256 MB). The original kernel
held one giant fp32 accumulator (BM, 2D+D)=(256,384) in tensor memory, capping
resident blocks so the long store latency could not overlap across blocks
(effective store BW ~0.9 TB/s vs a pure coalesced write's ~6.4 TB/s).

The win came from cutting the live fp32 accumulator so more blocks stay resident:
 * lr is its own kernel on a 2-D grid (M-blocks, 2): the second axis is the
   left/right "half", so each program carries only a (BM, 2D)=(BM,256) accumulator
   (instead of (BM,4D)). Weights for one half are interleaved [g0 p0 g1 p1 ...] so
   one fat GEMM + reshape/split recovers (g,p) locally for sigmoid(g)*p. The bdll
   store transposes to (D,BM) so consecutive rows m are contiguous (addr c*LL+m)
   -> coalesced. This half-accumulator runs at BM=128/num_warps=4 and roughly
   doubles store overlap vs the fused kernel.
 * gate is a separate, lighter kernel with a (BM, D) accumulator and a fully
   contiguous (M,D) store, which reaches ~1.6 TB/s on its own.
Splitting the two epilogues (rather than folding the gate's accumulator into the lr
programs) keeps each kernel at its own occupancy sweet spot; measured faster than
any single fused launch we found. Tradeoff: x is read by both kernels, but the
re-read is far cheaper than the occupancy lost to a wider accumulator. K-loop with
BLOCK_K keeps num_stages live (operand-load / MMA pipelining). I/O contract — the
trimul_front_triton signature, outputs, and bdll/blld layouts — is unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of
from miniworld_engine.kernels.trimul_inproj.triton._autotune import get_seq_group


_trimul_front_lr_prune = make_cache_prune(
    "trimul_front_lr", dtype_of=tensor_dtype_of("x_ptr"),
    bucket_of=key_bucket_of("GROUP_M", "D"),
)


@triton.autotune(
    # lr kernel, plain 1-D grid over M-blocks. Each program computes BOTH halves
    # SEQUENTIALLY, so the live fp32 accumulator is only (BM,2D)=(BM,256) at a time
    # (the left accumulator is freed before the right is built) AND x is read once
    # per block (the left/right loops hit it warm in L2). This beat both the fused
    # (BM,4D) kernel and a 2-axis (M-blocks,2) split: small accumulator => many
    # resident blocks => the transposed bdll store overlaps across blocks, while the
    # single x-read avoids the cold re-read a separate-grid-per-half would pay.
    # Curated for sm_100: BM in {64,128,256}, num_warps in {4,8}, BK<=K so the
    # K-loop pipelines (num_stages live). BM=128/num_warps=4 is the sweet spot.
    configs=[
        triton.Config({"BM": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BK": 64}, num_warps=4, num_stages=3),
    ],
    key=["GROUP_M", "D"],
    prune_configs_by={"early_config_prune": _trimul_front_lr_prune},
)
@triton.jit
def _lr_kernel(
    x_ptr, wlr_ptr,                        # x:(M,K)  wlr:(K,4D)=[Lhalf|Rhalf]
    lr_ptr,                                # lr: bdll planes (2D, LL)
    M, LL,
    K: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    # int64 M-index: bdll store offset is (D+rd)*LL + m with LL=L*L, so at large L
    # (e.g. L>=4096 at D=128, or d_pair=512 sooner) the flat offset exceeds int32.
    # Promote the row index and LL to int64 — mirrors the hardened backward
    # (trimul_inproj/triton/back_fused.py). tl.dot loads stay tile-local.
    rm = pid.to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    rk = tl.arange(0, BK)
    r2d = tl.arange(0, 2 * D)
    rd = tl.arange(0, D)
    LL = LL.to(tl.int64)
    mmask = rm < M

    # ---- left half: wlr cols [0:2D) interleaved [g0 p0 g1 p1 ...] ----
    x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]               # (BM,BK)
    w_ptrs = wlr_ptr + rk[:, None] * (4 * D) + r2d[None, :]      # (BK,2D)
    acc = tl.zeros((BM, 2 * D), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x_ptrs, mask=mmask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(w_ptrs), acc)
        x_ptrs += BK
        w_ptrs += BK * (4 * D)
    g, p = tl.split(tl.reshape(acc, (BM, D, 2)))
    out = tl.sigmoid(g) * p                                      # (BM,D)
    # bdll plane c at addr c*LL + m; transpose to (D,BM) -> coalesced runs over m.
    tl.store(lr_ptr + rd[:, None] * LL + rm[None, :],
             tl.trans(out).to(lr_ptr.dtype.element_ty), mask=rm[None, :] < M)

    # ---- right half: wlr cols [2D:4D) ----
    x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = wlr_ptr + 2 * D + rk[:, None] * (4 * D) + r2d[None, :]
    acc = tl.zeros((BM, 2 * D), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x_ptrs, mask=mmask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(w_ptrs), acc)
        x_ptrs += BK
        w_ptrs += BK * (4 * D)
    g, p = tl.split(tl.reshape(acc, (BM, D, 2)))
    out = tl.sigmoid(g) * p
    tl.store(lr_ptr + (D + rd[:, None]) * LL + rm[None, :],
             tl.trans(out).to(lr_ptr.dtype.element_ty), mask=rm[None, :] < M)


_trimul_front_gate_prune = make_cache_prune(
    "trimul_front_gate", dtype_of=tensor_dtype_of("x_ptr"),
    bucket_of=key_bucket_of("GROUP_M", "D"),
)


@triton.autotune(
    # gate kernel: (BM,D) acc, fully contiguous (M,D) store. BM=64/num_warps=4 wins.
    configs=[
        triton.Config({"BM": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BK": 64}, num_warps=8, num_stages=2),
    ],
    key=["GROUP_M", "D"],
    prune_configs_by={"early_config_prune": _trimul_front_gate_prune},
)
@triton.jit
def _gate_kernel(
    x_ptr, wg_ptr,                         # x:(M,K)  wg:(K,D)
    gate_ptr,                              # gate:(M,D)
    M,
    K: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    # int64 M-index (gate store is m*D+d, M=L*L): matches _lr_kernel hardening.
    rm = pid.to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    rk = tl.arange(0, BK)
    rd = tl.arange(0, D)
    mmask = rm < M

    x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]      # (BM,BK)
    w_ptrs = wg_ptr + rk[:, None] * D + rd[None, :]      # (BK,D)
    acc = tl.zeros((BM, D), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x_ptrs, mask=mmask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, w, acc)
        x_ptrs += BK
        w_ptrs += BK * D
    tl.store(gate_ptr + rm[:, None] * D + rd[None, :],
             tl.sigmoid(acc).to(gate_ptr.dtype.element_ty), mask=mmask[:, None])


def trimul_front_triton(x, WL, WLg, WR, WRg, Wg):
    """x:(B,L,L,D) -> (left_bdll, right_bdll, gate_blld). B=1."""
    # B==1 by design: bdll intermediates put batch OUTSIDE the channel dim, so the view
    # shortcuts here assume one batch. B>1 was implemented (batched grid axis) + verified
    # correct, but is SLOWER than looping this B==1 path per batch — the large bdll
    # intermediates (~300 MB at B=8,L=384) thrash L2 (40 MB) when chained. Loop over B at the
    # caller if you need it. See docs/kernel-optimization/trimul_batch_generalization.
    B, L, L2, D = x.shape
    assert B == 1 and L == L2
    M, LL = L * L, L * L
    x_flat = x.reshape(M, D)
    # Interleave each half's (gate-logit, proj) columns: col 2c = Wxg[:,c], 2c+1 =
    # Wx[:,c], so the kernel's reshape((...,2)).split() recovers (g, p) per half.
    left = torch.stack([WLg, WL], dim=2).reshape(D, 2 * D)   # (D,2D) = [g0 p0 ...]
    right = torch.stack([WRg, WR], dim=2).reshape(D, 2 * D)
    Wlr = torch.cat([left, right], dim=1).contiguous()       # (D,4D)
    lr = torch.empty(B, 2 * D, L, L, device=x.device, dtype=x.dtype)  # bdll storage
    gate = torch.empty(M, D, device=x.device, dtype=x.dtype)          # blld
    lr_grid = lambda meta: (triton.cdiv(M, meta["BM"]),)     # noqa: E731
    g_grid = lambda meta: (triton.cdiv(M, meta["BM"]),)      # noqa: E731
    group_m = get_seq_group(M)
    _lr_kernel[lr_grid](x_flat, Wlr, lr, M, LL, K=D, D=D, GROUP_M=group_m)
    _gate_kernel[g_grid](x_flat, Wg.contiguous(), gate, M, K=D, D=D, GROUP_M=group_m)
    return lr[:, :D], lr[:, D:], gate.view(B, L, L, D)
