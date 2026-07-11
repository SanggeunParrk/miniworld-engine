"""Fused LayerNorm + SwiGLU-expand forward for the Transition module.

The Transition op is

    x   = LayerNorm(x)                       # ln_in (affine, eps)
    a   = x @ Wa^T,  b = x @ Wb^T            # expand_a / expand_b  (N, d)->(N, n*d)
    h   = a * sigmoid(a) * b                 # SwiGLU gate -> (M, n*d)
    out = h @ Ws^T                           # squeeze  (n*d -> d)

This module fuses the **front half** (LN + both expand GEMMs + SwiGLU gate) into a
single Triton kernel and leaves the ``squeeze`` as a plain ``torch.matmul`` (a clean,
well-tuned (M, n*d) x (n*d, d) GEMM). Fusing squeeze would force the expand kernel to
hold a full ``n*d``-wide row per block (no N-tiling) and blow up the accumulator register
budget, so it is deliberately left out.

Following the M1 LayerNormLinear design, the LN row statistics are computed in a
SEPARATE pass (``stats_triton`` -> rstd[m], c1[m]=mean*rstd) so the expand kernel does no
reduction and is free to tile BLOCK_K < d. The fused kernel then loads x once per
M-block, normalizes on-chip (``x*rstd - c1`` then affine), and reuses the one normalized
tile for BOTH the A and B projections across the N-loop.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.kernels.layernorm_linear.triton.stats import stats_triton

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "transition"
USE_SAVEDXN_SPLIT_BWD = os.getenv("TRANSITION_SAVEDXN_SPLIT_BWD", "0") == "1"


def _cuda_b2b_train_enabled() -> bool:
    """Use the fast inference b2b CUDA kernel for the TRAINING forward too (Version A / save_xn=False:
    saves no xn, backward recomputes it). Same env toggle as the inference dispatch."""
    return os.getenv("MINIWORLD_TRANSITION_CUDA_B2B", "1").strip().lower() not in {"0", "false", "off"}
_TRANSITION_LNBWD_PRIVATIZE_REPLICAS = 64


def _transition_fuse_stats_enabled() -> bool:
    return os.getenv("MINIWORLD_TRANSITION_FUSE_STATS", "").strip().lower() in {"1", "true", "on"}


def get_seq_group(length: int) -> int:
    """Bucket flattened LxL rows so autotune follows the L sweep without per-M noise."""
    group_lengths = [
        32 * 32,
        64 * 64,
        128 * 128,
        256 * 256,
        384 * 384,
        48 * 128,
        48 * 256,
        48 * 384,
        48 * 512,
        48 * 1024,
        48 * 2048,
        48 * 3072,
        48 * 4096,
    ]
    for group_idx, group_length in enumerate(sorted(group_lengths)):
        if length <= group_length:
            return group_idx
    return len(group_lengths) - 1


if AUTOTUNE:
    _configs = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bm in (16, 32, 64, 128)
        for bn in (64, 128, 256, 512)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ]
else:
    _configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    ]


def _expand_early_prune(configs, named_args, **kwargs):
    """Drop configs whose shared-memory footprint exceeds the device limit BEFORE they are
    compiled/benched. The expand kernel stages wa+wb tiles ``[BLOCK_K, BLOCK_N]`` (BLOCK_K =
    next_pow2(K); K is NOT tiled) across ``num_stages``, plus the single ``[BLOCK_M, BLOCK_K]``
    xn tile, all bf16. Autotune's bench-time OutOfResources pruning is unsafe under CUDA-graph
    capture -- the failing bench fires DURING capture and fatally poisons the stream (seen at
    d>=256: a 430KB/856KB config OOMs mid-capture). Pruning up front keeps only capturable
    configs so the fused path (and its sm100 gatebwd backward) is usable at d=256 under cudagraph."""
    import triton as _triton

    K = named_args["K"] if "K" in named_args else kwargs.get("K")
    if K is None:
        return list(configs)
    bk = _triton.next_power_of_2(int(K))
    try:
        limit = _triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device(),
        )["max_shared_mem"]
    except Exception:  # noqa: BLE001 -- fall back to a conservative sm100 smem budget
        limit = 227 * 1024

    def _smem(c):
        bm = c.kwargs["BLOCK_M"]
        bn = c.kwargs["BLOCK_N"]
        # num_stages copies of the pipelined wa+wb tiles + the single xn tile (bf16 = 2 B)
        return c.num_stages * 2 * bk * bn * 2 + bm * bk * 2

    kept = [c for c in configs if _smem(c) <= limit]
    return kept or [min(configs, key=_smem)]


# fmt: off
@triton.autotune(
    configs=_configs,
    key=["GROUP_M", "ND", "K", "SAVE_XN"],
    prune_configs_by={"early_config_prune": _expand_early_prune},
)
@triton.jit
def _transition_expand_gate_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, out_ptr, xn_ptr,
    M, ND, K, GROUP_M: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,   # Wa, Wb share layout: (ND, K) row-major -> stride_wn=K, stride_wk=1
    stride_om, stride_on,
    stride_nm, stride_nk,   # xn out: (M, K) row-major (only used when SAVE_XN)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SAVE_XN: tl.constexpr,
):
    # One program owns BLOCK_M rows and ALL of ND: LayerNorm is applied ONCE per row and
    # the normalized tile is reused for both the A and B projections across the N-loop.
    pid_m = tl.program_id(0).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    k_mask = k < K

    # --- normalize once (stats precomputed): xn = (x*rstd - c1) * g + beta ---
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
        mask=row_mask[:, None] & k_mask[None, :], other=0.0,
    ).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
    g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    xhat = x * rstd[:, None] - c1[:, None]
    xn = (xhat * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)  # (BM, BK)

    # --- Version B: stash the (single) normalized tile so backward can reuse it ---
    if SAVE_XN:
        tl.store(
            xn_ptr + rows[:, None] * stride_nm + k[None, :] * stride_nk,
            xn, mask=row_mask[:, None] & k_mask[None, :],
        )

    # --- loop the two projections over N-tiles, reusing the one normalized X tile ---
    for n0 in range(0, ND, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < ND
        wa = tl.load(  # (BLOCK_K, BLOCK_N): w[k, n] = W[cols[n], k]
            wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        wb = tl.load(
            wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        a = tl.dot(xn, wa, out_dtype=tl.float32)
        b = tl.dot(xn, wb, out_dtype=tl.float32)
        gate = a * tl.sigmoid(a) * b
        tl.store(
            out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
            gate.to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )
# fmt: on


def transition_expand_gate(
    x2: torch.Tensor,         # (M, K) contiguous
    ln_weight: torch.Tensor,  # (K,)
    ln_bias: torch.Tensor,    # (K,)
    wa: torch.Tensor,         # (ND, K) = expand_a.weight
    wb: torch.Tensor,         # (ND, K) = expand_b.weight
    eps: float,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,  # (rstd, c1) precomputed
    save_xn: bool = False,
):
    """LayerNorm(x) then SwiGLU(expand_a, expand_b) -> expand (M, ND). Stats fused-out.

    ``save_xn`` (Version B): also emit the normalized x (M, K) via a single in-kernel store
    so the backward can reuse it instead of recomputing. Returns ``(expand, xn)`` then; else
    just ``expand``.
    """
    M, K = x2.shape
    ND = wa.shape[0]
    assert wa.shape[1] == K and wb.shape == wa.shape
    assert K <= 1024, "fused expand assumes K fits one BLOCK_K (next_pow2(K) <= 1024)"

    rstd, c1 = stats if stats is not None else stats_triton(x2, eps)
    expand = torch.empty(M, ND, device=x2.device, dtype=x2.dtype)
    xn = torch.empty(M, K, device=x2.device, dtype=x2.dtype) if save_xn else expand
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731 (N looped in-kernel)
    _transition_expand_gate_kernel[grid](
        x2, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), expand, xn,
        M, ND, K, get_seq_group(M),
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        expand.stride(0), expand.stride(1),
        xn.stride(0), xn.stride(1),
        BLOCK_K=triton.next_power_of_2(K),
        SAVE_XN=save_xn,
    )
    return (expand, xn) if save_xn else expand


# Single baked winner (BLOCK_M=64, BLOCK_N=64) for the d=128 b2b path. NOT env-gated:
# multi-config autotune was timing-UNSTABLE here (cached bad configs -> 0.49-0.64ms runs);
# the single baked config is stable at ~0.31ms. (Unlike the expand kernel, which autotunes.)
_b2b_configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


# fmt: off
@triton.autotune(configs=_b2b_configs, key=["GROUP_M", "ND", "K", "D", "SAVE_XN", "FUSE_STATS"])
@triton.jit
def _transition_b2b_kernel(
    x_ptr, rstd_ptr, c1_ptr, rstd_out_ptr, c1_out_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, ws_ptr, out_ptr, xn_ptr,
    M, ND, K: tl.constexpr, D: tl.constexpr, GROUP_M: tl.constexpr, EPS: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_sd, stride_sn,    # Ws: (D, ND) row-major
    stride_om, stride_od,
    stride_nm, stride_nk,    # xn out: (M, K) row-major (only used when SAVE_XN)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SAVE_XN: tl.constexpr, FUSE_STATS: tl.constexpr,
):
    # Back-to-back: one program owns BLOCK_M rows and ALL of ND. It builds the gated h
    # tile-by-tile and ACCUMULATES the squeeze out[BM, D] += h_chunk @ Ws[:, chunk]^T, so
    # the (M, ND) intermediate h never touches HBM. Only valid when K fits one BLOCK_K.
    pid_m = tl.program_id(0).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    k_mask = k < K

    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
        mask=row_mask[:, None] & k_mask[None, :], other=0.0,
    ).to(tl.float32)
    if FUSE_STATS:
        inv_k = 1.0 / K
        mean = tl.sum(x, axis=1) * inv_k
        x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(x_centered * x_centered, axis=1) * inv_k
        rstd = tl.rsqrt(var + EPS)
        c1 = mean * rstd
        tl.store(rstd_out_ptr + rows, rstd, mask=row_mask)
        tl.store(c1_out_ptr + rows, c1, mask=row_mask)
    else:
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
    g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)

    # --- Version B: stash the (single) normalized tile for backward reuse ---
    if SAVE_XN:
        tl.store(
            xn_ptr + rows[:, None] * stride_nm + k[None, :] * stride_nk,
            xn, mask=row_mask[:, None] & k_mask[None, :],
        )

    dcols = tl.arange(0, D)
    out_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < ND
        wa = tl.load(
            wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        wb = tl.load(
            wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        a = tl.dot(xn, wa, out_dtype=tl.float32)
        b = tl.dot(xn, wb, out_dtype=tl.float32)
        h = (a * tl.sigmoid(a) * b).to(x_ptr.dtype.element_ty)  # (BM, BN)
        ws_t = tl.load(  # (BN, D): Ws[d, cols]^T
            ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
            mask=col_mask[:, None], other=0.0,
        )
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32)
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        out_acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None],
    )
# fmt: on


def transition_b2b(
    x2: torch.Tensor,         # (M, K) contiguous
    ln_weight: torch.Tensor,  # (K,)
    ln_bias: torch.Tensor,    # (K,)
    wa: torch.Tensor,         # (ND, K)
    wb: torch.Tensor,         # (ND, K)
    ws: torch.Tensor,         # (D, ND) = squeeze.weight
    eps: float,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,  # (rstd, c1) precomputed
    save_xn: bool = False,
    fuse_stats: bool | None = None,
):
    """Fully fused LN + SwiGLU expand + squeeze -> out (M, D). h never hits HBM.

    Requires K to fit one BLOCK_K (K = next_pow2(K) <= 1024) AND the (x row + weight tiles)
    working set to fit smem — practical only for small K (d <= 128). Caller falls back to
    ``transition_expand_gate`` + ``torch.matmul`` for larger K. ``stats`` lets the caller
    pass precomputed (rstd, c1) so the backward can reuse the same LN statistics.

    ``save_xn`` (Version B): also emit the normalized x (M, K) via a single in-kernel store
    for backward reuse. Returns ``(out, xn)`` then; else just ``out``.

    With ``fuse_stats=True`` (or ``MINIWORLD_TRANSITION_FUSE_STATS=1/true/on``), the
    kernel computes and stores the LayerNorm stats on-chip. Then returns
    ``(out, rstd, c1, xn)`` when ``save_xn`` else ``(out, rstd, c1)``.
    """
    M, K = x2.shape
    ND = wa.shape[0]
    D = ws.shape[0]
    fuse_stats = _transition_fuse_stats_enabled() if fuse_stats is None else fuse_stats
    if fuse_stats:
        if stats is not None:
            raise ValueError("transition_b2b(fuse_stats=True) computes stats in-kernel; do not pass stats")
        rstd = torch.empty(M, device=x2.device, dtype=torch.float32)
        c1 = torch.empty(M, device=x2.device, dtype=torch.float32)
    else:
        rstd, c1 = stats if stats is not None else stats_triton(x2, eps)
    out = torch.empty(M, D, device=x2.device, dtype=x2.dtype)
    xn = torch.empty(M, K, device=x2.device, dtype=x2.dtype) if save_xn else out
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _transition_b2b_kernel[grid](
        x2, rstd, c1, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), ws.contiguous(), out, xn,
        M, ND, K, D, get_seq_group(M), eps,
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        ws.stride(0), ws.stride(1),
        out.stride(0), out.stride(1),
        xn.stride(0), xn.stride(1),
        BLOCK_K=triton.next_power_of_2(K),
        SAVE_XN=save_xn,
        FUSE_STATS=fuse_stats,
    )
    if fuse_stats:
        return (out, rstd, c1, xn) if save_xn else (out, rstd, c1)
    return (out, xn) if save_xn else out


# ---------------------------------------------------------------------------
# K-TILED back-to-back (bounded smem -> runs at ANY d; scales with K).  [UNVERIFIED:
# written while GPU was unavailable; must be cos-checked + benched on H100 before use.]
#
# The full-K-row b2b above loads BLOCK_K = next_pow2(K), so its weight tiles are [K, BLOCK_N]
# and overflow smem at d>=256. This variant tiles K (inner k-loop): weight tiles are
# [BLOCK_K, BLOCK_N], BOUNDED regardless of d -> no OOM, and the k-loop pipelines (good
# large-K scaling). The squeeze fusion is unchanged: one program owns BLOCK_M rows x ALL of
# n*d, accumulates out_acc[BLOCK_M, D] across the N-chunk loop, writes once. Separate stats
# (rstd, c1 precomputed) let us normalize each k-tile without the full row.
# ---------------------------------------------------------------------------
if AUTOTUNE:
    _b2b_kt_configs = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for bk in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ]
else:
    _b2b_kt_configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ]


# fmt: off
@triton.autotune(configs=_b2b_kt_configs, key=["GROUP_M", "ND", "K", "D"])
@triton.jit
def _transition_b2b_ktiled_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, ws_ptr, out_ptr,
    M, ND, K, D: tl.constexpr, GROUP_M: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_sd, stride_sn,    # Ws: (D, ND) row-major
    stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program owns BLOCK_M rows and ALL of ND. Inner k-loop keeps weight tiles
    # [BLOCK_K, BLOCK_N] (bounded smem at any d); squeeze accumulated in out_acc across the
    # N-chunk loop; h never leaves regs. No atomics.
    pid_m = tl.program_id(0).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
    dcols = tl.arange(0, D)
    out_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < ND
        a_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < K
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
            beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
            xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)
            wa = tl.load(
                wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            wb = tl.load(
                wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            a_acc += tl.dot(xn, wa, out_dtype=tl.float32)
            b_acc += tl.dot(xn, wb, out_dtype=tl.float32)
        h = (a_acc * tl.sigmoid(a_acc) * b_acc).to(x_ptr.dtype.element_ty)  # (BM, BN)
        ws_t = tl.load(
            ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
            mask=col_mask[:, None], other=0.0,
        )  # (BN, D)
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32)
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        out_acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None],
    )
# fmt: on


def transition_b2b_ktiled(
    x2: torch.Tensor,         # (M, K) contiguous
    ln_weight: torch.Tensor,  # (K,)
    ln_bias: torch.Tensor,    # (K,)
    wa: torch.Tensor,         # (ND, K)
    wb: torch.Tensor,         # (ND, K)
    ws: torch.Tensor,         # (D, ND)
    eps: float,
) -> torch.Tensor:
    """K-tiled fully-fused LN + SwiGLU expand + squeeze -> out (M, D), h off HBM, any d.

    UNVERIFIED until cos-checked + benched on H100. Bounded smem (weight tiles
    [BLOCK_K, BLOCK_N]) -> no OOM at d>=256, unlike ``transition_b2b``.
    """
    M, K = x2.shape
    ND = wa.shape[0]
    D = ws.shape[0]
    rstd, c1 = stats_triton(x2, eps)
    out = torch.empty(M, D, device=x2.device, dtype=x2.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _transition_b2b_ktiled_kernel[grid](
        x2, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), ws.contiguous(), out,
        M, ND, K, D, get_seq_group(M),
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        ws.stride(0), ws.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# Back-to-back squeeze fusion fits smem only for small K (the model's d=128). Above this,
# the full-K-row load + weight tiles overflow shared memory, so fall back to the two-step
# path (expand kernel writes h, then a cuBLAS squeeze). The K-tiled variant above lifts this
# limit once verified.
_B2B_MAX_K = 128


# ---------------------------------------------------------------------------
# SEPARATE backward kernels. The expand recompute (a=xn@Wa, b=xn@Wb) is done ONCE in a
# single fused kernel that emits h (for dWs), dA and dB (the old path recomputed a,b TWICE
# — transition_fwd_kernel for h, transition_bwd_kernel for dA/dB). Plus the LayerNorm
# backward kernel. GEMMs (grad_expand, dWs, dWa, dWb, d_xn) stay cuBLAS.
# ---------------------------------------------------------------------------
if AUTOTUNE:
    _egb_configs = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for bk in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ]
else:
    _egb_configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ]


# fmt: off
@triton.autotune(
    configs=_egb_configs,
    key=["GROUP_M", "ND", "K", "NORMALIZE", "STORE_H", "STACK_DAB"],
)
@triton.jit
def _transition_expand_gatebwd_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr, wa_ptr, wb_ptr, ge_ptr,
    h_ptr, dA_ptr, dB_ptr, dAB_ptr, xn_ptr,
    M, ND, K, GROUP_M: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_gm, stride_gn,    # grad_expand / h / dA / dB: (M, ND) row-major
    stride_abm, stride_abn,  # dAB: (M, 2*ND) row-major, [dA | dB]
    stride_nm, stride_nk,    # xn out: (M, K) row-major
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NORMALIZE: tl.constexpr, STORE_H: tl.constexpr, STACK_DAB: tl.constexpr,
):
    # Recompute a=xn@Wa, b=xn@Wb ONCE (tile M,ND; loop K). Emits:
    #   h  = silu(a)*b                  (for dWs = go^T @ h)
    #   dA = grad_expand * b * silu'(a) ;  dB = grad_expand * silu(a)   (SwiGLU gate bwd)
    #
    # Two modes:
    #   NORMALIZE=True  (Version A): x_ptr is the RAW x; normalize inline from saved stats
    #     (xn=(x*rstd-c1)*g+beta) and emit xn (written once by the pid_n==0 CTAs) for the
    #     wgrad GEMMs. No separate normalize pass, no saved xn needed.
    #   NORMALIZE=False (Version B): x_ptr is ALREADY the saved xn; load it directly, skip
    #     the normalize math AND the xn emit (the caller already holds xn).
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < M
    cmask = cols < ND
    if NORMALIZE:
        rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
        c1 = tl.load(c1_ptr + rows, mask=rmask, other=0.0)
    et = h_ptr.dtype.element_ty
    a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        xkmask = rmask[:, None] & k_mask[None, :]
        x = tl.load(x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=xkmask, other=0.0)
        if NORMALIZE:
            xf = x.to(tl.float32)
            g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
            beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
            xn = ((xf * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(et)
            if pid_n == 0:
                tl.store(xn_ptr + rows[:, None] * stride_nm + k[None, :] * stride_nk, xn, mask=xkmask)
        else:
            xn = x  # x_ptr already holds the saved normalized x
        wa = tl.load(wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                     mask=k_mask[:, None] & cmask[None, :], other=0.0)
        wb = tl.load(wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                     mask=k_mask[:, None] & cmask[None, :], other=0.0)
        a += tl.dot(xn, wa, out_dtype=tl.float32)
        b += tl.dot(xn, wb, out_dtype=tl.float32)
    sig = tl.sigmoid(a)
    silu = a * sig
    goff = rows[:, None] * stride_gm + cols[None, :] * stride_gn
    gmask = rmask[:, None] & cmask[None, :]
    ge = tl.load(ge_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
    if STORE_H:
        tl.store(h_ptr + goff, (silu * b).to(et), mask=gmask)
    dA = (ge * b * (sig + silu * (1.0 - sig))).to(et)
    dB = (ge * silu).to(et)
    if STACK_DAB:
        tl.store(
            dAB_ptr + rows[:, None] * stride_abm + cols[None, :] * stride_abn,
            dA,
            mask=gmask,
        )
        tl.store(
            dAB_ptr + rows[:, None] * stride_abm + (cols[None, :] + ND) * stride_abn,
            dB,
            mask=gmask,
        )
    else:
        tl.store(dA_ptr + goff, dA, mask=gmask)
        tl.store(dB_ptr + goff, dB, mask=gmask)
# fmt: on


def _transition_expand_gatebwd(x2, rstd, c1, gamma, beta, wa, wb, grad_expand):
    """Version A: normalize x inline + recompute a,b once -> (h, dA, dB, xn)."""
    M, K = x2.shape
    ND = wa.shape[0]
    h = torch.empty_like(grad_expand)
    dA = torch.empty_like(grad_expand)
    dB = torch.empty_like(grad_expand)
    xn = torch.empty_like(x2)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(ND, meta["BLOCK_N"]))  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        x2, rstd, c1, gamma.contiguous(), beta.contiguous(),
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dA, dB, dA, xn,
        M, ND, K, get_seq_group(M),
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        grad_expand.stride(0), grad_expand.stride(1),
        dA.stride(0), dA.stride(1),
        xn.stride(0), xn.stride(1),
        NORMALIZE=True,
        STORE_H=True,
        STACK_DAB=False,
    )
    return h, dA, dB, xn


def _transition_expand_gatebwd_stacked(x2, rstd, c1, gamma, beta, wa, wb, grad_expand):
    """Version A stacked: normalize x inline + recompute a,b once -> (h, dAB=[dA|dB], xn).

    Same kernel as the split Version A but with STACK_DAB=True, so the downstream weight/input
    grads use the SAME two larger GEMMs (dWab, dAB@w_ab) as Version B — only the gate-bwd stage
    differs between the versions (recompute xn here vs reuse saved xn).
    """
    M, K = x2.shape
    ND = wa.shape[0]
    h = torch.empty_like(grad_expand)
    dAB = torch.empty(M, ND * 2, device=x2.device, dtype=grad_expand.dtype)
    xn = torch.empty_like(x2)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(ND, meta["BLOCK_N"]))  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        x2, rstd, c1, gamma.contiguous(), beta.contiguous(),
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dAB, dAB, dAB, xn,
        M, ND, K, get_seq_group(M),
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        grad_expand.stride(0), grad_expand.stride(1),
        dAB.stride(0), dAB.stride(1),
        xn.stride(0), xn.stride(1),
        NORMALIZE=True,
        STORE_H=True,
        STACK_DAB=True,
    )
    return h, dAB, xn


def _transition_expand_gatebwd_savedxn(xn, wa, wb, grad_expand, *, store_h: bool = True):
    """Version B: reuse the SAVED xn (no normalize, no emit) + recompute a,b once -> (h, dA, dB).

    ``xn`` is the (M, K) normalized x emitted and saved by the forward kernel. The gatebwd
    kernel reads it directly as the GEMM operand (NORMALIZE=False), so the stats / gamma /
    beta and the inline-normalize-per-N-tile work are skipped entirely.
    """
    M, K = xn.shape
    ND = wa.shape[0]
    h = torch.empty_like(grad_expand) if store_h else grad_expand
    dA = torch.empty_like(grad_expand)
    dB = torch.empty_like(grad_expand)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(ND, meta["BLOCK_N"]))  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        xn, xn, xn, xn, xn,          # rstd/c1/g/beta unused when NORMALIZE=False (pass xn as filler)
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dA, dB, dA, xn,           # dAB/xn_ptr unused — pass existing tensors as filler
        M, ND, K, get_seq_group(M),
        xn.stride(0), xn.stride(1),
        wa.stride(0), wa.stride(1),
        grad_expand.stride(0), grad_expand.stride(1),
        dA.stride(0), dA.stride(1),
        xn.stride(0), xn.stride(1),
        NORMALIZE=False,
        STORE_H=store_h,
        STACK_DAB=False,
    )
    return (h, dA, dB) if store_h else (dA, dB)


def _transition_expand_gatebwd_savedxn_stacked(xn, wa, wb, grad_expand):
    """Version B stacked: emit h and dAB=[dA | dB] for two larger GEMMs downstream."""
    M, K = xn.shape
    ND = wa.shape[0]
    h = torch.empty_like(grad_expand)
    dAB = torch.empty(M, ND * 2, device=xn.device, dtype=grad_expand.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(ND, meta["BLOCK_N"]))  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        xn, xn, xn, xn, xn,
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dAB, dAB, dAB, xn,
        M, ND, K, get_seq_group(M),
        xn.stride(0), xn.stride(1),
        wa.stride(0), wa.stride(1),
        grad_expand.stride(0), grad_expand.stride(1),
        dAB.stride(0), dAB.stride(1),
        xn.stride(0), xn.stride(1),
        NORMALIZE=False,
        STORE_H=True,
        STACK_DAB=True,
    )
    return h, dAB


# fmt: off
@triton.autotune(
    configs=[triton.Config({"BLOCK_M": bm}, num_warps=nw)
             for bm in (1, 2, 4, 8, 16) for nw in (2, 4, 8)],
    key=["GROUP_M", "K"], reset_to_zero=["dg_ptr", "db_ptr"],
)
@triton.jit
def _transition_ln_bwd_kernel(
    dxn_ptr, x_ptr, rstd_ptr, c1_ptr, g_ptr, dx_ptr, dg_ptr, db_ptr,
    M, K, GROUP_M: tl.constexpr,
    stride_m, stride_k,
    dg_stride_replica, dg_stride_k,
    db_stride_replica, db_stride_k,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_REPLICAS: tl.constexpr, PRIVATIZE_DGDB: tl.constexpr,
):
    # LayerNorm backward consuming the SAVED stats (rstd, c1=mean*rstd), one pass over K:
    #   x_hat = x*rstd - c1 = (x-mean)*rstd ; wdy = gamma*dxn
    #   dx = rstd*(wdy - mean_k(wdy) - x_hat*mean_k(wdy*x_hat))
    #   dgamma += sum_m(dxn*x_hat) ; dbeta += sum_m(dxn)   (atomic over M-blocks)
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    rmask = rows < M
    kmask = k < K
    mask = rmask[:, None] & kmask[None, :]
    off = rows[:, None] * stride_m + k[None, :] * stride_k
    dxn = tl.load(dxn_ptr + off, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=rmask, other=0.0)
    g = tl.load(g_ptr + k, mask=kmask, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None] - c1[:, None]
    wdy = g[None, :] * dxn
    wdy = tl.where(mask, wdy, 0.0)
    x_hat = tl.where(mask, x_hat, 0.0)
    inv_k = 1.0 / K
    ca = tl.sum(x_hat * wdy, axis=1) * inv_k
    cb = tl.sum(wdy, axis=1) * inv_k
    dx = (wdy - (x_hat * ca[:, None] + cb[:, None])) * rstd[:, None]
    tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    pdg = tl.sum(dxn * x_hat, axis=0)
    pdb = tl.sum(dxn, axis=0)
    if PRIVATIZE_DGDB:
        replica = pid % NUM_REPLICAS
        tl.atomic_add(dg_ptr + replica * dg_stride_replica + k * dg_stride_k, pdg, mask=kmask)
        tl.atomic_add(db_ptr + replica * db_stride_replica + k * db_stride_k, pdb, mask=kmask)
    else:
        tl.atomic_add(dg_ptr + k, pdg, mask=kmask)
        tl.atomic_add(db_ptr + k, pdb, mask=kmask)
# fmt: on


def _transition_ln_bwd(dxn, x2, rstd, c1, gamma):
    """LayerNorm backward from saved stats -> (dx, dgamma, dbeta)."""
    M, K = x2.shape
    # Fastest path: hand-CUDA warp-per-row LN backward (register column-partials -> no atomics/no
    # spill, persistent grid SM*WAVES, vectorized loads). ~1.10x over privatized-Triton at K=128
    # (326 vs 358us, cos 1.0). Needs bf16/fp16 + K<=512 + contiguous + weight dtype == x dtype.
    # CUDA takes `mean`; recover it from c1=mean*rstd. Any mismatch -> graceful Triton fallback.
    if (
        os.getenv("MINIWORLD_TRANSITION_LNBWD_CUDA", "1").strip().lower()
        not in {"0", "false", "off"}
        and x2.dtype in (torch.float16, torch.bfloat16)
        and gamma.dtype == x2.dtype
        and K <= 512
        and x2.is_contiguous()
        and dxn.is_contiguous()
    ):
        try:
            from miniworld_kernels.kernels.layernorm.cuda import layer_norm_bwd_cuda

            mean = c1 / rstd
            return layer_norm_bwd_cuda(dxn, x2, gamma, mean, rstd)
        except Exception:  # noqa: BLE001 (build unavailable / dtype edge) -> Triton fallback
            pass
    dx = torch.empty_like(x2)
    # Default ON: privatizing dgamma/dbeta atomics across N replicas cuts atomic contention
    # (measured 1.31x @L=1024, ncu confirmed membar was the top stall). Set to 0/false/off to
    # restore the single-accumulator atomic path.
    privatize_dgdb = os.getenv(
        "MINIWORLD_TRANSITION_LNBWD_PRIVATIZE", "1").strip().lower() not in {"0", "false", "off"}
    if privatize_dgdb:
        num_replicas = _TRANSITION_LNBWD_PRIVATIZE_REPLICAS
        dgamma_acc = torch.zeros((num_replicas, K), device=x2.device, dtype=torch.float32)
        dbeta_acc = torch.zeros((num_replicas, K), device=x2.device, dtype=torch.float32)
    else:
        num_replicas = 1
        dgamma_acc = torch.zeros(K, device=x2.device, dtype=torch.float32)
        dbeta_acc = torch.zeros(K, device=x2.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _transition_ln_bwd_kernel[grid](
        dxn, x2, rstd, c1, gamma.contiguous(), dx, dgamma_acc, dbeta_acc,
        M, K, get_seq_group(M), x2.stride(0), x2.stride(1),
        dgamma_acc.stride(0), dgamma_acc.stride(-1),
        dbeta_acc.stride(0), dbeta_acc.stride(-1),
        BLOCK_K=triton.next_power_of_2(K),
        NUM_REPLICAS=num_replicas, PRIVATIZE_DGDB=privatize_dgdb,
    )
    if privatize_dgdb:
        dgamma = dgamma_acc.sum(dim=0)
        dbeta = dbeta_acc.sum(dim=0)
    else:
        dgamma = dgamma_acc
        dbeta = dbeta_acc
    return dx, dgamma, dbeta


class TritonTransitionFusedFunction(torch.autograd.Function):
    """Forward: fused (stats + LN + expand + SwiGLU) + squeeze.

    Backward is SEPARATE (non-fused): explicit per-stage ops (squeeze -> SwiGLU gate ->
    expand -> LayerNorm), reusing the LN stats (rstd, c1) saved by forward so mean/rstd are
    never recomputed.

    Two backward versions, selected by the ``save_xn`` flag:

    * Version A (``save_xn=False``, default): forward saves NO xn. The gatebwd kernel
      re-normalizes x inline from the saved stats (recomputes xn) and emits it for the wgrad
      GEMMs. Less memory; pays the inline-normalize-per-N-tile cost again in backward.
    * Version B (``save_xn=True``): forward EMITS the normalized x (xn, one extra store, it
      is already computed on-chip) and SAVES it. The gatebwd kernel reads the saved xn
      directly (NORMALIZE=False) — no recompute, no emit. Costs an extra (M, K) tensor.

    Both share the LayerNorm-backward kernel (which always needs raw x + stats for x_hat).
    """

    @typecheck
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        x: Float[torch.Tensor, "... d"],
        ln_weight: Float[torch.Tensor, "d"],
        ln_bias: Float[torch.Tensor, "d"],
        expand_a_weight: Float[torch.Tensor, "nd d"],
        expand_b_weight: Float[torch.Tensor, "nd d"],
        squeeze_weight: Float[torch.Tensor, "d nd"],
        n: int,
        eps: float,
        save_xn: bool = False,
    ) -> Float[torch.Tensor, "... d"]:
        orig_shape = x.shape
        K = orig_shape[-1]
        x2 = x.reshape(-1, K)

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x2 = x2.to(dtype)
            ln_weight = ln_weight.to(dtype)
            ln_bias = ln_bias.to(dtype)
            expand_a_weight = expand_a_weight.to(dtype)
            expand_b_weight = expand_b_weight.to(dtype)
            squeeze_weight = squeeze_weight.to(dtype)
        x2 = x2.contiguous()

        xn = None
        # Memory-light training path: reuse the FAST inference hand-CUDA b2b forward (fused
        # squeeze, h never in HBM) for d<=256 where it fits smem (d=128 ~1.29x vs triton,
        # d=256 confirmed). Version A (save_xn=False) saves no xn, so the shape-general
        # backward recomputes it — the CUDA kernel emits nothing beyond `out`. stats are needed
        # by the backward anyway, so stats_triton is not extra work. This gate is INDEPENDENT of
        # _B2B_MAX_K (the triton-b2b smem bound); on any failure we fall back to the split.
        _is_sm100 = torch.cuda.get_device_capability(x2.device)[0] == 10  # noqa: PLR2004
        cuda_b2b_ok = (
            (not save_xn)
            and n == 4
            and x2.dtype == torch.bfloat16
            and x2.is_cuda
            and x2.shape[0] % 128 == 0
            and (
                # sm_100, d>=256 ONLY: the cutlass-DSL b2b_fwd_sm100 forward fits smem
                # (the triton b2b/expand OOMs at d>=256) and keeps the fast gatebwd_sm100
                # backward usable there (~1.4-1.5x vs the legacy split). d=128 is DELIBERATELY
                # excluded: the triton b2b path below is faster at d=128 (602us vs 725us
                # training step, the AF3 shape) -- routing it to the cute fwd was a regression.
                (_is_sm100 and K in (256, 512))
                or (_cuda_b2b_train_enabled() and K in (128, 256))
            )
        )
        if cuda_b2b_ok:
            rstd, c1 = stats_triton(x2, eps)
            out = None
            if torch.cuda.get_device_capability(x2.device)[0] == 10:
                # B200 sm_100: the hand-CUDA sm90 b2b can't build (Hopper wgmma/TMA); use the
                # cutlass-DSL sm100 forward. Version A backward (below) is arch-agnostic and
                # recomputes xn from the saved stats, so it works unchanged with this forward.
                try:
                    from miniworld_kernels.kernels.transition.cute.b2b_fwd_sm100 import (
                        transition_b2b_sm100_ln,
                    )
                    out = transition_b2b_sm100_ln(
                        x2, ln_weight, ln_bias,
                        expand_a_weight, expand_b_weight, squeeze_weight, eps,
                    )
                except Exception:  # noqa: BLE001  DSL unavailable -> fall through
                    out = None
            if out is None:
                try:
                    from miniworld_kernels.kernels.transition.cuda import transition_b2b_fwd
                    out = transition_b2b_fwd(
                        x2, rstd, c1,
                        ln_weight.contiguous(), ln_bias.contiguous(),
                        expand_a_weight.contiguous(), expand_b_weight.contiguous(),
                        squeeze_weight.contiguous(),
                    )
                except Exception:  # noqa: BLE001  build unavailable -> split fallback (always fits)
                    expand = transition_expand_gate(
                        x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, eps,
                        stats=(rstd, c1), save_xn=False,
                    )
                    out = torch.matmul(expand, squeeze_weight.T)
        elif K <= _B2B_MAX_K:
            # Back-to-back fused (triton): squeeze folded in, h never materialized in HBM.
            if _transition_fuse_stats_enabled():
                res = transition_b2b(
                    x2, ln_weight, ln_bias,
                    expand_a_weight, expand_b_weight, squeeze_weight, eps,
                    save_xn=save_xn, fuse_stats=True,
                )
                if save_xn:
                    out, rstd, c1, xn = res
                else:
                    out, rstd, c1 = res
            else:
                # LN stats computed once and reused: by the forward kernel AND saved for the
                # separate backward (so backward never recomputes mean/rstd).
                rstd, c1 = stats_triton(x2, eps)
                res = transition_b2b(
                    x2, ln_weight, ln_bias,
                    expand_a_weight, expand_b_weight, squeeze_weight, eps,
                    stats=(rstd, c1), save_xn=save_xn, fuse_stats=False,
                )
                out, xn = res if save_xn else (res, None)
        else:
            # The expand kernel tiles K, so it keeps the separate stats pass.
            rstd, c1 = stats_triton(x2, eps)
            res = transition_expand_gate(
                x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, eps,
                stats=(rstd, c1), save_xn=save_xn,
            )
            expand, xn = res if save_xn else (res, None)
            out = torch.matmul(expand, squeeze_weight.T)

        if save_xn:
            ctx.save_for_backward(
                x2, rstd, c1, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight, xn,
            )
        else:
            ctx.save_for_backward(
                x2, rstd, c1, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight,
            )
        ctx.has_xn = save_xn
        ctx.n = n
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        return out.reshape(orig_shape)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_output: torch.Tensor):
        # SEPARATE (non-fused) backward: explicit per-stage ops, reusing the LN stats
        # (rstd, c1) saved by forward (no mean/rstd recompute). GEMMs are bf16 (matching
        # forward); gate/LN math accumulates in fp32.
        #   out = h @ Ws^T;  h = silu(a)*b;  a = xn@Wa^T, b = xn@Wb^T;  xn = (x-mean)*rstd*g+beta
        if ctx.has_xn:
            (
                x2, rstd, c1, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight, xn_saved,
            ) = ctx.saved_tensors
        else:
            (
                x2, rstd, c1, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight,
            ) = ctx.saved_tensors
            xn_saved = None
        orig_shape = ctx.orig_shape
        dt = x2.dtype
        K = x2.shape[-1]              # input dim
        D = squeeze_weight.shape[0]   # output dim (= K for Transition)

        go = grad_output.reshape(-1, D)
        if go.dtype != dt:
            go = go.to(dt)

        grad_expand = go @ squeeze_weight         # (1) dh  [M, ND]

        # sm100 (B200) Version A: the tuned sm100 gate-backward kernel replaces the slow
        # Triton _transition_expand_gatebwd (which was ~50% of the training step). xn is
        # recomputed from the saved LN stats (Version A) — it is needed by the wgrad GEMMs
        # anyway. Falls through to the Triton path if the DSL kernel is unavailable.
        if (not ctx.has_xn) and torch.cuda.get_device_capability(x2.device)[0] == 10:
            _sm100_ok = True
            try:
                from miniworld_kernels.kernels.transition.cute.gatebwd_sm100 import (
                    transition_expand_gatebwd_sm100,
                )
            except Exception:  # noqa: BLE001
                _sm100_ok = False
            if _sm100_ok:
                # Recompute xn with the tuned LN kernel (~13µs, matches the forward's
                # transition_b2b_sm100_ln); a torch stats-formula recompute is ~15x slower
                # (fp32 intermediates + many passes).
                from miniworld_kernels.kernels.layernorm.interface import layernorm_kernel
                xn = layernorm_kernel(x2, ln_weight, ln_bias, ctx.eps)
                h, dA, dB = transition_expand_gatebwd_sm100(
                    xn, expand_a_weight.contiguous(), expand_b_weight.contiguous(),
                    grad_expand,
                )
                dWs = go.t() @ h
                dWa = dA.t() @ xn
                dWb = dB.t() @ xn
                d_xn = dA @ expand_a_weight + dB @ expand_b_weight
                dx, dgamma, dbeta = _transition_ln_bwd(d_xn, x2, rstd, c1, ln_weight)
                return (
                    dx.reshape(orig_shape),
                    dgamma.to(ln_weight.dtype),
                    dbeta.to(ln_bias.dtype),
                    dWa, dWb, dWs, None, None, None,
                )

        # (2) gate backward is the ONLY stage that differs between Version A/B:
        #   B (has_xn):   reuse the saved xn (no re-normalize).
        #   A (recompute): re-normalize x from saved stats inline.
        # Both emit stacked dAB=[dA|dB] so stages (3)(4)(5)(6) below are version-INDEPENDENT.
        # USE_SAVEDXN_SPLIT_BWD keeps the old split comparator (has_xn only, default off).
        if ctx.has_xn and USE_SAVEDXN_SPLIT_BWD:
            h, dA, dB = _transition_expand_gatebwd_savedxn(
                xn_saved, expand_a_weight, expand_b_weight, grad_expand,
            )
            xn = xn_saved
            dWs = go.t() @ h
            dWa = dA.t() @ xn
            dWb = dB.t() @ xn
            d_xn = dA @ expand_a_weight + dB @ expand_b_weight
            dx, dgamma, dbeta = _transition_ln_bwd(d_xn, x2, rstd, c1, ln_weight)
            return (
                dx.reshape(orig_shape),
                dgamma.to(ln_weight.dtype),
                dbeta.to(ln_bias.dtype),
                dWa, dWb, dWs, None, None, None,
            )

        if ctx.has_xn:
            h, dAB = _transition_expand_gatebwd_savedxn_stacked(
                xn_saved, expand_a_weight, expand_b_weight, grad_expand,
            )
            xn = xn_saved
        else:
            h, dAB, xn = _transition_expand_gatebwd_stacked(
                x2, rstd, c1, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, grad_expand,
            )

        # (3)(4)(5)(6) SHARED across versions (stacked): two larger GEMMs + LN bwd.
        dWs = go.t() @ h                          # (3) [D, ND]
        w_ab = torch.cat((expand_a_weight, expand_b_weight), dim=0)
        dWab = dAB.t() @ xn                        # (4) fuses dWa,dWb -> [2*ND, K]
        dWa = dWab[: expand_a_weight.shape[0]]
        dWb = dWab[expand_a_weight.shape[0] :]
        if (
            os.getenv("TRANSITION_DAB_LNBWD", "0") == "1"
            and K <= 128
            and torch.cuda.get_device_capability(x2.device)[0] >= 9
        ):
            from miniworld_kernels.kernels.transition.cute.dab_lnbwd import (
                transition_dab_lnbwd_cute,
            )

            dx = transition_dab_lnbwd_cute(dAB, w_ab, x2, ln_weight, rstd, c1)
            db_ab = dAB.sum(0)
            # xn = gamma*xhat + beta, so dAB.T@xhat is recovered from dAB.T@xn. Experimental/gated.
            t_xhat = (
                dWab.float() - db_ab.float()[:, None] * ln_bias.float()[None, :]
            ) / ln_weight.float()[None, :]
            dgamma = (w_ab.float() * t_xhat).sum(0)
            dbeta = db_ab.float() @ w_ab.float()
            return (
                dx.reshape(orig_shape),
                dgamma.to(ln_weight.dtype),
                dbeta.to(ln_bias.dtype),
                dWa, dWb, dWs, None, None, None,
            )
        d_xn = dAB @ w_ab                          # (5) fuses dA@Wa + dB@Wb -> [M, K]

        # (6) LayerNorm backward from saved stats -> dx, dgamma, dbeta (hand-CUDA at d<=512 bf16).
        dx, dgamma, dbeta = _transition_ln_bwd(d_xn, x2, rstd, c1, ln_weight)
        return (
            dx.reshape(orig_shape),
            dgamma.to(ln_weight.dtype),
            dbeta.to(ln_bias.dtype),
            dWa, dWb, dWs, None, None, None,
        )


def triton_transition_fused(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor,
    n: int,
    eps: float = 1e-5,
    save_xn: bool = False,
) -> torch.Tensor:
    """Fully fused Transition forward (LN folded in).

    ``save_xn`` selects the backward version: False (default) = Version A (recompute xn in
    backward, less memory); True = Version B (save xn in forward, reuse in backward).
    """
    return TritonTransitionFusedFunction.apply(
        x, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps, save_xn
    )
