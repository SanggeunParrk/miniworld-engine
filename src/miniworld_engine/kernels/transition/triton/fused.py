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

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for

import os

import torch
from miniworld_engine import settings
import triton
import triton.language as tl

from miniworld_engine.kernels._tiles import check_tile_axes, tile_grid, tile_order

from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.autotune.shape_key import both_key, pack, length_of, rows_of
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

AUTOTUNE = settings.current().autotunes("transition")
def use_savedxn_split_bwd() -> bool:
    """Read at call time; see layernorm.compile_native._ln_bwd_override."""
    return settings.current().transition_savedxn_split_bwd


def _gatebwd_wgmma_enabled() -> bool:
    """Route the sm90 large-d (K in {256,512}) Version-A gate-backward through the hand-CUDA
    WGMMA kernel (beats the Triton recompute). Default on; set settings.transition_gatebwd_wgmma
    to A/B against the Triton path."""
    return settings.current().transition_gatebwd_wgmma


def _cuda_b2b_train_enabled() -> bool:
    """Use the fast inference b2b CUDA kernel for the TRAINING forward too (Version A / save_xn=False:
    saves no xn, backward recomputes it). Same env toggle as the inference dispatch."""
    return settings.current().transition_cuda_b2b
_TRANSITION_LNBWD_PRIVATIZE_REPLICAS = 64


def _transition_fuse_stats_enabled() -> bool:
    return settings.current().transition_fuse_stats


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_mixed
    return bucket_mixed(length)


def _shape_key(shape_key: int | None, rows: int, **axes: int) -> int:
    """The autotune shape key for a launch: L bucketed against the ``both`` set.

    Every launcher in this module takes ``shape_key`` = ``both_key(rows_of(<pre-flatten
    shape>))`` from the caller that still holds the activation's shape (the autograd Function
    below, or ``transition/cute/fused.py`` / ``transition/triton/main.py``). ``None`` is the
    TRANSITIONAL path for the driver/checker harnesses (``drivers/transition.py`` / ``checks/transition.py``,
    owned by the coordinator), which still call these launchers with no key: it buckets the
    flattened ROW count, which is exactly the L-vs-L*L ambiguity ``autotune.shape_key`` exists
    to remove. No model path reaches it.
    """
    return both_key(rows, **axes) if shape_key is None else pack(shape_key, **axes)



# BLOCK_K is a CSV tile. It used to arrive from the launcher as
# ``BLOCK_K=next_power_of_2(K)`` -- the whole d row in one tile, which is also what forced the
# smem cliff and the "no config fits" footgun below. The CSV reaches
# 1024 (the wrapper's K ceiling), so that single-tile schedule is still in the sweep; the k-loop
# in the kernel makes every smaller candidate correct instead of wrong, and bounds smem.


# fmt: off
@triton.autotune(configs=configs_for("transition_layernorm_expand_swiglu_triton"),
                 key=['shape_key', 'SAVE_XN'])
@triton.jit
def _transition_expand_gate_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, out_ptr, xn_ptr,
    # K is tl.constexpr (it is the model's d, fixed per module, and already in this kernel's
    # autotune key), so `BLOCK_K_D >= K` below is a COMPILE-TIME test and only one branch is emitted.
    M, ND, K: tl.constexpr, shape_key,
    stride_xm, stride_xk,
    stride_wn, stride_wk,   # Wa, Wb share layout: (ND, K) row-major -> stride_wn=K, stride_wk=1
    stride_om, stride_on,
    stride_nm, stride_nk,   # xn out: (M, K) row-major (only used when SAVE_XN)
    BLOCK_M1: tl.constexpr, BLOCK_K_ND: tl.constexpr, BLOCK_K_D: tl.constexpr,
    SAVE_XN: tl.constexpr,
):
    # One program owns BLOCK_M1 rows and ALL of ND. LayerNorm uses PRECOMPUTED row stats, so
    # normalizing a k-tile needs nothing but that tile: BLOCK_K_D tiles the d axis and the two
    # projections accumulate across it.
    pid_m = tl.program_id(0).to(tl.int64)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)

    if BLOCK_K_D >= K:
        # COVERING TILE -> the pre-tiling schedule: ONE x read, normalized ONCE, and the single
        # bf16 `xn` tile held in registers and reused by the SAVE_XN store and by both projections
        # of EVERY ND chunk. The general branch below has the x load nested inside the ND loop, so
        # it re-reads x ceil(ND/BLOCK_K_ND) times per row -- at ND=4d that is a 4x read amplification
        # of the kernel's largest input. Numerics are identical to the else-branch at BLOCK_K_D >= K:
        # its loops are single-trip and every expression here matches it term for term.
        k = tl.arange(0, BLOCK_K_D)
        k_mask = k < K
        km = row_mask[:, None] & k_mask[None, :]
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=km, other=0.0,
        ).to(tl.float32)
        g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        xhat = x * rstd[:, None] - c1[:, None]
        xn = (xhat * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)  # (BM, BK)
        if SAVE_XN:
            tl.store(xn_ptr + rows[:, None] * stride_nm + k[None, :] * stride_nk, xn, mask=km)
        for n0 in range(0, ND, BLOCK_K_ND):
            cols = n0 + tl.arange(0, BLOCK_K_ND)
            col_mask = cols < ND
            a = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
            b = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
            wa = tl.load(  # (BLOCK_K_D, BLOCK_K_ND): w[k, n] = W[cols[n], k]
                wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            wb = tl.load(
                wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            a = tl.dot(xn, wa, a, out_dtype=tl.float32)
            b = tl.dot(xn, wb, b, out_dtype=tl.float32)
            gate = a * tl.sigmoid(a) * b
            tl.store(
                out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
                gate.to(out_ptr.dtype.element_ty),
                mask=row_mask[:, None] & col_mask[None, :],
            )
    else:
        # --- Version B: stash the normalized x so backward can reuse it ---
        if SAVE_XN:
            for k0 in range(0, K, BLOCK_K_D):
                k = k0 + tl.arange(0, BLOCK_K_D)
                k_mask = k < K
                km = row_mask[:, None] & k_mask[None, :]
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=km, other=0.0,
                ).to(tl.float32)
                g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(
                    x_ptr.dtype.element_ty)
                tl.store(xn_ptr + rows[:, None] * stride_nm + k[None, :] * stride_nk, xn, mask=km)

        # --- loop the two projections over N-tiles; each contracts over the K-tiles ---
        for n0 in range(0, ND, BLOCK_K_ND):
            cols = n0 + tl.arange(0, BLOCK_K_ND)
            col_mask = cols < ND
            a = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
            b = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K_D):
                k = k0 + tl.arange(0, BLOCK_K_D)
                k_mask = k < K
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                xhat = x * rstd[:, None] - c1[:, None]
                xn = (xhat * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)  # (BM, BK)
                wa = tl.load(  # (BLOCK_K_D, BLOCK_K_ND): w[k, n] = W[cols[n], k]
                    wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0,
                )
                wb = tl.load(
                    wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0,
                )
                a = tl.dot(xn, wa, a, out_dtype=tl.float32)
                b = tl.dot(xn, wb, b, out_dtype=tl.float32)
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
    shape_key: int | None = None,   # both_key(rows_of(pre-flatten shape)) from the caller
):
    """LayerNorm(x) then SwiGLU(expand_a, expand_b) -> expand (M, ND). Stats fused-out.

    ``save_xn`` (Version B): also emit the normalized x (M, K) via a single in-kernel store
    so the backward can reuse it instead of recomputing. Returns ``(expand, xn)`` then; else
    just ``expand``.
    """
    M, K = x2.shape
    ND = wa.shape[0]
    assert wa.shape[1] == K and wb.shape == wa.shape

    # The old footgun guard lived here: with BLOCK_K pinned to next_pow2(K) the weight tiles were
    # [K, BLOCK_N] and at large K NO config fit device smem, so the wrapper had to detect that and
    # raise. BLOCK_K is a tuned, looped tile now -- the smallest candidate is 16, so a fitting
    # config always exists and `_expand_early_prune` selects among the ones that do. Nothing to
    # guard, and no K ceiling either.

    rstd, c1 = stats if stats is not None else stats_triton(x2, eps, shape_key=shape_key)
    expand = torch.empty(M, ND, device=x2.device, dtype=x2.dtype)
    xn = torch.empty(M, K, device=x2.device, dtype=x2.dtype) if save_xn else expand
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731 (N looped in-kernel)
    _transition_expand_gate_kernel[grid](
        x2, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), expand, xn,
        M, ND, K, _shape_key(shape_key, M, ND=ND, K=K),
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        expand.stride(0), expand.stride(1),
        xn.stride(0), xn.stride(1),
        SAVE_XN=save_xn,
    )
    return (expand, xn) if save_xn else expand


# The July winner for the d=128 b2b path was BLOCK_M1=64, ND chunk 64, full-D output, ~1.66 ms.
# THREE tuned extents now, deliberately decoupled: BLOCK_K_D (d contraction), BLOCK_N (ND inner
# chunk) and BLOCK_K_ND (squeeze output / D). BLOCK_N and BLOCK_K_ND briefly SHARED one axis to
# hold the sweep down -- but that made the covering schedule (BLOCK_K_ND >= D, full-D accumulator)
# also loop ND in D-wide chunks, and the July full-D-output-with-a-64-wide-ND-chunk config was then
# unreachable: measured +13% (1.87 vs 1.66) on the AF3 d=128 shape. The extra axis is worth it; the
# prune below keeps the resulting sweep small by pinning the two covering extents and only letting
# BLOCK_N/BLOCK_M1/warps/stages vary.




# ONE tl.constexpr is deliberately absent from the key below, and one is gone from the kernel.
#
# `D` -- GONE, not merely unkeyed. It was `ws.shape[0]` and `K` is `x2.shape[1]`, and both are
# the same d_hidden at every launcher (Transition's Linears are expand d -> n*d and squeeze
# n*d -> d; drivers/checks transition build ws as rows2d(k, nd)). The hand-CUDA twin asserts it
# outright (transition_b2b_kernel.cu: `TORCH_CHECK(D == K, ...)`), and the ADD_RESIDUAL branch
# below relies on it when it reloads x over the output columns. Its one read was the squeeze
# output mask, which `K` states exactly as well. `ND` is NOT implied -- n is a module argument
# (4 in Transition, 2 in ConditionedTransition) -- so it stays.
#
# `EPS` -- a numeric tolerance, not a shape and not a code path. It reaches the kernel only as
# `tl.rsqrt(var + EPS)` under FUSE_STATS (which IS keyed), it is nn.LayerNorm's `ln_in.eps`, and
# no value of it can change which tile is fastest. Keying on it would multiply the bucket count
# for nothing.
def _prefer_covering_b2b(configs, nargs, **_):
    """Covering-when-fits, sized tight: tune only the SMALLEST tile that covers the whole d row.

    The kernel has two schedules chosen at compile time by ``BLOCK_K_D >= K`` (and D == K here, so
    ``BLOCK_K_ND >= K`` covers the squeeze output too): the *covering* one reads x once and keeps the
    normalized row resident (the fast, July schedule), and the *k-tiled* ``else`` re-reads x per
    ND chunk (needed only when d is too large for one tile). Three things this fixes, all measured on
    the AF3 d=128 shape:

      * the config grid used to top out at BLOCK_K_D=64 < K=128, so the covering branch was
        UNREACHABLE and every launch took the k-tiled path (~4x x re-read) -- 4.17 vs 1.66 ms;
      * once 128/256 are in the grid the raw autotuner still mis-picks -- it chose BLOCK_K_ND=256
        at D=128 (half the squeeze tile masked-off waste) and measured 5.2 ms;
      * BLOCK_N (the ND inner chunk) is now its own axis, so the covering schedule can pair a full-D
        output (BLOCK_K_ND >= D) with a NARROW ND chunk -- the July BLOCK_N=64 config, which was
        unreachable while the ND chunk and the output width were one knob (+13%).

    So pin the two covering extents to their smallest covering value (BLOCK_K_D and BLOCK_K_ND) and
    leave the schedule's real degrees of freedom -- BLOCK_N, BLOCK_M1, warps, stages -- to the tuner.
    When no config covers K (d larger than every offered tile) the covering set is empty and the full
    list is returned -- the k-tiled ``else`` is exactly the fallback for that case.
    """
    k = nargs["K"]
    cov = [c for c in configs
           if c.kwargs["BLOCK_K_D"] >= k and c.kwargs["BLOCK_K_ND"] >= k]
    if not cov:
        return list(configs)
    dmin = min(c.kwargs["BLOCK_K_D"] for c in cov)
    ndmin = min(c.kwargs["BLOCK_K_ND"] for c in cov)
    # GROUP_M is the output-tile VISIT ORDER, and the covering schedule has exactly one output tile
    # (BLOCK_K_ND >= D -> cdiv(D, BLOCK_K_ND) == 1 -> pid_d == 0 always), so every GROUP_M value
    # compiles the identical kernel. Pin it to the smallest so the tuner does not bench each config
    # twice for a knob that cannot matter here.
    gmin = min(c.kwargs["GROUP_M"] for c in cov)
    return [c for c in cov
            if c.kwargs["BLOCK_K_D"] == dmin and c.kwargs["BLOCK_K_ND"] == ndmin
            and c.kwargs["GROUP_M"] == gmin]


# fmt: off
@triton.autotune(configs=configs_for("transition_fwd_b2b_triton"),
                 key=['shape_key', 'SAVE_XN', 'FUSE_STATS', 'ADD_RESIDUAL',
                      'HAS_LN'],
                 prune_configs_by={'early_config_prune': _prefer_covering_b2b})
@triton.jit
def _transition_b2b_kernel(
    x_ptr, rstd_ptr, c1_ptr, rstd_out_ptr, c1_out_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, ws_ptr, out_ptr, xn_ptr,
    M, ND, K: tl.constexpr, shape_key, EPS: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_sd, stride_sn,    # Ws: (D, ND) row-major
    stride_om, stride_od,
    stride_nm, stride_nk,    # xn out: (M, K) row-major (only used when SAVE_XN)
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K_ND: tl.constexpr,
    BLOCK_K_D: tl.constexpr,
    SAVE_XN: tl.constexpr, FUSE_STATS: tl.constexpr, ADD_RESIDUAL: tl.constexpr,
    HAS_LN: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Back-to-back: a program owns BLOCK_M1 rows x BLOCK_K_ND output columns and ALL of ND. It builds
    # the gated h tile-by-tile and ACCUMULATES the squeeze out[BM, BD] += h_chunk @ Ws[:, chunk]^T,
    # so the (M, ND) intermediate h never touches HBM. THREE tuned extents, decoupled: BLOCK_K_D tiles
    # the d contraction, BLOCK_N tiles the ND inner loop (how wide an expand chunk to gate at once),
    # and BLOCK_K_ND tiles the squeeze OUTPUT (D). They used to share one axis (BLOCK_K_ND drove both
    # the ND chunk and the output), which forced the covering schedule -- BLOCK_K_ND >= D for a full-D
    # accumulator -- to ALSO loop ND in D-wide chunks. The July winner was a full-D output with a
    # NARROWER ND chunk (BLOCK_N=64), unreachable while they were the same knob (measured +13% on the
    # AF3 d=128 shape). Splitting BLOCK_N back out restores it. At BLOCK_K_D >= K, BLOCK_K_ND >= D and
    # BLOCK_N a divisor of ND the grid is 1-D over M and the d-loop is single-trip -- the July schedule.
    # Visit order, tuned: see kernels/_tiles.py. b2b: ND is looped inside, so every program reads all of Wa/Wb -- the WEIGHTS are what gets
    # re-read here, not x, and the row-first walk this had may already be right. `K` is the
    # output width (D == K here). The `pid_d == 0` guards below test the VALUE, not the launch
    # order, so reordering leaves exactly one program doing the stats write.
    pid_m, pid_d = tile_order(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, BLOCK_M1), tl.cdiv(K, BLOCK_K_ND), GROUP_M)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M
    dcols = pid_d * BLOCK_K_ND + tl.arange(0, BLOCK_K_ND)   # squeeze output tile: BLOCK_K_ND-wide
    d_mask = dcols < K
    out_acc = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)

    if BLOCK_K_D >= K:
        # COVERING TILE. BLOCK_K_D and K are both tl.constexpr, so this comparison is resolved at
        # COMPILE time and only one of the two branches is ever emitted. One tile holds the whole
        # d row, so read x ONCE, normalize once, and reuse the single bf16 `xn` tile for the
        # stats, the SAVE_XN store and both projections of every ND chunk -- exactly the
        # pre-tiling schedule. The k-tiled `else` below is the general (BLOCK_K_D < K) form; at
        # BLOCK_K_D >= K its loops are single-trip and every expression here matches it term for
        # term, so the two branches are numerically identical.
        k = tl.arange(0, BLOCK_K_D)
        k_mask = k < K
        km = row_mask[:, None] & k_mask[None, :]
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=km, other=0.0,
        ).to(tl.float32)
        if FUSE_STATS:
            inv_k = 1.0 / K
            mean = tl.sum(x, axis=1) * inv_k
            x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
            var = tl.sum(x_centered * x_centered, axis=1) * inv_k
            rstd = tl.rsqrt(var + EPS)
            c1 = mean * rstd
            if pid_d == 0:
                tl.store(rstd_out_ptr + rows, rstd, mask=row_mask)
                tl.store(c1_out_ptr + rows, c1, mask=row_mask)
        elif HAS_LN:
            rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
            c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
        else:
            # No LayerNorm at all. rstd=1 and c1=0 -- with g=1, beta=0 below -- make the
            # SAME xn expression the identity, so the schedule, the masking and the
            # tiling stay ONE code path instead of two that can drift apart.
            rstd = tl.full([BLOCK_M1], 1.0, tl.float32)
            c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        if HAS_LN:
            g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
            beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        else:
            g = tl.full([BLOCK_K_D], 1.0, tl.float32)
            beta = tl.zeros([BLOCK_K_D], dtype=tl.float32)
        xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(
            x_ptr.dtype.element_ty)
        if SAVE_XN:
            if pid_d == 0:
                tl.store(xn_ptr + rows[:, None] * stride_nm + k[None, :] * stride_nk, xn, mask=km)
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
            ws_t = tl.load(  # (BN, BD): Ws[d, cols]^T
                ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
                mask=col_mask[:, None] & d_mask[None, :], other=0.0,
            )
            out_acc = tl.dot(h, ws_t, out_acc, out_dtype=tl.float32)
    else:
        if FUSE_STATS:
            # Two sweeps over K (mean, then CENTERED variance) so the fp32 algebra at BLOCK_K_D >= K is
            # exactly the original single-tile one.
            inv_k = 1.0 / K
            acc_s = tl.zeros([BLOCK_M1], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K_D):
                k = k0 + tl.arange(0, BLOCK_K_D)
                k_mask = k < K
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                acc_s += tl.sum(x, axis=1)
            mean = acc_s * inv_k
            acc_s = tl.zeros([BLOCK_M1], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K_D):
                k = k0 + tl.arange(0, BLOCK_K_D)
                k_mask = k < K
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
                acc_s += tl.sum(x_centered * x_centered, axis=1)
            var = acc_s * inv_k
            rstd = tl.rsqrt(var + EPS)
            c1 = mean * rstd
            # Only the first D-block writes the per-row stats (every D-block computes the same value;
            # one writer keeps it a single store rather than cdiv(D, BLOCK_K_ND) redundant ones).
            if pid_d == 0:
                tl.store(rstd_out_ptr + rows, rstd, mask=row_mask)
                tl.store(c1_out_ptr + rows, c1, mask=row_mask)
        elif HAS_LN:
            rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
            c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
        else:
            # No LayerNorm at all. rstd=1 and c1=0 -- with g=1, beta=0 below -- make the
            # SAME xn expression the identity, so the schedule, the masking and the
            # tiling stay ONE code path instead of two that can drift apart.
            rstd = tl.full([BLOCK_M1], 1.0, tl.float32)
            c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)

        # --- Version B: stash the normalized x for backward reuse (first D-block only) ---
        if SAVE_XN:
            if pid_d == 0:
                for k0 in range(0, K, BLOCK_K_D):
                    k = k0 + tl.arange(0, BLOCK_K_D)
                    k_mask = k < K
                    km = row_mask[:, None] & k_mask[None, :]
                    x = tl.load(
                        x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                        mask=km, other=0.0,
                    ).to(tl.float32)
                    if HAS_LN:
                        g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                        beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                    else:
                        g = tl.full([BLOCK_K_D], 1.0, tl.float32)
                        beta = tl.zeros([BLOCK_K_D], dtype=tl.float32)
                    xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(
                        x_ptr.dtype.element_ty)
                    tl.store(xn_ptr + rows[:, None] * stride_nm + k[None, :] * stride_nk, xn, mask=km)

        for n0 in range(0, ND, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < ND
            a = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
            b = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K_D):
                k = k0 + tl.arange(0, BLOCK_K_D)
                k_mask = k < K
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                if HAS_LN:
                    g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                    beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                else:
                    g = tl.full([BLOCK_K_D], 1.0, tl.float32)
                    beta = tl.zeros([BLOCK_K_D], dtype=tl.float32)
                xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(
                    x_ptr.dtype.element_ty)
                wa = tl.load(
                    wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0,
                )
                wb = tl.load(
                    wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0,
                )
                a = tl.dot(xn, wa, a, out_dtype=tl.float32)
                b = tl.dot(xn, wb, b, out_dtype=tl.float32)
            h = (a * tl.sigmoid(a) * b).to(x_ptr.dtype.element_ty)  # (BM, BN)
            ws_t = tl.load(  # (BN, BD): Ws[d, cols]^T
                ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
                mask=col_mask[:, None] & d_mask[None, :], other=0.0,
            )
            out_acc = tl.dot(h, ws_t, out_acc, out_dtype=tl.float32)
    if ADD_RESIDUAL:
        # Fuse the post-transition residual add: y = transition(x) + x. The residual is the
        # kernel's OWN pre-LN input x (the module never mutates it before `pair + transition(pair)`),
        # so no extra tensor arg — reload the input row tile over the D output columns (D == K here;
        # L2-hot from the LN load above) and add in fp32 before the single output store.
        res = tl.load(
            x_ptr + rows[:, None] * stride_xm + dcols[None, :] * stride_xk,
            mask=row_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        out_acc += res
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        out_acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None] & d_mask[None, :],
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
    add_residual: bool = False,
    shape_key: int | None = None,
    has_ln: bool = True,
):
    """Fully fused LN + SwiGLU expand + squeeze -> out (M, D). h never hits HBM.

    ``has_ln=False`` drops the LayerNorm and computes ``squeeze(SwiGLU(x @ Wa^T, x @ Wb^T))``
    -- the bare SwiGLU FFN, for a caller whose normalization is its own (an adaLN-modulated
    RMSNorm outside the block, say). ``ln_weight``/``ln_bias``/``stats`` are then unread and
    may be empty; ``fuse_stats`` must be off, since there are no statistics to fuse.

    Requires K to fit one BLOCK_K (K = next_pow2(K) <= 1024) AND the (x row + weight tiles)
    working set to fit smem — practical only for small K (d <= 128). Caller falls back to
    ``transition_expand_gate`` + ``torch.matmul`` for larger K. ``stats`` lets the caller
    pass precomputed (rstd, c1) so the backward can reuse the same LN statistics.

    ``save_xn`` (Version B): also emit the normalized x (M, K) via a single in-kernel store
    for backward reuse. Returns ``(out, xn)`` then; else just ``out``.

    With ``fuse_stats=True`` (or ``settings.transition_fuse_stats``), the
    kernel computes and stores the LayerNorm stats on-chip. Then returns
    ``(out, rstd, c1, xn)`` when ``save_xn`` else ``(out, rstd, c1)``.
    """
    M, K = x2.shape
    ND = wa.shape[0]
    D = ws.shape[0]
    fuse_stats = (False if not has_ln
                  else _transition_fuse_stats_enabled() if fuse_stats is None else fuse_stats)
    if not has_ln:
        if stats is not None:
            raise ValueError("transition_b2b(has_ln=False) normalizes nothing; do not pass stats")
        # Unread by the kernel under HAS_LN=False, but the launch still needs pointers.
        rstd = c1 = x2.new_empty(0, dtype=torch.float32)
        ln_weight = ln_bias = x2.new_empty(0)
    elif fuse_stats:
        if stats is not None:
            raise ValueError("transition_b2b(fuse_stats=True) computes stats in-kernel; do not pass stats")
        rstd = torch.empty(M, device=x2.device, dtype=torch.float32)
        c1 = torch.empty(M, device=x2.device, dtype=torch.float32)
    else:
        rstd, c1 = stats if stats is not None else stats_triton(x2, eps, shape_key=shape_key)
    out = torch.empty(M, D, device=x2.device, dtype=x2.dtype)
    xn = torch.empty(M, K, device=x2.device, dtype=x2.dtype) if save_xn else out
    check_tile_axes("transition_fwd_b2b_triton", D, K, "D (ws rows)", "K (x columns)")
    grid = lambda meta: tile_grid(M, D, meta["BLOCK_M1"], meta["BLOCK_K_ND"])  # noqa: E731
    _transition_b2b_kernel[grid](
        x2, rstd, c1, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), ws.contiguous(), out, xn,
        M, ND, K, _shape_key(shape_key, M, ND=ND, K=K), eps,
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        ws.stride(0), ws.stride(1),
        out.stride(0), out.stride(1),
        xn.stride(0), xn.stride(1),
        SAVE_XN=save_xn,
        FUSE_STATS=fuse_stats,
        ADD_RESIDUAL=add_residual,
        HAS_LN=has_ln,
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
# large-K scaling). The squeeze fusion is unchanged: one program owns BLOCK_M1 rows x ALL of
# n*d, accumulates out_acc[BLOCK_M1, D] across the N-chunk loop, writes once. Separate stats
# (rstd, c1 precomputed) let us normalize each k-tile without the full row.
# ---------------------------------------------------------------------------






# `D` is GONE from this kernel, not merely unkeyed: it is ws.shape[0] and K is
# x2.shape[1], both the module's d_hidden, so `K` (keyed) already partitions this axis. See the
# longer note on `_transition_b2b_kernel` above. `ND` is independent (n is a module argument).
# fmt: off
@triton.autotune(configs=configs_for("transition_fwd_b2b_ktiled_triton"), key=['shape_key'])
@triton.jit
def _transition_b2b_ktiled_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, ws_ptr, out_ptr,
    # K is tl.constexpr (model d, fixed per module, already in this kernel's autotune key) so the
    # `BLOCK_K_D >= K` guard below resolves at COMPILE time and only one branch is emitted.
    M, ND, K: tl.constexpr, shape_key,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_sd, stride_sn,    # Ws: (D, ND) row-major
    stride_om, stride_od,
    BLOCK_M1: tl.constexpr, BLOCK_K_ND: tl.constexpr, BLOCK_K_D: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # One program owns BLOCK_M1 rows x BLOCK_K_ND output columns and ALL of ND. Inner k-loop keeps
    # weight tiles [BLOCK_K_D, BLOCK_K_ND] (bounded smem at any d); squeeze accumulated in out_acc
    # across the N-chunk loop; h never leaves regs. No atomics.
    # Visit order, tuned: see kernels/_tiles.py. The K-tiled b2b: same operand story as the plain one above.
    pid_m, pid_d = tile_order(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, BLOCK_M1), tl.cdiv(K, BLOCK_K_ND), GROUP_M)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
    dcols = pid_d * BLOCK_K_ND + tl.arange(0, BLOCK_K_ND)   # squeeze output tile: BLOCK_K_ND-wide
    d_mask = dcols < K
    out_acc = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)

    if BLOCK_K_D >= K:
        # COVERING TILE -> read x ONCE and hold the normalized bf16 tile in registers across every
        # ND chunk, instead of re-reading and re-normalizing it inside the chunk loop
        # (ceil(ND/BLOCK_K_ND) times per row). Same arithmetic as the else-branch at BLOCK_K_D >= K,
        # where the k-loop is single-trip and a_acc/b_acc start from an exact fp32 zero.
        k = tl.arange(0, BLOCK_K_D)
        k_mask = k < K
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)
        for n0 in range(0, ND, BLOCK_K_ND):
            cols = n0 + tl.arange(0, BLOCK_K_ND)
            col_mask = cols < ND
            wa = tl.load(
                wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            wb = tl.load(
                wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            a_acc = tl.dot(xn, wa, out_dtype=tl.float32)
            b_acc = tl.dot(xn, wb, out_dtype=tl.float32)
            h = (a_acc * tl.sigmoid(a_acc) * b_acc).to(x_ptr.dtype.element_ty)  # (BM, BN)
            ws_t = tl.load(
                ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
                mask=col_mask[:, None] & d_mask[None, :], other=0.0,
            )  # (BN, BD)
            out_acc = tl.dot(h, ws_t, out_acc, out_dtype=tl.float32)
        tl.store(
            out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
            out_acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None] & d_mask[None, :],
        )
    else:
        for n0 in range(0, ND, BLOCK_K_ND):
            cols = n0 + tl.arange(0, BLOCK_K_ND)
            col_mask = cols < ND
            a_acc = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
            b_acc = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K_D):
                k = k0 + tl.arange(0, BLOCK_K_D)
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
                mask=col_mask[:, None] & d_mask[None, :], other=0.0,
            )  # (BN, BD)
            out_acc = tl.dot(h, ws_t, out_acc, out_dtype=tl.float32)
        tl.store(
            out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
            out_acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None] & d_mask[None, :],
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
    shape_key: int | None = None,
) -> torch.Tensor:
    """K-tiled fully-fused LN + SwiGLU expand + squeeze -> out (M, D), h off HBM, any d.

    UNVERIFIED until cos-checked + benched on H100. Bounded smem (weight tiles
    [BLOCK_K, BLOCK_N]) -> no OOM at d>=256, unlike ``transition_b2b``.
    """
    M, K = x2.shape
    ND = wa.shape[0]
    D = ws.shape[0]
    rstd, c1 = stats_triton(x2, eps, shape_key=shape_key)
    out = torch.empty(M, D, device=x2.device, dtype=x2.dtype)
    check_tile_axes("transition_fwd_b2b_ktiled_triton", D, K, "D (ws rows)", "K (x columns)")
    grid = lambda meta: tile_grid(M, D, meta["BLOCK_M1"], meta["BLOCK_K_ND"])  # noqa: E731
    _transition_b2b_ktiled_kernel[grid](
        x2, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), ws.contiguous(), out,
        M, ND, K, _shape_key(shape_key, M, ND=ND, K=K),
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


def _swiglu_b2b_fake(x2, wa, wb, ws, shape_key):
    """``out`` (M, D) in ``x2``'s dtype -- D is the squeeze weight's row count."""
    return x2.new_empty((x2.shape[0], ws.shape[0]))


@opaque(fake=_swiglu_b2b_fake, name="transition_swiglu_b2b")
def _swiglu_b2b(x2: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
                shape_key: int) -> torch.Tensor:
    """The no-LayerNorm b2b launch, and only the launch.

    Opaque for the same reason every other launcher here is: `transition_b2b` is reachable from
    Dynamo through this autograd Function, and a bare triton launch on a traced path is what
    `tests/compile/test_compile_wrap_coverage.py` exists to catch.
    """
    empty = x2.new_empty(0)
    return transition_b2b(x2, empty, empty, wa, wb, ws, 0.0,
                          has_ln=False, shape_key=shape_key)


class _SwiGLUFFNFused(torch.autograd.Function):
    """``squeeze(SwiGLU(x @ Wa^T, x @ Wb^T))`` in ONE kernel -- no LayerNorm, no residual."""

    @staticmethod
    def forward(ctx, x, expand_a_weight, expand_b_weight, squeeze_weight):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if x2.stride(1) != 1:
            x2 = x2.contiguous()
        out = _swiglu_b2b(
            x2, expand_a_weight.contiguous(), expand_b_weight.contiguous(),
            squeeze_weight.contiguous(), both_key(rows_of(orig_shape)),
        )
        ctx.save_for_backward(x2, expand_a_weight, expand_b_weight, squeeze_weight)
        ctx.shape, ctx.nd = orig_shape, expand_a_weight.shape[0]
        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        from miniworld_engine.kernels.transition.triton.main import swiglu_squeeze_backward

        x2, wa, wb, ws = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, ctx.shape[-1])
        if grad_output.stride(1) != 1:
            grad_output = grad_output.contiguous()
        return swiglu_squeeze_backward(x2, wa, wb, ws, grad_output, ctx.shape, ctx.nd)


def triton_swiglu_ffn(
    x: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor,
) -> torch.Tensor:
    """``W_down( SiLU(Wa @ x) * (Wb @ x) )`` -- a bare SwiGLU FFN, forward and backward.

    This is `Transition` with its LayerNorm and its residual taken away, which is exactly the
    FFN an adaLN-Zero block wants: that block normalizes with its own modulated RMSNorm and
    gates and adds the residual itself, so a Transition would apply a second, unwanted
    LayerNorm and a second residual.

    The expansion is read from ``expand_a_weight.shape[0]``, so a hidden size rounded to a
    multiple of 256 -- not to a multiple of ``d`` -- is fine; see `triton_transition`, which
    takes the same weights and is the fallback here.

    THE SPLIT PATH, not the back-to-back one, and that is a measured choice. b2b keeps the
    ``(M, ND)`` intermediate off HBM entirely -- 201 MB written and read at the atom shape,
    about 0.29 ms -- but pays for it: every program loops over ALL of ND and re-reads Wa/Wb,
    where the split runs two well-shaped GEMMs. At A=48, S=8192, d_atom 128, hidden 256, with
    a cache built for both (HAS_LN is in the b2b's autotune key, so the no-LayerNorm form is
    tuned as itself):

        eager PyTorch          fwd 1.69 ms  bwd 2.99  held 864 MB  peak 9622 MB
        split (this)           fwd 0.61     bwd 1.90  held  96 MB  peak 2039 MB
        b2b (_SwiGLUFFNFused)  fwd 1.36     bwd 1.89  held  96 MB  peak 2039 MB

    b2b loses the forward 2.2x and holds exactly the same memory -- h is recomputed in the
    backward either way, so never reaching HBM saves bandwidth, not held activation. The b2b
    path stays reachable through `_SwiGLUFFNFused` for a shape where that trade may turn
    (a much larger ND, where the h round-trip grows and the weight re-read does not).
    """
    from miniworld_engine.kernels.transition.triton.main import triton_transition

    return triton_transition(x, expand_a_weight, expand_b_weight, squeeze_weight)


# ---------------------------------------------------------------------------
# SEPARATE backward kernels. The expand recompute (a=xn@Wa, b=xn@Wb) is done ONCE in a
# single fused kernel that emits h (for dWs), dA and dB (the old path recomputed a,b TWICE
# — transition_fwd_kernel for h, transition_bwd_kernel for dA/dB). Plus the LayerNorm
# backward kernel. GEMMs (grad_expand, dWs, dWa, dWb, d_xn) stay cuBLAS.
# ---------------------------------------------------------------------------


# Cache-narrowing prune (no smem base prune on this kernel).


# fmt: off
@triton.autotune(configs=configs_for("transition_bwd_swiglu_recompute_triton"),
# STORE_H IS in the key: its guarded store is a whole extra (M, ND) tensor on a kernel whose
# other five stores are the outputs. STACK_DAB is NOT, and the difference is volume, not shape:
# both of its branches store the SAME two tiles with the same masks, into one 2*ND-wide buffer or
# two ND-wide ones, so only the addressing changes.
                 key=['shape_key', 'NORMALIZE', 'STORE_H'])
@triton.jit
def _transition_expand_gatebwd_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr, wa_ptr, wb_ptr, ge_ptr,
    h_ptr, dA_ptr, dB_ptr, dAB_ptr, xn_ptr,
    M, ND, K, shape_key,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_gm, stride_gn,    # grad_expand / h / dA / dB: (M, ND) row-major
    stride_abm, stride_abn,  # dAB: (M, 2*ND) row-major, [dA | dB]
    stride_nm, stride_nk,    # xn out: (M, K) row-major
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NORMALIZE: tl.constexpr, STORE_H: tl.constexpr, STACK_DAB: tl.constexpr,
    GROUP_M: tl.constexpr,
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
    # Visit order, tuned: see kernels/_tiles.py. Expand+gate backward. Launched from three sites, all with the same 2-D grid.
    pid_m, pid_n = tile_order(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, BLOCK_M1), tl.cdiv(ND, BLOCK_N), GROUP_M)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < M
    cmask = cols < ND
    if NORMALIZE:
        rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
        c1 = tl.load(c1_ptr + rows, mask=rmask, other=0.0)
    et = h_ptr.dtype.element_ty
    a = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    b = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
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


def _transition_expand_gatebwd_fake(x2, rstd, c1, gamma, beta, wa, wb, grad_expand,
                                    shape_key=None):
    """(h, dA, dB) each shaped like grad_expand (M, ND), plus the recomputed xn shaped like x2."""
    return (
        torch.empty_like(grad_expand),
        torch.empty_like(grad_expand),
        torch.empty_like(grad_expand),
        torch.empty_like(x2),
    )


@opaque(fake=_transition_expand_gatebwd_fake, name="transition_expand_gatebwd_recompute")
def _transition_expand_gatebwd(x2: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor,
                               gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor,
                               wb: torch.Tensor, grad_expand: torch.Tensor,
                               shape_key: int | None = None) -> tuple[
                                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Version A: normalize x inline + recompute a,b once -> (h, dA, dB, xn)."""
    M, K = x2.shape
    ND = wa.shape[0]
    h = torch.empty_like(grad_expand)
    dA = torch.empty_like(grad_expand)
    dB = torch.empty_like(grad_expand)
    xn = torch.empty_like(x2)
    grid = lambda meta: tile_grid(M, ND, meta["BLOCK_M1"], meta["BLOCK_N"])  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        x2, rstd, c1, gamma.contiguous(), beta.contiguous(),
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dA, dB, dA, xn,
        M, ND, K, _shape_key(shape_key, M, ND=ND, K=K),
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


def _transition_expand_gatebwd_stacked(x2, rstd, c1, gamma, beta, wa, wb, grad_expand,
                                       *, shape_key: int | None = None):
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
    grid = lambda meta: tile_grid(M, ND, meta["BLOCK_M1"], meta["BLOCK_N"])  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        x2, rstd, c1, gamma.contiguous(), beta.contiguous(),
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dAB, dAB, dAB, xn,
        M, ND, K, _shape_key(shape_key, M, ND=ND, K=K),
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


def _transition_expand_gatebwd_savedxn(xn, wa, wb, grad_expand, *, store_h: bool = True,
                                       shape_key: int | None = None):
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
    grid = lambda meta: tile_grid(M, ND, meta["BLOCK_M1"], meta["BLOCK_N"])  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        xn, xn, xn, xn, xn,          # rstd/c1/g/beta unused when NORMALIZE=False (pass xn as filler)
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dA, dB, dA, xn,           # dAB/xn_ptr unused — pass existing tensors as filler
        M, ND, K, _shape_key(shape_key, M, ND=ND, K=K),
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


def _transition_expand_gatebwd_savedxn_stacked_fake(xn, wa, wb, grad_expand, shape_key=None):
    """``h`` like ``grad_expand`` (M, ND), plus the stacked ``dAB`` (M, 2*ND) -- dA in the first ND
    columns, dB in the next -- which is the packing the two larger downstream GEMMs consume."""
    return (
        torch.empty_like(grad_expand),
        grad_expand.new_empty((xn.shape[0], wa.shape[0] * 2)),
    )


@opaque(fake=_transition_expand_gatebwd_savedxn_stacked_fake, name="transition_gatebwd_savedxn_stacked")
def _transition_expand_gatebwd_savedxn_stacked(
    xn: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    grad_expand: torch.Tensor,
    shape_key: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Version B stacked: emit h and dAB=[dA | dB] for two larger GEMMs downstream."""
    M, K = xn.shape
    ND = wa.shape[0]
    h = torch.empty_like(grad_expand)
    dAB = torch.empty(M, ND * 2, device=xn.device, dtype=grad_expand.dtype)
    grid = lambda meta: tile_grid(M, ND, meta["BLOCK_M1"], meta["BLOCK_N"])  # noqa: E731
    _transition_expand_gatebwd_kernel[grid](
        xn, xn, xn, xn, xn,
        wa.contiguous(), wb.contiguous(), grad_expand,
        h, dAB, dAB, dAB, xn,
        M, ND, K, _shape_key(shape_key, M, ND=ND, K=K),
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


# Cache-narrowing prune (no smem base prune). dtype from the input activation x_ptr (the
# 2nd pointer arg), NOT the dxn_ptr grad accumulator. reset_to_zero preserved below.


# fmt: off
# BLOCK_K is a CSV tile instead of the launcher's next_power_of_2(K). The row
# reductions (ca, cb) need ALL of K before dx can be formed, so the kernel walks K twice; at
# BLOCK_K >= K both sweeps are one iteration and the second reads an L2-hot row, i.e. the
# original single-tile schedule. reset_to_zero on dg/db is unchanged and still required (the
# dgamma/dbeta atomics accumulate across M-blocks AND across autotune trials).
# BLOCK_M1 comes from the CSV. A 1-row tile is not a tile, it is a per-row launch: it multiplies
# the grid by BLOCK_M1 and gives every reduction a one-element vector to sum -- keep CSV rows at
# or above 16.
# PRIVATIZE_DGDB is KEYED: it is a code path, not a shape. It picks between one dg/db
# accumulator (every program atomically adding to the same K columns) and NUM_REPLICAS strided
# private copies, and the whole reason it exists is that the contention it removes was the top
# stall (measured 1.31x @L=1024, ncu-confirmed membar). The two paths therefore want different
# BLOCK_M1/BLOCK_K -- the atomic path's cost scales with the number of M-blocks, the privatized
# one's does not. Both sides really do run: settings.transition_lnbwd_privatize defaults True and
# the autotune builder sweeps the off-default False side (builder.SWITCHES), so without this entry
# one bucket would hold both and serve each the other's winner.
#
# NUM_REPLICAS is NOT keyed: it is not free. It is the module-level constant
# _TRANSITION_LNBWD_PRIVATIZE_REPLICAS (64) when privatizing and 1 otherwise -- i.e. a function of
# PRIVATIZE_DGDB, and dead code on the non-privatized branch. Nothing (settings, env, launcher
# argument) can vary it independently, so PRIVATIZE_DGDB above already separates its two values.
@triton.autotune(configs=configs_for("layernorm_bwd_foldstats_triton"),
                 key=['shape_key', 'PRIVATIZE_DGDB'],
                 reset_to_zero=['dg_ptr', 'db_ptr'])
@triton.jit
def _transition_ln_bwd_kernel(
    dxn_ptr, x_ptr, rstd_ptr, c1_ptr, g_ptr, dx_ptr, dg_ptr, db_ptr,
    # K is tl.constexpr (model d, fixed per module, already in this kernel's autotune key) so the
    # `BLOCK_K >= K` guard below resolves at COMPILE time and only one branch is emitted.
    M, K: tl.constexpr, shape_key,
    stride_m, stride_k,
    dg_stride_replica, dg_stride_k,
    db_stride_replica, db_stride_k,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_REPLICAS: tl.constexpr, PRIVATIZE_DGDB: tl.constexpr,
):
    # LayerNorm backward consuming the SAVED stats (rstd, c1=mean*rstd), one pass over K:
    #   x_hat = x*rstd - c1 = (x-mean)*rstd ; wdy = gamma*dxn
    #   dx = rstd*(wdy - mean_k(wdy) - x_hat*mean_k(wdy*x_hat))
    #   dgamma += sum_m(dxn*x_hat) ; dbeta += sum_m(dxn)   (atomic over M-blocks)
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rows < M
    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=rmask, other=0.0)
    inv_k = 1.0 / K

    if BLOCK_K >= K:
        # COVERING TILE -> the pre-tiling single-pass schedule: dxn / x / gamma are read ONCE and
        # x_hat + wdy stay in registers for both the row reductions AND the dx epilogue, instead
        # of the two sweeps the general branch needs. Numerics are identical to the else-branch at
        # BLOCK_K >= K (its sweeps are single-trip and ca/cb start from an exact fp32 zero).
        k = tl.arange(0, BLOCK_K)
        kmask = k < K
        mask = rmask[:, None] & kmask[None, :]
        off = rows[:, None] * stride_m + k[None, :] * stride_k
        dxn = tl.load(dxn_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + k, mask=kmask, other=0.0).to(tl.float32)
        x_hat = tl.where(mask, x * rstd[:, None] - c1[:, None], 0.0)
        wdy = tl.where(mask, g[None, :] * dxn, 0.0)
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
    else:
        # pass A: the two row reductions over ALL of K.
        ca = tl.zeros([BLOCK_M1], dtype=tl.float32)
        cb = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < K
            mask = rmask[:, None] & kmask[None, :]
            off = rows[:, None] * stride_m + k[None, :] * stride_k
            dxn = tl.load(dxn_ptr + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + k, mask=kmask, other=0.0).to(tl.float32)
            x_hat = tl.where(mask, x * rstd[:, None] - c1[:, None], 0.0)
            wdy = tl.where(mask, g[None, :] * dxn, 0.0)
            ca += tl.sum(x_hat * wdy, axis=1)
            cb += tl.sum(wdy, axis=1)
        ca = ca * inv_k
        cb = cb * inv_k

        # pass B: dx, plus the dgamma/dbeta column partials.
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < K
            mask = rmask[:, None] & kmask[None, :]
            off = rows[:, None] * stride_m + k[None, :] * stride_k
            dxn = tl.load(dxn_ptr + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + k, mask=kmask, other=0.0).to(tl.float32)
            x_hat = tl.where(mask, x * rstd[:, None] - c1[:, None], 0.0)
            wdy = tl.where(mask, g[None, :] * dxn, 0.0)
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


def _transition_ln_bwd(dxn, x2, rstd, c1, gamma, *, shape_key: int | None = None):
    """LayerNorm backward from saved stats -> (dx, dgamma, dbeta)."""
    M, K = x2.shape
    # Fastest path: hand-CUDA warp-per-row LN backward (register column-partials -> no atomics/no
    # spill, persistent grid SM*WAVES, vectorized loads). ~1.10x over privatized-Triton at K=128
    # (326 vs 358us, cos 1.0). Needs bf16/fp16 + K<=512 + contiguous + weight dtype == x dtype.
    # CUDA takes `mean`; recover it from c1=mean*rstd. Any mismatch -> graceful Triton fallback.
    if (
        settings.current().transition_lnbwd_cuda
        and x2.dtype in (torch.float16, torch.bfloat16)
        and gamma.dtype == x2.dtype
        and K <= 512
        and x2.is_contiguous()
        and dxn.is_contiguous()
    ):
        try:
            from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

            mean = c1 / rstd
            return layer_norm_bwd_cuda(dxn, x2, gamma, mean, rstd)
        except Exception:  # noqa: BLE001 (build unavailable / dtype edge) -> Triton fallback
            pass
    dx = torch.empty_like(x2)
    # Default ON: privatizing dgamma/dbeta atomics across N replicas cuts atomic contention
    # (measured 1.31x @L=1024, ncu confirmed membar was the top stall). Set to 0/false/off to
    # restore the single-accumulator atomic path.
    privatize_dgdb = settings.current().transition_lnbwd_privatize
    if privatize_dgdb:
        num_replicas = _TRANSITION_LNBWD_PRIVATIZE_REPLICAS
        dgamma_acc = torch.zeros((num_replicas, K), device=x2.device, dtype=torch.float32)
        dbeta_acc = torch.zeros((num_replicas, K), device=x2.device, dtype=torch.float32)
    else:
        num_replicas = 1
        dgamma_acc = torch.zeros(K, device=x2.device, dtype=torch.float32)
        dbeta_acc = torch.zeros(K, device=x2.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    _transition_ln_bwd_kernel[grid](
        dxn, x2, rstd, c1, gamma.contiguous(), dx, dgamma_acc, dbeta_acc,
        M, K, _shape_key(shape_key, M, K=K), x2.stride(0), x2.stride(1),
        dgamma_acc.stride(0), dgamma_acc.stride(-1),
        dbeta_acc.stride(0), dbeta_acc.stride(-1),
        NUM_REPLICAS=num_replicas, PRIVATIZE_DGDB=privatize_dgdb,
    )
    if privatize_dgdb:
        dgamma = dgamma_acc.sum(dim=0)
        dbeta = dbeta_acc.sum(dim=0)
    else:
        dgamma = dgamma_acc
        dbeta = dbeta_acc
    return dx, dgamma, dbeta


def _fused_fwd_fake(x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight,
                    squeeze_weight, n, eps, save_xn, add_residual, shape_key):
    """Shapes only. Branches on ``save_xn`` (an argument) and never on the device, because a
    fake has to give the same STRUCTURE the compiled graph was traced with -- which of the
    b2b / split / cute paths below actually runs must not be visible from here."""
    m = x2.shape[0]
    return (
        x2.new_empty((m, squeeze_weight.shape[0])),
        x2.new_empty((m,), dtype=torch.float32),   # rstd
        x2.new_empty((m,), dtype=torch.float32),   # c1
        x2.new_empty(x2.shape) if save_xn else x2.new_empty((0, 0)),
    )


@opaque(fake=_fused_fwd_fake, name="transition_fused_fwd")
def _fused_fwd(
    x2: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor,
    n: int,
    eps: float,
    save_xn: bool,
    add_residual: bool,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every launch of the fused forward, and nothing else -> ``(out, rstd, c1, xn)``.

    Split out of ``TritonTransitionFusedFunction.forward`` so the flatten, the autocast casts,
    ``save_for_backward`` and the output reshape stay traceable while the arch dispatch below --
    device-capability queries, ``try``/``except`` build fallbacks, Triton and CuTeDSL launches --
    stays opaque. See ``kernels._compile``. ``x2`` arrives 2-D, contiguous and already cast.

    ``rstd``/``c1`` are returned, not merely saved, because the backward needs the LN stats and
    an op can only hand tensors back through its return. ``xn`` exists only when ``save_xn``; a
    schema cannot return ``None``, so the other case returns an EMPTY tensor and the caller --
    which knows ``save_xn`` -- is the one that decides whether to save it.
    """
    K = x2.shape[-1]
    # Fuse the post-transition residual add y = transition(x) + x into the forward output
    # (the residual is the module input x itself; D == K). Handled in-kernel on the fast
    # b2b paths; an explicit add on the fallback paths. Backward adds grad_output back to
    # dx (the identity path of x + f(x)). ``residual_pending`` stays True until a path
    # has folded the add.
    residual_pending = add_residual

    xn = None
    # Memory-light training path: reuse the FAST inference hand-CUDA b2b forward (fused
    # squeeze, h never in HBM) for d<=256 where it fits smem (d=128 ~1.29x vs triton,
    # d=256 confirmed). Version A (save_xn=False) saves no xn, so the shape-general
    # backward recomputes it — the CUDA kernel emits nothing beyond `out`. stats are needed
    # by the backward anyway, so stats_triton is not extra work. This gate is INDEPENDENT of
    # _B2B_MAX_K (the triton-b2b smem bound); on any failure we fall back to the split.
    _cap_major = torch.cuda.get_device_capability(x2.device)[0]
    _is_sm100 = _cap_major == 10  # noqa: PLR2004
    _is_sm90 = _cap_major == 9  # noqa: PLR2004  Hopper exactly (WGMMA/TMA hand-CUDA b2b)
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
            # Hand-CUDA b2b is Hopper (sm_90a) WGMMA/TMA -> gate on sm_90 exactly.
            # On pre-Hopper (sm_80 / A100) this must be False so we fall through to the
            # portable triton b2b (K<=128) / split (else) path instead of attempting a
            # Hopper-only kernel that can't launch here (was a per-call failed-build cost).
            or (_is_sm90 and _cuda_b2b_train_enabled() and K in (128, 256))
        )
    )
    if cuda_b2b_ok:
        rstd, c1 = stats_triton(x2, eps, shape_key=shape_key)
        out = None
        if torch.cuda.get_device_capability(x2.device)[0] == 10:
            # B200 sm_100: the hand-CUDA sm90 b2b can't build (Hopper wgmma/TMA); use the
            # cutlass-DSL sm100 forward. Version A backward (below) is arch-agnostic and
            # recomputes xn from the saved stats, so it works unchanged with this forward.
            try:
                from miniworld_engine.kernels.transition.cute.b2b_fwd_sm100 import (
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
                from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd
                out = transition_b2b_fwd(
                    x2, rstd, c1,
                    ln_weight.contiguous(), ln_bias.contiguous(),
                    expand_a_weight.contiguous(), expand_b_weight.contiguous(),
                    squeeze_weight.contiguous(),
                    add_residual=residual_pending,
                )
                residual_pending = False  # folded into the squeeze epilogue
            except Exception:  # noqa: BLE001  build unavailable -> split fallback (always fits)
                expand = transition_expand_gate(
                    x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, eps,
                    stats=(rstd, c1), save_xn=False, shape_key=shape_key,
                )
                out = torch.matmul(expand, squeeze_weight.T)
    elif K <= _B2B_MAX_K:
        # Back-to-back fused (triton): squeeze folded in, h never materialized in HBM.
        if _transition_fuse_stats_enabled():
            res = transition_b2b(
                x2, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight, eps,
                save_xn=save_xn, fuse_stats=True, add_residual=residual_pending,
                shape_key=shape_key,
            )
            residual_pending = False  # folded into the squeeze epilogue
            if save_xn:
                out, rstd, c1, xn = res
            else:
                out, rstd, c1 = res
        else:
            # LN stats computed once and reused: by the forward kernel AND saved for the
            # separate backward (so backward never recomputes mean/rstd).
            rstd, c1 = stats_triton(x2, eps, shape_key=shape_key)
            res = transition_b2b(
                x2, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight, eps,
                stats=(rstd, c1), save_xn=save_xn, fuse_stats=False,
                add_residual=residual_pending, shape_key=shape_key,
            )
            residual_pending = False  # folded into the squeeze epilogue
            out, xn = res if save_xn else (res, None)
    else:
        # Large K (K > _B2B_MAX_K). The full-K-row expand kernel loads BLOCK_K =
        # next_pow2(K) and OOMs smem at d>=256 on small-smem GPUs (e.g. A100, 163KB);
        # the K-tiled b2b keeps weight tiles [BLOCK_K, BLOCK_N] bounded at any d AND
        # fuses the squeeze (matching the K-tiled backward), so prefer it for the
        # forward-only path. save_xn=True still needs the materialized xn for its
        # stacked backward, so that legacy path keeps the expand + cuBLAS squeeze.
        rstd, c1 = stats_triton(x2, eps, shape_key=shape_key)
        if not save_xn:
            out = transition_b2b_ktiled(
                x2, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight, eps,
                shape_key=shape_key,
            )
            xn = None
        else:
            expand, xn = transition_expand_gate(
                x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, eps,
                stats=(rstd, c1), save_xn=True, shape_key=shape_key,
            )
            out = torch.matmul(expand, squeeze_weight.T)

    if residual_pending:
        # Fallback paths (sm100 cute fwd, split GEMM, build-unavailable) that did not fold
        # the residual in-kernel: add it explicitly. y = transition(x) + x, D == K.
        out = out + x2
        residual_pending = False
    return out, rstd, c1, (xn if xn is not None else x2.new_empty((0, 0)))


def _fused_bwd_fake(grad_output, x2, rstd, c1, ln_weight, ln_bias, expand_a_weight,
                    expand_b_weight, squeeze_weight, xn_saved, eps, has_xn, add_residual,
                    orig_shape, shape_key):
    """Shapes only -- the six real gradients, in ``forward``'s argument order."""
    return (
        grad_output.new_empty(tuple(orig_shape), dtype=x2.dtype),
        torch.empty_like(ln_weight),
        torch.empty_like(ln_bias),
        torch.empty_like(expand_a_weight),
        torch.empty_like(expand_b_weight),
        torch.empty_like(squeeze_weight),
    )


@opaque(fake=_fused_bwd_fake, name="transition_fused_bwd")
def _fused_bwd(
    grad_output: torch.Tensor,
    x2: torch.Tensor,
    rstd: torch.Tensor,
    c1: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor,
    xn_saved: torch.Tensor | None,
    eps: float,
    has_xn: bool,
    add_residual: bool,
    orig_shape: list[int],
    shape_key: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """The fused backward -> ``(dx, dgamma, dbeta, dWa, dWb, dWs)``.

    Split out of ``TritonTransitionFusedFunction.backward`` for the reason the forward is: the
    saved-tensor unpack stays traceable, the launches stay opaque. It returns only the six real
    gradients -- a ``torch.library`` schema cannot return ``None`` -- and the caller re-adds the
    four ``None`` slots that ``n``, ``eps``, ``save_xn`` and ``add_residual`` need.
    """
    # SEPARATE (non-fused) backward: explicit per-stage ops, reusing the LN stats
    # (rstd, c1) saved by forward (no mean/rstd recompute). GEMMs are bf16 (matching
    # forward); gate/LN math accumulates in fp32.
    #   out = h @ Ws^T;  h = silu(a)*b;  a = xn@Wa^T, b = xn@Wb^T;  xn = (x-mean)*rstd*g+beta

    def _finalize_dx(dx_flat):
        # y = x + f(x): the residual identity path contributes grad_output directly to dx.
        dxr = dx_flat.reshape(orig_shape)
        if add_residual:
            # In-place: reuse dx's freshly-computed storage (never saved/aliased) instead of
            # allocating a new M×D buffer. Matches the unfused AddBackward, which passes
            # grad_output through without a new buffer -> fusion stays memory-neutral.
            dxr = dxr.add_(grad_output.reshape(orig_shape).to(dxr.dtype))
        return dxr

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
    if (not has_xn) and torch.cuda.get_device_capability(x2.device)[0] == 10:
        _sm100_ok = True
        try:
            from miniworld_engine.kernels.transition.cute.gatebwd_sm100 import (
                transition_expand_gatebwd_sm100,
            )
        except Exception:  # noqa: BLE001
            _sm100_ok = False
        if _sm100_ok:
            # Recompute xn with the tuned LN kernel (~13µs, matches the forward's
            # transition_b2b_sm100_ln); a torch stats-formula recompute is ~15x slower
            # (fp32 intermediates + many passes).
            from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
            xn = layernorm_kernel(x2, ln_weight, ln_bias, eps)
            h, dA, dB = transition_expand_gatebwd_sm100(
                xn, expand_a_weight.contiguous(), expand_b_weight.contiguous(),
                grad_expand, shape_key=shape_key,
            )
            dWs = go.t() @ h
            dWa = dA.t() @ xn
            dWb = dB.t() @ xn
            d_xn = dA @ expand_a_weight + dB @ expand_b_weight
            dx, dgamma, dbeta = _transition_ln_bwd(d_xn, x2, rstd, c1, ln_weight,
                                                  shape_key=shape_key)
            return (
                _finalize_dx(dx),
                dgamma.to(ln_weight.dtype),
                dbeta.to(ln_bias.dtype),
                dWa, dWb, dWs,
            )

    # (2) gate backward is the ONLY stage that differs between Version A/B:
    #   B (has_xn):   reuse the saved xn (no re-normalize).
    #   A (recompute): re-normalize x from saved stats inline.
    # Both emit stacked dAB=[dA|dB] so stages (3)(4)(5)(6) below are version-INDEPENDENT.
    # use_savedxn_split_bwd() keeps the old split comparator (has_xn only, default off).
    if has_xn and use_savedxn_split_bwd():
        h, dA, dB = _transition_expand_gatebwd_savedxn(
            xn_saved, expand_a_weight, expand_b_weight, grad_expand, shape_key=shape_key,
        )
        xn = xn_saved
        dWs = go.t() @ h
        dWa = dA.t() @ xn
        dWb = dB.t() @ xn
        d_xn = dA @ expand_a_weight + dB @ expand_b_weight
        dx, dgamma, dbeta = _transition_ln_bwd(d_xn, x2, rstd, c1, ln_weight,
                                              shape_key=shape_key)
        return (
            _finalize_dx(dx),
            dgamma.to(ln_weight.dtype),
            dbeta.to(ln_bias.dtype),
            dWa, dWb, dWs,
        )

    if has_xn:
        h, dAB = _transition_expand_gatebwd_savedxn_stacked(
            xn_saved, expand_a_weight, expand_b_weight, grad_expand, shape_key=shape_key,
        )
        xn = xn_saved
    elif (
        _gatebwd_wgmma_enabled()
        and torch.cuda.get_device_capability(x2.device)[0] == 9  # Hopper (sm_90a)
        and K in (256, 512)
        and x2.dtype == torch.bfloat16
        and x2.shape[0] % 128 == 0
    ):
        # sm90 hand-CUDA WGMMA fused expand + gate-backward: beats the Triton recompute at
        # large d (d=256 ~1.02-1.05x, d=512 ~1.07-1.18x; d=128 stays Triton where it wins).
        # Falls back to Triton on any build/launch failure.
        try:
            from miniworld_engine.kernels.transition.cuda import (
                transition_expand_gatebwd_wgmma,
            )
            h, dAB, xn = transition_expand_gatebwd_wgmma(
                x2, rstd, c1, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, grad_expand,
            )
        except Exception:  # noqa: BLE001  build/launch unavailable -> Triton
            h, dAB, xn = _transition_expand_gatebwd_stacked(
                x2, rstd, c1, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, grad_expand, shape_key=shape_key,
            )
    else:
        h, dAB, xn = _transition_expand_gatebwd_stacked(
            x2, rstd, c1, ln_weight, ln_bias,
            expand_a_weight, expand_b_weight, grad_expand, shape_key=shape_key,
        )

    # (3)(4)(5)(6) SHARED across versions (stacked): two larger GEMMs + LN bwd.
    dWs = go.t() @ h                          # (3) [D, ND]
    w_ab = torch.cat((expand_a_weight, expand_b_weight), dim=0)
    dWab = dAB.t() @ xn                        # (4) fuses dWa,dWb -> [2*ND, K]
    # Both halves are returned, and a torch.library custom operator may not have one output alias
    # another ("may not alias any inputs to this custom operator OR OTHER RETURNS"). As slices of
    # one dWab they shared a storage, so every training backward raised RuntimeError under
    # settings.compile_wrap="custom_op" -- the default since 1e4c24b -- and the transition rows of
    # a bench run failed with it. Cloning ONE half is enough: it is what makes the two storages
    # distinct, and it keeps the fused [2*ND, K] wgrad GEMM that (4) exists for. `.contiguous()`
    # would not do it -- a dim-0 slice of a 2-D tensor already is contiguous and returns self.
    dWa = dWab[: expand_a_weight.shape[0]].clone()
    dWb = dWab[expand_a_weight.shape[0] :]
    if (
        settings.current().transition_dab_lnbwd
        and K <= 128
        and torch.cuda.get_device_capability(x2.device)[0] >= 9
    ):
        from miniworld_engine.kernels.transition.cute.dab_lnbwd import (
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
            _finalize_dx(dx),
            dgamma.to(ln_weight.dtype),
            dbeta.to(ln_bias.dtype),
            dWa, dWb, dWs,
        )
    d_xn = dAB @ w_ab                          # (5) fuses dA@Wa + dB@Wb -> [M, K]

    # (6) LayerNorm backward from saved stats -> dx, dgamma, dbeta (hand-CUDA at d<=512 bf16).
    dx, dgamma, dbeta = _transition_ln_bwd(d_xn, x2, rstd, c1, ln_weight,
                                           shape_key=shape_key)
    return (
        _finalize_dx(dx),
        dgamma.to(ln_weight.dtype),
        dbeta.to(ln_bias.dtype),
        dWa, dWb, dWs,
    )


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
        add_residual: bool = False,
    ) -> Float[torch.Tensor, "... d"]:
        orig_shape = x.shape
        K = orig_shape[-1]
        # L for the autotune shape key: shape[-2] of the activation BEFORE the flatten --
        # one rule for pair (B, L, L, D) and token/atom (B, L, D). Threaded into every
        # launcher below (and saved for the backward), so no launcher buckets a row count.
        shape_key = both_key(rows_of(orig_shape))
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
        out, rstd, c1, xn = _fused_fwd(
            x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight,
            n, eps, save_xn, add_residual, shape_key,
        )
        if not save_xn:
            xn = None   # the op returns an empty placeholder; only save_xn makes it real
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
        ctx.shape_key = shape_key
        ctx.add_residual = add_residual
        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
        dx, dgamma, dbeta, dWa, dWb, dWs = _fused_bwd(
            grad_output, x2, rstd, c1, ln_weight, ln_bias,
            expand_a_weight, expand_b_weight, squeeze_weight, xn_saved,
            ctx.eps, ctx.has_xn, ctx.add_residual, list(ctx.orig_shape), ctx.shape_key,
        )
        # n, eps, save_xn, add_residual take no gradient.
        return dx, dgamma, dbeta, dWa, dWb, dWs, None, None, None, None


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
    add_residual: bool = False,
) -> torch.Tensor:
    """Fully fused Transition forward (LN folded in).

    ``save_xn`` selects the backward version: False (default) = Version A (recompute xn in
    backward, less memory); True = Version B (save xn in forward, reuse in backward).

    ``add_residual`` folds the post-transition residual add ``y = transition(x) + x`` into the
    forward output (fused in-kernel on the b2b paths); the backward returns the identity
    contribution to ``dx``. The caller must then NOT add the residual again outside.
    """
    return TritonTransitionFusedFunction.apply(
        x, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps,
        save_xn, add_residual,
    )
