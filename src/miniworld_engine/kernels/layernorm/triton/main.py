
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/layernorm.py
from miniworld_engine.autotune.configs import configs_for
import os

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune.shape_key import both_key, length_of, rows_of
from miniworld_engine import settings


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_mixed
    return bucket_mixed(length)


# Opt-in: route the LN backward (dx/dw/db) through the hand-CUDA warp-per-row kernel for the regime
# where it beats the triton atomic path STANDALONE (bf16, 128<=N<=512, contiguous, no per-row
# scale) — measured ~1.1x over the triton atomic bwd on B200. DEFAULT OFF because the win is
# e2e-NEUTRAL in the trimul training graph: LN_in bwd is ~7% of the step and the CUDA path's second
# (reduce) kernel launch eats the ~12% compute win at these M (verified graph-time delta within
# run-to-run noise at L=384/768/1024). Enabling also forces the one-time nvcc JIT build on first
# `triton_layernorm` import. Set settings.layernorm_cuda_bwd to use it (kernels/layernorm/cuda/ now
# gencodes sm_80/90/100 + PTX so the ext loads and runs on B200).
def _ln_cuda_bwd_enabled() -> bool:
    """Read at call time; see compile_native._ln_bwd_override."""
    return settings.current().layernorm_cuda_bwd






# The two shipped kernels (`layer_norm_fwd_fused`, `layer_norm_bwd_dx_fused`) tile the FEATURE
# axis too. It used to arrive as BLOCK_N = triton.next_power_of_2(N) from every launch site — the
# whole row, a caller-computed constant the tuner never saw, and the reason BLOCK_M1 was the only
# thing being swept. It is the REDUCE axis (mean/var in the forward; the c1/c2 row sums and the
# dw/db column sums in the backward), so it is a CSV tile: N is
# d_hidden/d_pair at 128..1024 while BLOCK_N stops at 256, and a sweep that cannot express a whole
# row would turn "tiled" into "always two passes over X", which the tuner could not decline.
# still takes BLOCK_N from its caller.


# fmt: off


# `eps` is constexpr but deliberately NOT keyed: it only appears in `1 / sqrt(var + eps)`, so it
# branches nothing and shifts no work -- keying it would just multiply the bucket count.
@triton.autotune(configs=configs_for("layernorm_fwd_saveact_triton"),
                 key=['N', 'shape_key', 'HAS_ROWSCALE'])
@triton.jit
def layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, Rowscale,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_ROWSCALE: tl.constexpr,
):
    # Map the program id to the rows of X and Y it should compute.
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M

    # TWO-PASS (not Welford): pass 1 accumulates Σx and Σx² over the N tiles in fp32 — both are
    # plain sums, so tiling them is exact — and pass 2 re-reads X to normalize. LayerNorm re-uses
    # the row it just reduced, so a tiled reduce axis costs either a second read of X or a Welford
    # carry; the re-read is far simpler.
    # The old `var -= (BLOCK_K - N)/N * mean*mean` fixup is gone with it: it existed only because
    # the padded lanes were centred by `x - mean` without a mask. Here every tile is masked, so the
    # tail columns contribute exactly nothing to Σx or Σx².
    #
    # COVERING TILE (BLOCK_K >= N): both `for` loops would be single-trip, but the two tl.loads of
    # X are NOT CSE'd — the tl.store of Mean/Rstd sits between them and Triton cannot prove the raw
    # pointers do not alias — so the "collapses back to one read" claim was false: the kernel did
    # read X, read X, write Y (3 HBM passes instead of 2). Both N and BLOCK_K are tl.constexpr, so
    # the guard below is resolved at TRACE time and only ONE branch is ever emitted; the covering
    # tile degenerates to the pre-tiling single-pass schedule (read X once, keep it in registers).
    # The fast path uses the CENTRED variance Σ(x-mean)²/N — what the pre-tiling kernel used, and
    # numerically stabler — because x is already live. The uncentered Σx²/N - mean² form is kept in
    # the tiled branch, where it exists precisely so each tile is read exactly once.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                    mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        xc = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / N
        rstd = 1 / tl.sqrt(var + eps)

        tl.store(Mean + rows, mean, mask=row_mask)
        tl.store(Rstd + rows, rstd, mask=row_mask)

        w = tl.load(W + cols, mask=col_mask, other=0.0)
        b = tl.load(B + cols, mask=col_mask, other=0.0)
        y = xc * rstd[:, None] * w + b
        if HAS_ROWSCALE:  # fold a per-row scale (e.g. AF pair-mask) into the LN epilogue — free
            rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)
            y = y * rs[:, None]
        tl.store(Y + rows[:, None] * stride_r + cols[None, :] * stride_c, y, mask=mask)
    else:
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            mask = row_mask[:, None] & (cols[None, :] < N)
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            s += tl.sum(x, axis=1)
            ss += tl.sum(x * x, axis=1)
        mean = s / N
        var = ss / N - mean * mean
        rstd = 1 / tl.sqrt(var + eps)

        # Write mean / rstd
        tl.store(Mean + rows, mean, mask=row_mask)
        tl.store(Rstd + rows, rstd, mask=row_mask)

        if HAS_ROWSCALE:  # fold a per-row scale (e.g. AF pair-mask) into the LN epilogue — free
            rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)

        # Normalize and apply linear transformation
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=col_mask, other=0.0)
            b = tl.load(B + cols, mask=col_mask, other=0.0)
            x_hat = (x - mean[:, None]) * rstd[:, None]
            y = x_hat * w + b
            if HAS_ROWSCALE:
                y = y * rs[:, None]
            tl.store(Y + rows[:, None] * stride_r + cols[None, :] * stride_c, y, mask=mask)
# fmt: on


# fmt: off


# fmt: off


# HAS_ROWSCALE is keyed: it is constexpr, so the two variants already compile separately, but the
# autotune cache is keyed only on `key=[...]` -- without it the tile measured on the dense path is
# reused by the masked one, which does strictly more work per row.
@triton.autotune(configs=configs_for("layernorm_bwd_atomic_triton"),
                 key=['N', 'shape_key', 'HAS_ROWSCALE'],
                 reset_to_zero=['DW', 'DB'])
@triton.jit
def layer_norm_bwd_dx_fused(
    DX, DY, DW, DB,
    X, W, Mean, Rstd, Rowscale,
    stride_wc, stride_bc, stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_ROWSCALE: tl.constexpr,
):
    # Map the program id to the rows of X, DX, and DY it should compute.
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M
    mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:  # bwd of y = LN(x)*rs: scale incoming grad by rs (then dx/dw/db all follow)
        rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)

    # TWO-PASS over the N tiles. dx needs c1/c2, which reduce over the WHOLE row, so once the
    # feature axis is tiled the (x, dy) tiles must be visited twice. dw/db do NOT depend on c1/c2
    # (they reduce over M, per column), so their atomics are issued in pass 1 and pass 2 only
    # re-reads x/dy and writes dx. dw/db partials are fp32 (w is cast to fp32 before use),
    # unchanged from the untiled kernel; `reset_to_zero=[DW, DB]` is kept because the autotuner
    # re-runs every candidate against these accumulators.
    #
    # COVERING TILE (BLOCK_K >= N): the two loops are single-trip but the X/DY/W loads are NOT
    # CSE'd across them (the dw/db tl.atomic_add sits between, and Triton cannot prove the raw
    # pointers do not alias), so the covering config paid 2x the read traffic instead of collapsing
    # to the untiled schedule. N and BLOCK_K are both tl.constexpr, so this guard is resolved at
    # TRACE time and only one branch is emitted.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                    mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     mask=mask, other=0).to(tl.float32)
        if HAS_ROWSCALE:
            dy = dy * rs[:, None]
        w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        wdy = tl.where(mask, w[None, :] * dy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=1) / N
        c2 = tl.sum(wdy, axis=1) / N
        # Accumulate partial sums for dw/db (column reduction over this row tile)
        tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
        tl.atomic_add(DB + cols, tl.sum(dy, axis=0), mask=col_mask)
        dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
        tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0).to(tl.float32)
            if HAS_ROWSCALE:
                dy = dy * rs[:, None]
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            wdy = tl.where(mask, w[None, :] * dy, 0.0)
            c1 += tl.sum(xhat * wdy, axis=1)
            c2 += tl.sum(wdy, axis=1)
            # Accumulate partial sums for dw/db (column reduction over this row tile)
            tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
            tl.atomic_add(DB + cols, tl.sum(dy, axis=0), mask=col_mask)
        c1 = c1 / N
        c2 = c2 / N

        # Compute + write dx
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0).to(tl.float32)
            if HAS_ROWSCALE:
                dy = dy * rs[:, None]
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            wdy = tl.where(mask, w[None, :] * dy, 0.0)
            dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)
# fmt: on


def _ln_fwd_fake(x_2d, weight, bias, rs, eps, has_rs, shape_key):
    m = x_2d.shape[0]
    return (
        torch.empty_like(x_2d),
        x_2d.new_empty((m,), dtype=torch.float32),   # mean
        x_2d.new_empty((m,), dtype=torch.float32),   # rstd
    )


@opaque(fake=_ln_fwd_fake, name="layernorm_fwd")
def _ln_fwd(
    x_2d: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    rs: torch.Tensor | None,
    eps: float,
    has_rs: bool,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The launch: ``y = LN(x) * rs`` -> ``(y, mean, rstd)``. ``x_2d`` arrives flat and contiguous.

    Split out of ``TritonLayerNormFunction.forward`` so the reshape, the ``row_scale`` flattening
    and ``save_for_backward`` stay traceable -- see ``kernels._compile``. ``rs`` is None on the
    dense path; the kernel wants a real pointer there, so ``rstd`` stands in as the placeholder it
    never reads (``HAS_ROWSCALE=False``).
    """
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    mean = torch.empty(M, dtype=torch.float32, device=x_2d.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x_2d.device)
    # fmt: off
    grid = lambda META: [triton.cdiv(M, META["BLOCK_M1"])]
    layer_norm_fwd_fused[grid](
        x_2d, y_2d, weight, bias, mean, rstd, rs if has_rs else rstd,
        x_2d.stride(0), x_2d.stride(1),
        M, N, eps,
        shape_key=shape_key, HAS_ROWSCALE=has_rs,
    )
    # fmt: on
    return y_2d, mean, rstd


class TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
        row_scale: torch.Tensor | None = None,
    ):
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        has_rs = row_scale is not None
        # per-row scale (e.g. AF pair-mask) folded into the LN epilogue — y = LN(x)*rs, FREE
        # (no extra (M,N) multiply / HBM round-trip). rs reshaped to [M], broadcast over N.
        rs = row_scale.reshape(-1).to(x_2d.dtype).contiguous() if has_rs else None
        y_2d, mean, rstd = _ln_fwd(
            x_2d, weight, bias, rs, eps, has_rs,
            # L = x.shape[-2], read BEFORE the reshape above -- one rule for pair
            # (B, L, L, D) and token/atom (B, L, D). Never the row count M.
            both_key(rows_of(x.shape)),
        )

        ctx.save_for_backward(
            x_2d.to(torch.bfloat16),
            weight,
            mean,
            rstd,
            rs if has_rs else None,
        )
        ctx.has_rowscale = has_rs
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd, rs = ctx.saved_tensors
        dx, dw, db = _ln_bwd(
            dy.reshape(-1, dy.shape[-1]).contiguous(), x, weight, mean, rstd, rs,
            ctx.has_rowscale, list(ctx.input_shape),
            # ctx.input_shape is the forward's PRE-flatten shape; L = its [-2].
            both_key(rows_of(ctx.input_shape)),
        )
        return dx, dw, db, None, None   # eps, row_scale take no gradient


def _ln_bwd_fake(dy_2d, x, weight, mean, rstd, rs, has_rs, input_shape, shape_key):
    n = x.shape[-1]
    return (
        dy_2d.new_empty(tuple(input_shape)),
        x.new_empty((n,), dtype=torch.float32),
        x.new_empty((n,), dtype=torch.float32),
    )


@opaque(fake=_ln_bwd_fake, name="layernorm_bwd")
def _ln_bwd(
    dy_2d: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    rs: torch.Tensor | None,
    has_rs: bool,
    input_shape: list[int],
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward launches -> ``(dx, dweight, dbias)``.

    Split out of ``TritonLayerNormFunction.backward`` (see ``kernels._compile``). It returns only
    the three real gradients -- a ``torch.library`` schema cannot return ``None``, so the caller
    re-adds the two ``None`` slots that ``eps`` and ``row_scale`` need. Both the hand-CUDA and the
    Triton path live in here, so which one a card takes is invisible to the compiled graph -- as it
    must be, since the fake cannot know.
    """
    x = x.to(dy_2d.dtype)
    M, N = x.shape

    # Hand-CUDA warp-per-row backward (register column-partials, no atomics/shared/spill).
    # It now supports the row_scale (AF pair-mask) fold: the CUDA kernel scales the incoming
    # grad by rs per row (dx/dw/db follow, cos=1.0 vs triton). Measured B200 (M=L², N=128,
    # fwd+bwd of triton_layernorm):
    #   MASKED (has_rs): CUDA beats triton 1.17x @L512, 1.28x @L1024 — triton pays a real
    #                    rowscale penalty (+26% bwd) that the CUDA path (one FMA) avoids.
    #   DENSE  (no rs):  ~neutral (1.00x @L512, 1.07x @L1024) and slightly SLOWER at L384.
    # So auto-take CUDA for the masked path (always wins), but keep dense behind the opt-in
    # env flag. Lazy import so the nvcc JIT build only triggers when this path is taken; any
    # build/run failure falls through to triton.
    if (x.dtype == torch.bfloat16 and 128 <= N <= 512 and (has_rs or _ln_cuda_bwd_enabled())):
        try:
            from ..cuda import layer_norm_bwd_cuda
            dx_c, dw_c, db_c = layer_norm_bwd_cuda(
                dy_2d, x.contiguous(), weight, mean, rstd,
                row_scale=rs if has_rs else None,
            )
            return (dx_c.view(tuple(input_shape)), dw_c.float(), db_c.float(),
                    None, None)
        except Exception:  # noqa: BLE001 - portable triton fallback on any CUDA-path failure
            pass

    # allocate output
    dx_2d = torch.empty_like(dy_2d)
    dw = torch.zeros(N, dtype=torch.float32, device=x.device)
    db = torch.zeros(N, dtype=torch.float32, device=x.device)

    # fmt: off
    grid = lambda META: [triton.cdiv(M, META["BLOCK_M1"])]
    layer_norm_bwd_dx_fused[grid](
        dx_2d, dy_2d, dw, db,
        x, weight, mean, rstd, rs if has_rs else rstd,  # rs folds the mask grad in (free)
        dw.stride(0), db.stride(0), x.stride(0), x.stride(1),
        M, N,
        shape_key=shape_key, HAS_ROWSCALE=has_rs,
    )
    # fmt: on

    return dx_2d.view(tuple(input_shape)), dw, db


def triton_layernorm(x, weight, bias, eps, row_scale=None):
    """LayerNorm (autograd). Optional `row_scale` [M] folds a per-row scale into the LN epilogue
    (fwd) and the grad into the LN backward (bwd) — y = LN(x)*rs, the AF pair-mask applied FREE
    (no separate (M,N) multiply). rs=None -> plain LN."""
    return TritonLayerNormFunction.apply(x, weight, bias, eps, row_scale)


@opaque(fake=lambda x, weight, bias, eps, row_scale: torch.empty_like(x),
        name="layernorm_masked_fwd")
def triton_layernorm_masked(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    row_scale: torch.Tensor,
) -> torch.Tensor:
    """Forward-only LN with a per-row scale folded into the epilogue (free) — for the
    AF triangle pair-mask: y = LN(x) * row_scale[row]. row_scale is [M] (or anything
    reshapeable to [M]); broadcast over the feature dim. No autograd (bench/inference)."""
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    y_2d = torch.empty_like(x_2d)
    M, N = x_2d.shape
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    rs = row_scale.reshape(-1).to(x_2d.dtype).contiguous()
    # fmt: off
    grid = lambda META: [triton.cdiv(M, META["BLOCK_M1"])]
    layer_norm_fwd_fused[grid](
        x_2d, y_2d, weight, bias, mean, rstd, rs,
        x_2d.stride(0), x_2d.stride(1),
        M, N, eps,
        shape_key=both_key(rows_of(x.shape)), HAS_ROWSCALE=True,
    )
    # fmt: on
    return y_2d.view_as(x)
