"""RMSNorm forward and backward, structured after ``layernorm/triton/main.py``.

RMSNorm is LayerNorm without the mean: ``y = x * rstd * w`` with
``rstd = 1/sqrt(mean(x^2) + eps)``. The kernels here are the layernorm ones with the mean removed,
which takes four things out and leaves the schedule alone:

  * no ``Mean`` buffer, and no store of it -- the backward reloads only ``rstd``. That is 4 bytes
    a row less held between forward and backward.
  * no centring, so ``xhat = x * rstd`` reads x directly where LayerNorm needs ``x - mean``.
  * no bias term and no ``db``.
  * the backward loses its ``c2``. For LayerNorm ``dx = (wdy - (xhat*c1 + c2)) * rstd``; here the
    derivative of ``rstd`` has only the one term, so ``dx = (wdy - xhat*c1) * rstd`` with
    ``c1 = sum(xhat*wdy)/N``.

What is kept, because it was measured there and the shape regime is the same kind:

  * TWO-PASS over the N tiles rather than Welford -- ``sum(x^2)`` is a plain sum, so tiling it is
    exact, and the second read of x is simpler than a Welford carry.
  * a COVERING-TILE branch (``BLOCK_K >= N``) that keeps x in registers and never re-reads it.
    Both N and BLOCK_K are ``tl.constexpr``, so only one branch is ever emitted.
  * both tile axes tuned, and the row statistics kept in fp32 whatever the activation dtype.

``HAS_WEIGHT`` is new. The two callers disagree: ``modules/swa_atom_attention`` normalizes q and k
with no learnable weight, while ``kernels/triangle_attention/whole_op.py`` passes one. A constexpr
branch serves both without a dummy ones-vector, which would otherwise cost a full N-wide load and
multiply per row for the caller that has no weight.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key, pack, rows_of
from miniworld_engine.kernels._compile import opaque


@triton.autotune(configs=configs_for("rmsnorm_fwd_triton"), key=["shape_key", "HAS_WEIGHT"])
@triton.jit
def rmsnorm_fwd_kernel(
    X, Y, W, Rstd,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr,
):
    """``x / sqrt(mean(x^2) + eps) (* w)`` over the last axis. No modulation: the adaLN modulate
    that folds a projection into this step is `rmsnorm_adamod`, its own family; this one is the
    bare normalization the SWA q/k and triangle-attention call sites want."""
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M

    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                    mask=mask, other=0.0).to(tl.float32)
        # Masked lanes load 0.0 and contribute exactly nothing to the sum of squares, so the tail
        # columns need no fixup -- the same reason layernorm dropped its own.
        var = tl.sum(x * x, axis=1) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)

        y = x * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = y * w[None, :]
        tl.store(Y + rows[:, None] * stride_r + cols[None, :] * stride_c, y, mask=mask)
    else:
        ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            mask = row_mask[:, None] & (cols[None, :] < N)
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            ss += tl.sum(x * x, axis=1)
        rstd = 1 / tl.sqrt(ss / N + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)

        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            y = x * rstd[:, None]
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                y = y * w[None, :]
            tl.store(Y + rows[:, None] * stride_r + cols[None, :] * stride_c, y, mask=mask)


@triton.autotune(configs=configs_for("rmsnorm_bwd_triton"), key=["shape_key", "HAS_WEIGHT"],
                 reset_to_zero=["DW"])
@triton.jit
def rmsnorm_bwd_kernel(
    DX, DY, DW, X, W, Rstd,
    stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr,
):
    """``dx`` and, with a weight, ``dweight``.

    ``dweight`` is fp32 and accumulated across row tiles with ``atomic_add``, so its buffer is
    zeroed by the caller (not merely by the autotuner's ``reset_to_zero``, which fires only while
    a config is being benchmarked).
    """
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M
    rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)

    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                    mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, x * rstd[:, None], 0.0)
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
            wdy = tl.where(mask, dy * w[None, :], 0.0)
        else:
            wdy = tl.where(mask, dy, 0.0)
        # No c2: without the mean there is only the rstd term in the derivative.
        c1 = tl.sum(xhat * wdy, axis=1) / N
        dx = (wdy - xhat * c1[:, None]) * rstd[:, None]
        tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, x * rstd[:, None], 0.0)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
                wdy = tl.where(mask, dy * w[None, :], 0.0)
            else:
                wdy = tl.where(mask, dy, 0.0)
            c1 += tl.sum(xhat * wdy, axis=1)
        c1 = c1 / N

        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, x * rstd[:, None], 0.0)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                wdy = tl.where(mask, dy * w[None, :], 0.0)
            else:
                wdy = tl.where(mask, dy, 0.0)
            dx = (wdy - xhat * c1[:, None]) * rstd[:, None]
            tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)


@triton.autotune(configs=configs_for("rmsnorm_adamod_fwd_triton"),
                 key=["shape_key", "HAS_WEIGHT"])
@triton.jit
def rmsnorm_adamod_fwd_kernel(
    Q, C, WSC, WSH, WG, W, Y, Rstd, GATE,
    stride_qr, stride_qc, stride_cr, stride_cc, stride_wn, stride_wk,
    M, N: tl.constexpr, K: tl.constexpr, eps: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_DC: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr,
):
    """``rmsnorm(q) * (1 + c@Wsc^T) + c@Wsh^T`` -- the modulation projection folded in.

    The projection output never reaches HBM. That is the whole point: at the atom-DiT shape
    (M = 48*8192, d_model 128, d_cond 384) the `[M, 2*d]` scale/shift pair is 96 MB held between
    forward and backward, and dropping it halves what the step holds, 193.5 MB to 97.5 MB. The
    forward also comes out ahead -- 0.61 ms against 0.77 for two cuBLAS GEMMs and a fused
    modulate -- because the two projection writes it skips cost more than a hand-tiled `tl.dot`
    gives up. Over a full training step that lead is spent again (see the backward); the memory
    is what survives.

    Two tile axes and two passes. Pass one reduces q over the normalized axis for rstd; pass two
    walks the same axis again and, per tile, contracts c against the two weight slices with
    `tl.dot`. c is re-read once per N tile -- one tile at d_model 128, six at 768 -- which is the
    price of not materialising the projection.
    """
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M

    if BLOCK_K >= N:
        # ONE tile over the normalized axis, so q is read once and held: the reduction that
        # makes rstd and the output that consumes it are the same registers. The tiled branch
        # below cannot -- rstd needs the whole row before any output element is final -- and
        # reads q a second time, and c once per N tile.
        #
        # It does NOT win at d_model 128, where it is reachable (BLOCK_K 128): a full sweep of
        # the ladder picks BLOCK_M1=64 BLOCK_K=64 over every covering config. Two reasons, both
        # worth knowing before reaching for this branch again. The re-reads it removes are L1/L2
        # hits, not HBM traffic -- the second N tile wants the same 48 KB of c the first just
        # read -- so there is far less to win than the byte count suggests. And covering costs
        # registers: acc_sc and acc_sh are each [BLOCK_M1, BLOCK_K] fp32, so doubling BLOCK_K
        # doubles both and forces BLOCK_M1 down.
        #
        # It stays because it is the right shape of code for a NARROW normalized axis (a model
        # config with d_model 64, or ragged mode's 125), where covering is the only tiling and
        # the accumulators are small. The autotuner decides; this branch just has to exist for
        # it to be able to.
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                    mask=mask, other=0.0).to(tl.float32)
        rstd = 1 / tl.sqrt(tl.sum(q * q, axis=1) / N + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)

        acc_sc = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
        acc_sh = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
        # The block's THIRD chunk, from the same c tile, and NOT optional. adaLN projects six
        # of these from one conditioning vector and every sub-layer uses its scale, its shift
        # AND its gate, so a no-gate variant is a second compiled kernel and a second cache
        # bucket that nothing reaches. No extra read here -- c is in registers for the other
        # two, and the weight tile is the only new load.
        acc_g = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_DC):
            ks = k0 + tl.arange(0, BLOCK_DC)
            k_mask = ks < K
            c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            wmask = k_mask[:, None] & col_mask[None, :]
            wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                          mask=wmask, other=0.0)
            wsh = tl.load(WSH + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                          mask=wmask, other=0.0)
            acc_sc += tl.dot(c, wsc)
            acc_sh += tl.dot(c, wsh)
            wg = tl.load(WG + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                         mask=wmask, other=0.0)
            acc_g += tl.dot(c, wg)
        y = q * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = y * w[None, :]
        y = y * (1.0 + acc_sc) + acc_sh
        tl.store(Y + rows[:, None] * stride_qr + cols[None, :] * stride_qc, y, mask=mask)
        tl.store(GATE + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                 acc_g, mask=mask)
        return

    ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_K):
        cols = n0 + tl.arange(0, BLOCK_K)
        mask = row_mask[:, None] & (cols[None, :] < N)
        q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                    mask=mask, other=0.0).to(tl.float32)
        ss += tl.sum(q * q, axis=1)
    rstd = 1 / tl.sqrt(ss / N + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)

    for n0 in range(0, N, BLOCK_K):
        cols = n0 + tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        acc_sc = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
        acc_sh = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
        # The block's THIRD chunk, from the same c tile, and NOT optional. adaLN projects six
        # of these from one conditioning vector and every sub-layer uses its scale, its shift
        # AND its gate, so a no-gate variant is a second compiled kernel and a second cache
        # bucket that nothing reaches. No extra read here -- c is in registers for the other
        # two, and the weight tile is the only new load.
        acc_g = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_DC):
            ks = k0 + tl.arange(0, BLOCK_DC)
            k_mask = ks < K
            # bf16 operands into tl.dot, fp32 accumulator: casting to fp32 first would drop the
            # tensor cores this loop exists to use.
            c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            wmask = k_mask[:, None] & col_mask[None, :]
            wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                          mask=wmask, other=0.0)
            wsh = tl.load(WSH + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                          mask=wmask, other=0.0)
            acc_sc += tl.dot(c, wsc)
            acc_sh += tl.dot(c, wsh)
            wg = tl.load(WG + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                         mask=wmask, other=0.0)
            acc_g += tl.dot(c, wg)
        q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                    mask=mask, other=0.0).to(tl.float32)
        y = q * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = y * w[None, :]
        y = y * (1.0 + acc_sc) + acc_sh
        tl.store(Y + rows[:, None] * stride_qr + cols[None, :] * stride_qc, y, mask=mask)
        tl.store(GATE + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                 acc_g, mask=mask)


def _rms_fwd_fake(x_2d, weight, eps, has_w, shape_key):
    """``y`` like ``x_2d``, plus ``rstd`` as (M,) fp32 -- the kernel reduces and stores the row
    statistic in fp32 whatever the activation dtype is, and the backward reloads it as such."""
    return torch.empty_like(x_2d), x_2d.new_empty((x_2d.shape[0],), dtype=torch.float32)


@opaque(fake=_rms_fwd_fake, name="rmsnorm_fwd")
def _rms_fwd(x_2d: torch.Tensor, weight: torch.Tensor | None, eps: float, has_w: bool,
             shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``x * rstd (* w)`` over the last axis of a flattened (M, N)."""
    m, n = x_2d.shape
    y = torch.empty_like(x_2d)
    rstd = x_2d.new_empty((m,), dtype=torch.float32)
    empty = x_2d.new_empty((0,))
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_fwd_kernel[grid](
        x_2d, y, weight if has_w else empty, rstd,
        x_2d.stride(0), x_2d.stride(1), m, n, eps,
        shape_key=shape_key, HAS_WEIGHT=has_w,
    )
    return y, rstd


def _rms_bwd_fake(dy_2d, x_2d, weight, rstd, has_w, input_shape, shape_key):
    """``(dx at the pre-flatten shape, dweight (N,) fp32)``."""
    return (dy_2d.new_empty(tuple(input_shape)),
            x_2d.new_empty((x_2d.shape[1],), dtype=torch.float32))


@opaque(fake=_rms_bwd_fake, name="rmsnorm_bwd")
def _rms_bwd(dy_2d: torch.Tensor, x_2d: torch.Tensor, weight: torch.Tensor | None,
             rstd: torch.Tensor, has_w: bool, input_shape: list[int], shape_key: int
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dx, dweight)``.

    ``dweight`` is fp32 and zeroed here, not merely by the autotuner's ``reset_to_zero``: the
    kernel accumulates it across row tiles with ``atomic_add``, and reset_to_zero fires only
    while a config is being benchmarked -- a cache-served shape would start from allocator junk.
    """
    m, n = x_2d.shape
    dx = torch.empty_like(x_2d)
    dw = x_2d.new_zeros((n,), dtype=torch.float32)
    empty = x_2d.new_empty((0,))
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_bwd_kernel[grid](
        dx, dy_2d, dw, x_2d, weight if has_w else empty, rstd,
        x_2d.stride(0), x_2d.stride(1), m, n,
        shape_key=shape_key, HAS_WEIGHT=has_w,
    )
    return dx.reshape(tuple(input_shape)), dw


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = list(x.shape)
        n = shape[-1]
        x_2d = x.reshape(-1, n)
        if x_2d.stride(1) != 1:
            x_2d = x_2d.contiguous()
        has_w = weight is not None
        # both_key(rows) + the normalized width, as layernorm's launchers key it: these kernels
        # run on both streams, and one length means two different launches for them.
        key = both_key(rows_of(x.shape), N=n)
        y, rstd = _rms_fwd(x_2d, weight, eps, has_w, key)
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.shape, ctx.key, ctx.has_w = shape, key, has_w
        return y.reshape(tuple(shape))

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, x_2d.shape[1])
        if dy_2d.stride(1) != 1:
            dy_2d = dy_2d.contiguous()
        dx, dw = _rms_bwd(dy_2d, x_2d, weight, rstd, ctx.has_w, ctx.shape, ctx.key)
        dweight = dw.to(weight.dtype) if ctx.has_w else None
        return dx, dweight, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None = None,
                   eps: float = 1e-5) -> torch.Tensor:
    """``x / sqrt(mean(x^2) + eps) * weight`` over the LAST axis, in one pass.

    ``weight=None`` is the unweighted form, which is what the SWA q/k normalization uses; the
    triangle-attention call site passes one. Shape is preserved; anything before the last axis is
    flattened into rows.
    """
    return _RMSNorm.apply(x, weight, eps)


@triton.autotune(configs=configs_for("rmsnorm_adamod_bwd_triton"),
                 key=["shape_key", "HAS_WEIGHT"], reset_to_zero=["DW"])
@triton.jit
def rmsnorm_adamod_bwd_kernel(
    DQ, DSD, DW, DY, Q, C, WSC, W, Rstd,
    stride_qr, stride_qc, stride_cr, stride_cc, stride_wn, stride_wk,
    stride_sr, stride_sc,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_DC: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr,
):
    """``dq``, ``dscale`` and ``dweight`` for the fused modulate. ``dshift`` IS ``dy``.

    ``scale`` is RECOMPUTED here (``c @ Wsc^T``) rather than saved: saving it is the 192 MB the
    forward fusion exists to not spend, so the backward pays one GEMM instead. The three large
    GEMMs the chain still needs -- ``dWsc``, ``dWsh``, ``dc`` -- stay in the caller, on cuBLAS,
    which is better at them than a hand-tiled `tl.dot` and needs no scratch here.

    The covering-tile branch (BLOCK_K >= N, which is the atom-DiT d_model of 128) keeps
    everything in registers and recomputes nothing twice. The tiled branch cannot: ``c1`` reduces
    over the whole row, so the GEMM is recomputed in the second pass rather than spilling
    ``dnormed`` to a [M, N] scratch buffer.
    """
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M
    rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)

    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        acc_sc = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_DC):
            ks = k0 + tl.arange(0, BLOCK_DC)
            k_mask = ks < K
            c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                          mask=k_mask[:, None] & col_mask[None, :], other=0.0)
            acc_sc += tl.dot(c, wsc)
        q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                    mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                     mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, q * rstd[:, None], 0.0)
        normed = xhat
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            normed = xhat * w[None, :]
        # dscale and dy go into ONE [M, 2N] buffer, side by side. Both are already in
        # registers here, so the second store is the whole cost of a concatenation the
        # caller would otherwise pay for -- and it turns the caller's four GEMMs into two:
        #   dWpair = [dscale|dy]^T @ cs      (was dWsc and dWsh separately)
        #   dc     = [dscale|dy] @ [Wsc;Wsh] (was two [M,N]@[N,K] and an add)
        tl.store(DSD + rows[:, None] * stride_sr + cols[None, :] * stride_sc,
                 dy * normed, mask=mask)
        tl.store(DSD + rows[:, None] * stride_sr + (N + cols[None, :]) * stride_sc,
                 dy, mask=mask)
        dnormed = tl.where(mask, dy * (1.0 + acc_sc), 0.0)
        if HAS_WEIGHT:
            tl.atomic_add(DW + cols, tl.sum(dnormed * xhat, axis=0), mask=col_mask)
            wdy = tl.where(mask, dnormed * w[None, :], 0.0)
        else:
            wdy = dnormed
        c1 = tl.sum(xhat * wdy, axis=1) / N
        dq = (wdy - xhat * c1[:, None]) * rstd[:, None]
        tl.store(DQ + rows[:, None] * stride_qr + cols[None, :] * stride_qc, dq, mask=mask)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            acc_sc = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_DC):
                ks = k0 + tl.arange(0, BLOCK_DC)
                k_mask = ks < K
                c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                              mask=k_mask[:, None] & col_mask[None, :], other=0.0)
                acc_sc += tl.dot(c, wsc)
            q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, q * rstd[:, None], 0.0)
            normed = xhat
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                normed = xhat * w[None, :]
            # dscale and dy go into ONE [M, 2N] buffer, side by side. Both are already in
            # registers here, so the second store is the whole cost of a concatenation the
            # caller would otherwise pay for -- and it turns the caller's four GEMMs into two:
            #   dWpair = [dscale|dy]^T @ cs      (was dWsc and dWsh separately)
            #   dc     = [dscale|dy] @ [Wsc;Wsh] (was two [M,N]@[N,K] and an add)
            tl.store(DSD + rows[:, None] * stride_sr + cols[None, :] * stride_sc,
                     dy * normed, mask=mask)
            tl.store(DSD + rows[:, None] * stride_sr + (N + cols[None, :]) * stride_sc,
                     dy, mask=mask)
            dnormed = tl.where(mask, dy * (1.0 + acc_sc), 0.0)
            if HAS_WEIGHT:
                tl.atomic_add(DW + cols, tl.sum(dnormed * xhat, axis=0), mask=col_mask)
                wdy = tl.where(mask, dnormed * w[None, :], 0.0)
            else:
                wdy = dnormed
            c1 += tl.sum(xhat * wdy, axis=1)
        c1 = c1 / N

        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            acc_sc = tl.zeros([BLOCK_M1, BLOCK_K], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_DC):
                ks = k0 + tl.arange(0, BLOCK_DC)
                k_mask = ks < K
                c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                              mask=k_mask[:, None] & col_mask[None, :], other=0.0)
                acc_sc += tl.dot(c, wsc)
            q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, q * rstd[:, None], 0.0)
            dnormed = tl.where(mask, dy * (1.0 + acc_sc), 0.0)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                wdy = tl.where(mask, dnormed * w[None, :], 0.0)
            else:
                wdy = dnormed
            dq = (wdy - xhat * c1[:, None]) * rstd[:, None]
            tl.store(DQ + rows[:, None] * stride_qr + cols[None, :] * stride_qc, dq, mask=mask)


def _adamod_fwd_fake(q_2d, c_2d, wsc, wsh, wg, weight, eps, has_w, shape_key):
    """``y`` and ``gate`` like ``q_2d``, plus ``rstd`` as (M,) fp32."""
    return (torch.empty_like(q_2d), q_2d.new_empty((q_2d.shape[0],), dtype=torch.float32),
            torch.empty_like(q_2d))


@opaque(fake=_adamod_fwd_fake, name="rmsnorm_adamod_fwd")
def _adamod_fwd(q_2d: torch.Tensor, c_2d: torch.Tensor, wsc: torch.Tensor, wsh: torch.Tensor,
                wg: torch.Tensor, weight: torch.Tensor | None, eps: float, has_w: bool,
                shape_key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rmsnorm(q) * (1 + c@wsc^T) + c@wsh^T`` and ``c@wg^T``, all from one read of the c tile."""
    m, n = q_2d.shape
    k = c_2d.shape[1]
    empty = q_2d.new_empty((0,))
    y = torch.empty_like(q_2d)
    rstd = q_2d.new_empty((m,), dtype=torch.float32)
    gate = torch.empty_like(q_2d)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_adamod_fwd_kernel[grid](
        q_2d, c_2d, wsc, wsh, wg, weight if has_w else empty, y, rstd, gate,
        q_2d.stride(0), q_2d.stride(1), c_2d.stride(0), c_2d.stride(1),
        wsc.stride(0), wsc.stride(1),
        m, n, k, eps, shape_key=shape_key, HAS_WEIGHT=has_w,
    )
    return y, rstd, gate


#: scale, shift, gate -- what one adaLN sub-layer consumes, and therefore how wide the stacked
#: gradient buffer is. Named because it appears in the buffer's shape and in the slicing that
#: reads it back, and those two must not drift.
_ADALN_CHUNKS = 3


def _adamod_bwd_fake(dy_2d, q_2d, c_2d, wsc, weight, rstd, has_w, shape_key):
    """``(dq, the stacked (M, 3N) gradient buffer, dweight)``. The big GEMMs are the caller's."""
    m, nn = q_2d.shape
    return (torch.empty_like(q_2d), q_2d.new_empty((m, _ADALN_CHUNKS * nn)),
            q_2d.new_empty((nn,), dtype=torch.float32))


@opaque(fake=_adamod_bwd_fake, name="rmsnorm_adamod_bwd")
def _adamod_bwd(dy_2d: torch.Tensor, q_2d: torch.Tensor, c_2d: torch.Tensor, wsc: torch.Tensor,
                weight: torch.Tensor | None, rstd: torch.Tensor, has_w: bool,
                shape_key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``dq``, ``[dscale | dy]`` stacked as one (M, 2N), and ``dweight``.

    Stacked because every consumer wants them together: the weight gradients are one
    ``[dscale|dy]^T @ cs`` instead of two, and ``dc`` is one ``[dscale|dy] @ [Wsc;Wsh]`` instead
    of two GEMMs and an add. The kernel holds both in registers already, so the stacking is a
    store and not a concatenation. ``scale`` is still recomputed rather than read back.
    """
    m, n = q_2d.shape
    k = c_2d.shape[1]
    dq = torch.empty_like(q_2d)
    # Three chunks wide. The kernel is unaware of the third -- it writes slots 0 and 1 through
    # `stride_sr`, and the wider stride simply leaves room after them for the caller's dgate.
    dsd = q_2d.new_empty((m, _ADALN_CHUNKS * n))
    # ZEROED, not `new_empty`: the kernel accumulates dweight across row tiles with
    # `atomic_add`, and the autotuner's `reset_to_zero` only fires on the paths that
    # benchmark a config -- once a shape is tuned and served from cache, nothing clears
    # the buffer and the accumulation starts from whatever the allocator handed back.
    dw = q_2d.new_zeros((n,), dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_adamod_bwd_kernel[grid](
        dq, dsd, dw, dy_2d, q_2d, c_2d, wsc,
        weight if has_w else q_2d.new_empty((0,)), rstd,
        q_2d.stride(0), q_2d.stride(1), c_2d.stride(0), c_2d.stride(1),
        wsc.stride(0), wsc.stride(1),
        dsd.stride(0), dsd.stride(1),
        m, n, k, shape_key=shape_key, HAS_WEIGHT=has_w,
    )
    return dq, dsd, dw


class _RMSNormAdaMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, c, w_scale, w_shift, w_gate, weight, eps):
        shape = list(q.shape)
        n = shape[-1]
        q2 = q.reshape(-1, n)
        c2 = c.reshape(-1, c.shape[-1])
        if q2.stride(1) != 1:
            q2 = q2.contiguous()
        if c2.stride(1) != 1:
            c2 = c2.contiguous()
        wsc, wsh = w_scale.contiguous(), w_shift.contiguous()
        has_w = weight is not None
        wg = w_gate.contiguous()
        # d_cond is in the key, not just d_model. BLOCK_DC tiles the conditioning width, so two
        # blocks with the same rows and d_model but different d_cond -- 128 on the atom side and
        # 384 on the token side, which is exactly the pair the model builds -- want different
        # configs and would otherwise have shared one cache entry.
        key = both_key(rows_of(q.shape), N=n, K=c2.shape[1])
        y, rstd, gate = _adamod_fwd(q2, c2, wsc, wsh, wg, weight, eps, has_w, key)
        # q and c, NOT scale/shift: saving the projection outputs is the [M, 2N] this exists to
        # not spend, so the backward recomputes `scale` from what the forward already needed.
        ctx.save_for_backward(q2, c2, wsc, wsh, wg, weight, rstd)
        ctx.shape, ctx.cshape, ctx.key, ctx.has_w = shape, list(c.shape), key, has_w
        return y.reshape(tuple(shape)), gate.reshape(tuple(shape))

    @staticmethod
    def backward(ctx, dy, dgate):
        q2, c2, wsc, wsh, wg, weight, rstd = ctx.saved_tensors
        n = q2.shape[1]
        dy2 = dy.reshape(-1, n)
        if dy2.stride(1) != 1:
            dy2 = dy2.contiguous()
        # dshift IS dy, and the kernel writes it beside dscale in ONE stacked buffer; the gate
        # takes the third column block, which the kernel does not know about -- it strides over
        # a wider buffer and the caller fills the slot. So both chains are a SINGLE GEMM each
        # rather than one per chunk.
        dq, dsd, dw = _adamod_bwd(dy2, q2, c2, wsc, weight, rstd, ctx.has_w, ctx.key)
        dg2 = dgate.reshape(-1, n)
        dsd[:, 2 * n:].copy_(dg2 if dg2.stride(1) == 1 else dg2.contiguous())
        w_stack = torch.cat((wsc, wsh, wg), dim=0)              # [3N, K], 98 KB a slice
        dw_stack = dsd.mT @ c2                                  # [3N, K]  ONE GEMM
        dc = (dsd @ w_stack).reshape(tuple(ctx.cshape))         # [M, K]   ONE GEMM
        dweight = dw.to(weight.dtype) if ctx.has_w else None
        return (dq.reshape(tuple(ctx.shape)), dc, dw_stack[:n].to(wsc.dtype),
                dw_stack[n:2 * n].to(wsh.dtype), dw_stack[2 * n:].to(wg.dtype), dweight, None)


def triton_rmsnorm_adamod(q: torch.Tensor, c: torch.Tensor, w_scale: torch.Tensor,
                          w_shift: torch.Tensor, w_gate: torch.Tensor,
                          weight: torch.Tensor | None = None,
                          eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """``rmsnorm(q) * (1 + c @ w_scale^T) + c @ w_shift^T``, projection included, fwd and bwd.

    ``c`` IS THE ACTIVATED CONDITIONING -- the caller passes ``SiLU(c)``, not ``c``. adaLN is
    ``Sequential(nn.SiLU(), Linear(d_cond, 6 * d_atom, bias=False))``, and the activation is one
    elementwise pass that ALL SIX chunks share, so it belongs where it is computed once. Folding
    it in here was tried and measured, and it loses twice over: this kernel is reached twice per
    block and re-reads ``c`` once per N tile, so the activation is recomputed four times where
    the caller did it once; and the gate chunks keep an ordinary ``Linear`` on it, so the tensor
    gets built anyway and the forward saves nothing. The backward is worse still -- the weight
    gradients contract ``SiLU(c)``, so the kernel has to WRITE it back out, which tripled its
    stores and took the block's adaLN backward from 6.86 ms to 13.87.

    ``w_gate`` is the block's THIRD chunk for this sub-layer and is REQUIRED: ``(y, gate)`` is
    always the return. It costs no extra read of ``c`` -- the tile is in registers for the other
    two projections, and the only new load is a weight slice -- and the alternative, a separate
    ``Linear`` for the gates, reads the whole conditioning again and puts two more GEMMs in the
    backward. It is not a flag because every adaLN sub-layer uses all three of its chunks: a
    gateless variant would be a second compiled kernel, a second cache bucket and a second set
    of build units that no caller reaches.

    ``w_scale``/``w_shift``/``w_gate`` are ``(d_model, d_cond)`` slices of the block's adaLN
    projection -- slices, not the whole 6-chunk weight, so the total GEMM work is what it was.
    No bias: the call site this is written for uses ``Linear(..., bias=False)``.

    Neither ``scale`` nor ``shift`` ever reaches HBM, in either direction: the forward keeps the
    projection in registers, and the backward recomputes ``scale`` from ``c`` rather than saving
    it. Measured at the atom-DiT shape (M = 48*8192, d_model 128, d_cond 384) against the same
    thing written with a shared 6-chunk projection, over one block's adaLN portion:

        eager                     fwd 3.55 ms  bwd 5.05  step 8.59  held 1443 MB
        shared GEMM + modulate    fwd 2.84     bwd 3.94  step 6.79  held 1251 MB
        this                      fwd 2.00     bwd 5.49  step 7.49  held  675 MB
    """
    return _RMSNormAdaMod.apply(q, c, w_scale, w_shift, w_gate, weight, eps)
