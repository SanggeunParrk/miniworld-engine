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


@triton.autotune(configs=configs_for("rmsnorm_fwd_triton"), key=["shape_key", "HAS_WEIGHT", "HAS_MODULATION"])
@triton.jit
def rmsnorm_fwd_kernel(
    X, Y, W, Rstd, Scale, Shift,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr, HAS_MODULATION: tl.constexpr,
):
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
        if HAS_MODULATION:
            sc = tl.load(Scale + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     mask=mask, other=0.0).to(tl.float32)
            sh = tl.load(Shift + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     mask=mask, other=0.0).to(tl.float32)
            y = y * (1.0 + sc) + sh
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
            if HAS_MODULATION:
                sc = tl.load(Scale + rows[:, None] * stride_r + cols[None, :] * stride_c,
                             mask=mask, other=0.0).to(tl.float32)
                sh = tl.load(Shift + rows[:, None] * stride_r + cols[None, :] * stride_c,
                             mask=mask, other=0.0).to(tl.float32)
                y = y * (1.0 + sc) + sh
            tl.store(Y + rows[:, None] * stride_r + cols[None, :] * stride_c, y, mask=mask)


@triton.autotune(configs=configs_for("rmsnorm_bwd_triton"), key=["shape_key", "HAS_WEIGHT", "HAS_MODULATION"],
                 reset_to_zero=["DW"])
@triton.jit
def rmsnorm_bwd_kernel(
    DX, DY, DW, DSCALE, X, W, Rstd, Scale,
    stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr, HAS_MODULATION: tl.constexpr,
):
    """``dx`` and, under HAS_MODULATION, ``dscale``.

    ``dshift`` is not written: ``y = normed*(1+scale) + shift`` makes it exactly ``dy``, so the
    autograd Function hands ``dy`` back for it. Writing it would cost a second [M, N] store of a
    tensor the caller already holds.
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
        # `normed` is what the modulation multiplies: xhat, weighted if there is a weight.
        normed = xhat
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            normed = xhat * w[None, :]
        gain = tl.full([BLOCK_M1, BLOCK_K], 1.0, tl.float32)
        if HAS_MODULATION:
            sc = tl.load(Scale + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0.0).to(tl.float32)
            gain = 1.0 + sc
            tl.store(DSCALE + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     dy * normed, mask=mask)
        dnormed = tl.where(mask, dy * gain, 0.0)
        if HAS_WEIGHT:
            tl.atomic_add(DW + cols, tl.sum(dnormed * xhat, axis=0), mask=col_mask)
            wdy = tl.where(mask, dnormed * w[None, :], 0.0)
        else:
            wdy = dnormed
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
            normed = xhat
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                normed = xhat * w[None, :]
            gain = tl.full([BLOCK_M1, BLOCK_K], 1.0, tl.float32)
            if HAS_MODULATION:
                sc = tl.load(Scale + rows[:, None] * stride_r + cols[None, :] * stride_c,
                             mask=mask, other=0.0).to(tl.float32)
                gain = 1.0 + sc
                tl.store(DSCALE + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         dy * normed, mask=mask)
            dnormed = tl.where(mask, dy * gain, 0.0)
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
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, x * rstd[:, None], 0.0)
            gain = tl.full([BLOCK_M1, BLOCK_K], 1.0, tl.float32)
            if HAS_MODULATION:
                sc = tl.load(Scale + rows[:, None] * stride_r + cols[None, :] * stride_c,
                             mask=mask, other=0.0).to(tl.float32)
                gain = 1.0 + sc
            dnormed = tl.where(mask, dy * gain, 0.0)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                wdy = tl.where(mask, dnormed * w[None, :], 0.0)
            else:
                wdy = dnormed
            dx = (wdy - xhat * c1[:, None]) * rstd[:, None]
            tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)


@triton.autotune(configs=configs_for("rmsnorm_adamod_fwd_triton"),
                 key=["shape_key", "HAS_WEIGHT"])
@triton.jit
def rmsnorm_adamod_fwd_kernel(
    Q, C, WSC, WSH, W, Y, Rstd,
    stride_qr, stride_qc, stride_cr, stride_cc, stride_wn, stride_wk,
    M, N: tl.constexpr, K: tl.constexpr, eps: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
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
        for k0 in range(0, K, BLOCK_C):
            ks = k0 + tl.arange(0, BLOCK_C)
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
        y = q * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = y * w[None, :]
        y = y * (1.0 + acc_sc) + acc_sh
        tl.store(Y + rows[:, None] * stride_qr + cols[None, :] * stride_qc, y, mask=mask)
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
        for k0 in range(0, K, BLOCK_C):
            ks = k0 + tl.arange(0, BLOCK_C)
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
        q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                    mask=mask, other=0.0).to(tl.float32)
        y = q * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = y * w[None, :]
        y = y * (1.0 + acc_sc) + acc_sh
        tl.store(Y + rows[:, None] * stride_qr + cols[None, :] * stride_qc, y, mask=mask)


def _rms_fwd_fake(x_2d, weight, scale, shift, eps, has_w, has_mod, shape_key):
    """``y`` like ``x_2d``, plus ``rstd`` as (M,) fp32 -- the kernel reduces and stores the row
    statistic in fp32 whatever the activation dtype is, and the backward reloads it as such."""
    return torch.empty_like(x_2d), x_2d.new_empty((x_2d.shape[0],), dtype=torch.float32)


@opaque(fake=_rms_fwd_fake, name="rmsnorm_fwd")
def _rms_fwd(x_2d: torch.Tensor, weight: torch.Tensor | None, scale: torch.Tensor | None,
             shift: torch.Tensor | None, eps: float, has_w: bool, has_mod: bool,
             shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x * rstd (* w)) (* (1+scale) + shift)`` over the last axis of a flattened (M, N)."""
    m, n = x_2d.shape
    y = torch.empty_like(x_2d)
    rstd = x_2d.new_empty((m,), dtype=torch.float32)
    empty = x_2d.new_empty((0,))
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_fwd_kernel[grid](
        x_2d, y, weight if has_w else empty, rstd,
        scale if has_mod else empty, shift if has_mod else empty,
        x_2d.stride(0), x_2d.stride(1), m, n, eps,
        shape_key=shape_key, HAS_WEIGHT=has_w, HAS_MODULATION=has_mod,
    )
    return y, rstd


def _rms_bwd_fake(dy_2d, x_2d, weight, scale, rstd, has_w, has_mod, input_shape, shape_key):
    """``(dx at the pre-flatten shape, dweight (N,) fp32, dscale like x_2d)``."""
    return (dy_2d.new_empty(tuple(input_shape)),
            x_2d.new_empty((x_2d.shape[1],), dtype=torch.float32),
            torch.empty_like(x_2d))


@opaque(fake=_rms_bwd_fake, name="rmsnorm_bwd")
def _rms_bwd(dy_2d: torch.Tensor, x_2d: torch.Tensor, weight: torch.Tensor | None,
             scale: torch.Tensor | None, rstd: torch.Tensor, has_w: bool, has_mod: bool,
             input_shape: list[int], shape_key: int
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dx, dweight, dscale)``. ``dshift`` is not here: it is exactly ``dy``.

    ``dweight`` is fp32 and zeroed by the autotuner's ``reset_to_zero``, because the kernel
    accumulates it across row tiles with ``atomic_add``.
    """
    m, n = x_2d.shape
    dx = torch.empty_like(x_2d)
    # ZEROED, not `new_empty`: the kernel accumulates dweight across row tiles with
    # `atomic_add`, and the autotuner's `reset_to_zero` only fires on the paths that
    # benchmark a config -- once a shape is tuned and served from cache, nothing clears
    # the buffer and the accumulation starts from whatever the allocator handed back.
    dw = x_2d.new_zeros((n,), dtype=torch.float32)
    dscale = torch.empty_like(x_2d) if has_mod else x_2d.new_empty((0,))
    empty = x_2d.new_empty((0,))
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_bwd_kernel[grid](
        dx, dy_2d, dw, dscale, x_2d, weight if has_w else empty, rstd,
        scale if has_mod else empty,
        x_2d.stride(0), x_2d.stride(1), m, n,
        shape_key=shape_key, HAS_WEIGHT=has_w, HAS_MODULATION=has_mod,
    )
    return dx.reshape(tuple(input_shape)), dw, dscale


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, shift, eps):
        shape = list(x.shape)
        n = shape[-1]
        x_2d = x.reshape(-1, n)
        if x_2d.stride(1) != 1:
            x_2d = x_2d.contiguous()
        has_w, has_mod = weight is not None, scale is not None
        s2d = scale.reshape(-1, n).contiguous() if has_mod else None
        h2d = shift.reshape(-1, n).contiguous() if has_mod else None
        # both_key(rows) + the normalized width, as layernorm's launchers key it: these kernels
        # run on both streams, and one length means two different launches for them.
        key = both_key(rows_of(x.shape), N=n)
        y, rstd = _rms_fwd(x_2d, weight, s2d, h2d, eps, has_w, has_mod, key)
        ctx.save_for_backward(x_2d, weight, s2d, rstd)
        ctx.shape, ctx.key, ctx.has_w, ctx.has_mod = shape, key, has_w, has_mod
        return y.reshape(tuple(shape))

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, s2d, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, x_2d.shape[1])
        if dy_2d.stride(1) != 1:
            dy_2d = dy_2d.contiguous()
        dx, dw, dscale = _rms_bwd(dy_2d, x_2d, weight, s2d, rstd, ctx.has_w, ctx.has_mod,
                                  ctx.shape, ctx.key)
        dweight = dw.to(weight.dtype) if ctx.has_w else None
        if not ctx.has_mod:
            return dx, dweight, None, None, None
        # dshift IS dy: y = normed*(1+scale) + shift. Reshaped, not recomputed.
        return dx, dweight, dscale.reshape(tuple(ctx.shape)), dy, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None = None,
                   eps: float = 1e-5) -> torch.Tensor:
    """``x / sqrt(mean(x^2) + eps) * weight`` over the LAST axis, in one pass.

    ``weight=None`` is the unweighted form, which is what the SWA q/k normalization uses; the
    triangle-attention call site passes one. Shape is preserved; anything before the last axis is
    flattened into rows.
    """
    return _RMSNorm.apply(x, weight, None, None, eps)


def triton_rmsnorm_modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                            weight: torch.Tensor | None = None,
                            eps: float = 1e-5) -> torch.Tensor:
    """``rmsnorm(x) * (1 + scale) + shift`` -- the DiT block's modulate step, in ONE pass.

    ``scale`` and ``shift`` are per-element (they are chunks of a projection of the conditioning
    vector, so they carry the same shape as ``x``), not per-channel. Written separately the step
    is three passes and, worse, it holds ``rmsnorm(x)`` for the backward; here that intermediate
    never reaches HBM and the backward recomputes it from ``x`` and the saved ``rstd``.

    ``1 + scale``, not ``scale``: adaLN-Zero initialises the projection at zero so an untrained
    block is the identity, which is a different op from the ``sigmoid(scale)`` the ``adaln``
    family computes for AF3 -- that is why this is here and not a mode of that one.
    """
    return _RMSNorm.apply(x, weight, scale, shift, eps)


@triton.autotune(configs=configs_for("rmsnorm_adamod_bwd_triton"),
                 key=["shape_key", "HAS_WEIGHT"], reset_to_zero=["DW"])
@triton.jit
def rmsnorm_adamod_bwd_kernel(
    DQ, DSCALE, DW, DY, Q, C, WSC, W, Rstd,
    stride_qr, stride_qc, stride_cr, stride_cc, stride_wn, stride_wk,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
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
        for k0 in range(0, K, BLOCK_C):
            ks = k0 + tl.arange(0, BLOCK_C)
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
        tl.store(DSCALE + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                 dy * normed, mask=mask)
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
            for k0 in range(0, K, BLOCK_C):
                ks = k0 + tl.arange(0, BLOCK_C)
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
            tl.store(DSCALE + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                     dy * normed, mask=mask)
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
            for k0 in range(0, K, BLOCK_C):
                ks = k0 + tl.arange(0, BLOCK_C)
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


def _adamod_fwd_fake(q_2d, c_2d, wsc, wsh, weight, eps, has_w, shape_key):
    """``y`` like ``q_2d``, plus ``rstd`` as (M,) fp32."""
    return torch.empty_like(q_2d), q_2d.new_empty((q_2d.shape[0],), dtype=torch.float32)


@opaque(fake=_adamod_fwd_fake, name="rmsnorm_adamod_fwd")
def _adamod_fwd(q_2d: torch.Tensor, c_2d: torch.Tensor, wsc: torch.Tensor, wsh: torch.Tensor,
                weight: torch.Tensor | None, eps: float, has_w: bool,
                shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``rmsnorm(q) * (1 + c@wsc^T) + c@wsh^T`` with the projection kept in registers."""
    m, n = q_2d.shape
    k = c_2d.shape[1]
    y = torch.empty_like(q_2d)
    rstd = q_2d.new_empty((m,), dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_adamod_fwd_kernel[grid](
        q_2d, c_2d, wsc, wsh, weight if has_w else q_2d.new_empty((0,)), y, rstd,
        q_2d.stride(0), q_2d.stride(1), c_2d.stride(0), c_2d.stride(1),
        wsc.stride(0), wsc.stride(1),
        m, n, k, eps, shape_key=shape_key, HAS_WEIGHT=has_w,
    )
    return y, rstd


def _adamod_bwd_fake(dy_2d, q_2d, c_2d, wsc, weight, rstd, has_w, shape_key):
    """``(dq, dscale, dweight)``. The three big GEMMs are the caller's, on cuBLAS."""
    return (torch.empty_like(q_2d), torch.empty_like(q_2d),
            q_2d.new_empty((q_2d.shape[1],), dtype=torch.float32))


@opaque(fake=_adamod_bwd_fake, name="rmsnorm_adamod_bwd")
def _adamod_bwd(dy_2d: torch.Tensor, q_2d: torch.Tensor, c_2d: torch.Tensor, wsc: torch.Tensor,
                weight: torch.Tensor | None, rstd: torch.Tensor, has_w: bool,
                shape_key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``dq``, ``dscale`` and ``dweight``; ``scale`` is recomputed rather than read back."""
    m, n = q_2d.shape
    k = c_2d.shape[1]
    dq = torch.empty_like(q_2d)
    dscale = torch.empty_like(q_2d)
    # ZEROED, not `new_empty`: the kernel accumulates dweight across row tiles with
    # `atomic_add`, and the autotuner's `reset_to_zero` only fires on the paths that
    # benchmark a config -- once a shape is tuned and served from cache, nothing clears
    # the buffer and the accumulation starts from whatever the allocator handed back.
    dw = q_2d.new_zeros((n,), dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rmsnorm_adamod_bwd_kernel[grid](
        dq, dscale, dw, dy_2d, q_2d, c_2d, wsc,
        weight if has_w else q_2d.new_empty((0,)), rstd,
        q_2d.stride(0), q_2d.stride(1), c_2d.stride(0), c_2d.stride(1),
        wsc.stride(0), wsc.stride(1),
        m, n, k, shape_key=shape_key, HAS_WEIGHT=has_w,
    )
    return dq, dscale, dw


class _RMSNormAdaMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, c, w_scale, w_shift, weight, eps):
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
        key = both_key(rows_of(q.shape), N=n)
        y, rstd = _adamod_fwd(q2, c2, wsc, wsh, weight, eps, has_w, key)
        # q and c, NOT scale/shift: saving the projection outputs is the [M, 2N] this exists to
        # not spend, so the backward recomputes `scale` from what the forward already needed.
        ctx.save_for_backward(q2, c2, wsc, wsh, weight, rstd)
        ctx.shape, ctx.cshape, ctx.key, ctx.has_w = shape, list(c.shape), key, has_w
        return y.reshape(tuple(shape))

    @staticmethod
    def backward(ctx, dy):
        q2, c2, wsc, wsh, weight, rstd = ctx.saved_tensors
        n = q2.shape[1]
        dy2 = dy.reshape(-1, n)
        if dy2.stride(1) != 1:
            dy2 = dy2.contiguous()
        dq, dscale, dw = _adamod_bwd(dy2, q2, c2, wsc, weight, rstd, ctx.has_w, ctx.key)
        # dshift IS dy, so the shift half of every chain below is spelled with dy directly.
        dwsc = (dscale.mT @ c2).to(wsc.dtype)
        dwsh = (dy2.mT @ c2).to(wsh.dtype)
        # In place on the first product, not `torch.addmm(dscale @ wsc, dy2, wsh)`: that spells
        # one [M, K] temporary for the product and a second for the sum, and at the atom-DiT
        # shape each of those is 302 MB written and read for nothing.
        dc = dscale @ wsc
        dc.addmm_(dy2, wsh)
        dc = dc.reshape(tuple(ctx.cshape))
        dweight = dw.to(weight.dtype) if ctx.has_w else None
        return dq.reshape(tuple(ctx.shape)), dc, dwsc, dwsh, dweight, None


def triton_rmsnorm_adamod(q: torch.Tensor, c: torch.Tensor, w_scale: torch.Tensor,
                          w_shift: torch.Tensor, weight: torch.Tensor | None = None,
                          eps: float = 1e-5) -> torch.Tensor:
    """``rmsnorm(q) * (1 + c @ w_scale^T) + c @ w_shift^T``, projection included, fwd and bwd.

    ``w_scale``/``w_shift`` are the two ``(d_model, d_cond)`` slices of the block's adaLN
    projection -- the slices, not the whole 6-chunk weight, so the total GEMM work is what it was.
    No bias: the call site this is written for uses ``Linear(..., bias=False)``.

    Neither ``scale`` nor ``shift`` ever reaches HBM, in either direction: the forward keeps the
    projection in registers, and the backward recomputes ``scale`` from ``c`` rather than saving
    it. THAT is what this buys. Measured at the atom-DiT shape (M = 48*8192, d_model 128, d_cond
    384), against the same thing written as two `Linear`s and `triton_rmsnorm_modulate`:

        held 193.5 MB -> 97.5 MB, peak 3392 MB -> 2240 MB
        forward 0.77 ms -> 0.61, backward 2.26 -> 2.40, full step 3.03 -> 3.01

    So: a memory win, and a time WASH. The forward is genuinely faster -- it skips two [M, N]
    writes -- and the backward gives that back recomputing `scale`, which is the trade the
    recompute makes on purpose. Do not reach for this expecting a faster step.
    """
    return _RMSNormAdaMod.apply(q, c, w_scale, w_shift, weight, eps)
