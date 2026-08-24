
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/adaln.py
from miniworld_engine.autotune.configs import configs_for
import os

import torch
from miniworld_engine import settings
import triton
import triton.language as tl



AUTOTUNE = settings.current().autotunes("adaln")




# shape_key is the SHAPE bucket, and shape means L -- the atom count -- not the row count the
# kernels receive. This module is level=atom in kernels/registry.csv, so the wrapper is
# `atom_key`, and the value comes from `length_of(x.shape)` -- x's shape[-2] BEFORE the
# `reshape(-1, nx)` below. The flattened M is B*A, which is why it cannot be the key: two batch
# sizes at the same A are the same shape for tuning, and M cannot tell them apart.
from miniworld_engine.autotune.shape_key import atom_key, length_of


# USE_BF16/USE_FP16 belong in the key: they select the tl.dot operand precision -- a 16-bit
# tensor-core MMA versus the fp32 `input_precision="ieee"` path -- so they are a code path with a
# different best tile, not a tolerance. They are NOT implied by the cache's `dtype` component
# (the dtype of the first tensor operand, `dtype_of_args` in autotune/cache.py): `compute_dtype`
# below is the AUTOCAST dtype when autocast is on, while X/DY stay whatever the caller passed, so
# one fp32 operand reaches both flag settings and the two compiles would share one cache entry.
@triton.autotune(configs=configs_for("adaln_fwd_saveact_triton"),
                 key=['NX', 'NC', 'USE_BF16', 'USE_FP16', 'shape_key'])
@triton.jit
def adaln_fwd_kernel(  # noqa: C901, PLR0912, PLR0915
    X,
    Cond,
    LnW,
    ScaleW,
    ScaleB,
    BiasW,
    Y,
    XHat,
    CondNorm,
    Gate,
    RstdX,
    RstdC,
    stride_xr,
    stride_xc,
    stride_cr,
    stride_cc,
    stride_swr,
    stride_swc,
    stride_bwr,
    stride_bwc,
    M,
    NX: tl.constexpr,
    NC: tl.constexpr,
    eps_x,
    eps_cond,
    USE_BF16: tl.constexpr,
    USE_FP16: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K_NX: tl.constexpr,
    BLOCK_K_NC: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M

    mean_x = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for x_start in range(0, NX, BLOCK_K_NX):
        x_cols = x_start + tl.arange(0, BLOCK_K_NX)
        x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
        x_offsets = rows[:, None] * stride_xr + x_cols[None, :] * stride_xc
        x = tl.load(X + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
        mean_x += tl.sum(tl.where(x_mask, x, 0.0), axis=1)
    mean_x /= NX

    var_x = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for x_start in range(0, NX, BLOCK_K_NX):
        x_cols = x_start + tl.arange(0, BLOCK_K_NX)
        x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
        x_offsets = rows[:, None] * stride_xr + x_cols[None, :] * stride_xc
        x = tl.load(X + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
        x_centered = tl.where(x_mask, x - mean_x[:, None], 0.0)
        var_x += tl.sum(x_centered * x_centered, axis=1)
    rstd_x = tl.rsqrt(var_x / NX + eps_x)

    mean_cond = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for c_start in range(0, NC, BLOCK_K_NC):
        c_cols = c_start + tl.arange(0, BLOCK_K_NC)
        c_mask = row_mask[:, None] & (c_cols[None, :] < NC)
        c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc
        cond = tl.load(Cond + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean_cond += tl.sum(tl.where(c_mask, cond, 0.0), axis=1)
    mean_cond /= NC

    var_cond = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for c_start in range(0, NC, BLOCK_K_NC):
        c_cols = c_start + tl.arange(0, BLOCK_K_NC)
        c_mask = row_mask[:, None] & (c_cols[None, :] < NC)
        c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc
        cond = tl.load(Cond + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        cond_centered = tl.where(c_mask, cond - mean_cond[:, None], 0.0)
        var_cond += tl.sum(cond_centered * cond_centered, axis=1)
    rstd_cond = tl.rsqrt(var_cond / NC + eps_cond)

    tl.store(RstdX + rows, rstd_x, mask=row_mask)
    tl.store(RstdC + rows, rstd_cond, mask=row_mask)

    for c_start in range(0, NC, BLOCK_K_NC):
        c_cols = c_start + tl.arange(0, BLOCK_K_NC)
        c_mask = row_mask[:, None] & (c_cols[None, :] < NC)
        c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc
        cond = tl.load(Cond + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        cond_norm = (cond - mean_cond[:, None]) * rstd_cond[:, None]
        tl.store(CondNorm + c_offsets, cond_norm, mask=c_mask)

    for x_start in range(0, NX, BLOCK_K_NX):
        x_cols = x_start + tl.arange(0, BLOCK_K_NX)
        x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
        x_offsets = rows[:, None] * stride_xr + x_cols[None, :] * stride_xc

        x = tl.load(X + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
        x_hat = (x - mean_x[:, None]) * rstd_x[:, None]

        scale_b = tl.load(ScaleB + x_cols, mask=x_cols < NX, other=0.0).to(tl.float32)
        scale = tl.zeros([BLOCK_M1, BLOCK_K_NX], dtype=tl.float32)
        bias = tl.zeros([BLOCK_M1, BLOCK_K_NX], dtype=tl.float32)

        for c_start in range(0, NC, BLOCK_K_NC):
            c_cols = c_start + tl.arange(0, BLOCK_K_NC)
            c_mask = row_mask[:, None] & (c_cols[None, :] < NC)
            c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc
            cond = tl.load(Cond + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
            cond_norm = (cond - mean_cond[:, None]) * rstd_cond[:, None]
            lnw = tl.load(LnW + c_cols, mask=c_cols < NC, other=0.0).to(tl.float32)
            cond_aff = cond_norm * lnw[None, :]

            if USE_BF16:
                cond_aff = cond_aff.to(tl.bfloat16)
            elif USE_FP16:
                cond_aff = cond_aff.to(tl.float16)

            scale_offsets = (
                c_cols[:, None] * stride_swc + x_cols[None, :] * stride_swr
            )
            scale_mask = (c_cols[:, None] < NC) & (x_cols[None, :] < NX)
            scale_w = tl.load(
                ScaleW + scale_offsets,
                mask=scale_mask,
                other=0.0,
            ).to(tl.float32)
            if USE_BF16:
                scale_w = scale_w.to(tl.bfloat16)
            elif USE_FP16:
                scale_w = scale_w.to(tl.float16)

            bias_offsets = c_cols[:, None] * stride_bwc + x_cols[None, :] * stride_bwr
            bias_mask = (c_cols[:, None] < NC) & (x_cols[None, :] < NX)
            bias_w = tl.load(
                BiasW + bias_offsets,
                mask=bias_mask,
                other=0.0,
            ).to(tl.float32)
            if USE_BF16:
                bias_w = bias_w.to(tl.bfloat16)
            elif USE_FP16:
                bias_w = bias_w.to(tl.float16)

            scale += tl.dot(cond_aff, scale_w)
            bias += tl.dot(cond_aff, bias_w)

        scale += scale_b[None, :]

        # Round AFTER the sigmoid, not before: tl.sigmoid takes fp32/fp64 only, and rounding the
        # accumulator first bought nothing -- `scale` feeds no tl.dot from here on (unlike the
        # cond_aff/scale_w/bias_w casts above, which do). The stored Gate keeps its low-precision
        # dtype so the backward reconstructs exactly what the forward used.
        gate = tl.sigmoid(scale)
        if USE_BF16:
            gate = gate.to(tl.bfloat16)
            bias = bias.to(tl.bfloat16)
        elif USE_FP16:
            gate = gate.to(tl.float16)
            bias = bias.to(tl.float16)
        y = gate.to(tl.float32) * x_hat + bias.to(tl.float32)

        tl.store(XHat + x_offsets, x_hat, mask=x_mask)
        tl.store(Gate + x_offsets, gate, mask=x_mask)
        tl.store(Y + x_offsets, y, mask=x_mask)




@triton.autotune(configs=configs_for("adaln_bwd_dx_dbias_triton"),
                 # USE_BF16/USE_FP16: tl.dot operand precision, see adaln_fwd_kernel.
                 key=['shape_key', 'NX', 'NC', 'USE_BF16', 'USE_FP16'],
                 reset_to_zero=['DScaleB'])
@triton.jit
def adaln_bwd_input_kernel(  # noqa: PLR0915
    DY,
    DX,
    DCond,
    DScaleB,
    XHat,
    CondNorm,
    Gate,
    LnW,
    ScaleW,
    BiasW,
    RstdX,
    RstdC,
    stride_yr,
    stride_yc,
    stride_cr,
    stride_cc,
    stride_swr,
    stride_swc,
    stride_bwr,
    stride_bwc,
    M,
    NX: tl.constexpr,
    NC: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP16: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K_NX: tl.constexpr,
    BLOCK_K_NC: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M

    if BLOCK_K_NX >= NX and BLOCK_K_NC >= NC:
        # COVERING TILES. BLOCK_K_NX/NX and BLOCK_K_NC/NC are all tl.constexpr, so this test is
        # resolved at COMPILE time and only one of the two branches is ever emitted.
        #
        # When one tile spans each axis the four sweeps below degenerate to a single trip each,
        # and the general form then (a) re-reads DY / XHat / Gate FOUR times and (b) re-runs the
        # whole grad_cond_aff GEMM (2 tl.dot) a second time just to form dcond. Here the operands
        # are read ONCE and the fp32 accumulator `grad_cond_aff` is kept in REGISTERS and reused
        # for both the c1/c2 reductions and the dcond epilogue -- halving this kernel's tl.dot
        # count. Every expression is copied verbatim from the general branch, so at a covering
        # config the two are numerically identical.
        x_cols = tl.arange(0, BLOCK_K_NX)
        x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
        x_offsets = rows[:, None] * stride_yr + x_cols[None, :] * stride_yc

        dy_raw = tl.load(DY + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
        x_hat = tl.load(XHat + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
        gate = tl.load(Gate + x_offsets, mask=x_mask, other=0.0).to(tl.float32)

        grad_xhat = tl.where(x_mask, dy_raw * gate, 0.0)
        dscale = tl.where(x_mask, dy_raw * x_hat * gate * (1.0 - gate), 0.0)
        sum_grad_x = tl.sum(grad_xhat, axis=1)
        sum_grad_xhat = tl.sum(grad_xhat * tl.where(x_mask, x_hat, 0.0), axis=1)
        tl.atomic_add(
            DScaleB + x_cols,
            tl.sum(dscale, axis=0).to(DScaleB.dtype.element_ty),
            mask=x_cols < NX,
        )

        c1_x = sum_grad_xhat / NX
        c2_x = sum_grad_x / NX
        rstd_x = tl.load(RstdX + rows, mask=row_mask, other=0.0).to(tl.float32)

        c_cols = tl.arange(0, BLOCK_K_NC)
        c_mask = row_mask[:, None] & (c_cols[None, :] < NC)
        c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc
        cond_norm = tl.load(CondNorm + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        lnw = tl.load(LnW + c_cols, mask=c_cols < NC, other=0.0).to(tl.float32)

        dy = tl.where(x_mask, dy_raw, 0.0)
        dscale = dy * tl.where(x_mask, x_hat, 0.0) * gate * (1.0 - gate)
        weight_mask = (x_cols[:, None] < NX) & (c_cols[None, :] < NC)
        scale_w = tl.load(
            ScaleW + (x_cols[:, None] * stride_swr + c_cols[None, :] * stride_swc),
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        bias_w = tl.load(
            BiasW + (x_cols[:, None] * stride_bwr + c_cols[None, :] * stride_bwc),
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        grad_cond_aff = tl.zeros([BLOCK_M1, BLOCK_K_NC], dtype=tl.float32)
        grad_cond_aff = tl.dot(
            dscale, scale_w, acc=grad_cond_aff,
            input_precision="ieee", out_dtype=tl.float32,
        )
        grad_cond_aff = tl.dot(
            dy, bias_w, acc=grad_cond_aff,
            input_precision="ieee", out_dtype=tl.float32,
        )
        grad_cond_norm = grad_cond_aff * lnw[None, :]

        sum_grad_cond = tl.sum(tl.where(c_mask, grad_cond_norm, 0.0), axis=1)
        sum_grad_cond_norm = tl.sum(
            tl.where(c_mask, grad_cond_norm * cond_norm, 0.0),
            axis=1,
        )
        c1_cond = sum_grad_cond_norm / NC
        c2_cond = sum_grad_cond / NC
        rstd_cond = tl.load(RstdC + rows, mask=row_mask, other=0.0).to(tl.float32)

        # dx epilogue, reusing the ONE dy/gate/x_hat read above.
        dx = (dy_raw * gate - (x_hat * c1_x[:, None] + c2_x[:, None])) * rstd_x[:, None]
        tl.store(DX + x_offsets, dx, mask=x_mask)

        # dcond epilogue, reusing the ONE grad_cond_norm computed above (no second GEMM).
        dcond = (
            grad_cond_norm - (cond_norm * c1_cond[:, None] + c2_cond[:, None])
        ) * rstd_cond[:, None]
        tl.store(DCond + c_offsets, dcond, mask=c_mask)
    else:
        sum_grad_x = tl.zeros([BLOCK_M1], dtype=tl.float32)
        sum_grad_xhat = tl.zeros([BLOCK_M1], dtype=tl.float32)

        for x_start in range(0, NX, BLOCK_K_NX):
            x_cols = x_start + tl.arange(0, BLOCK_K_NX)
            x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
            x_offsets = rows[:, None] * stride_yr + x_cols[None, :] * stride_yc

            dy = tl.load(DY + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            x_hat = tl.load(XHat + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            gate = tl.load(Gate + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            grad_xhat = tl.where(x_mask, dy * gate, 0.0)
            dscale = tl.where(x_mask, dy * x_hat * gate * (1.0 - gate), 0.0)

            sum_grad_x += tl.sum(grad_xhat, axis=1)
            sum_grad_xhat += tl.sum(grad_xhat * tl.where(x_mask, x_hat, 0.0), axis=1)

            partial_dscale_b = tl.sum(dscale, axis=0)
            tl.atomic_add(
                DScaleB + x_cols,
                partial_dscale_b.to(DScaleB.dtype.element_ty),
                mask=x_cols < NX,
            )

        c1_x = sum_grad_xhat / NX
        c2_x = sum_grad_x / NX
        rstd_x = tl.load(RstdX + rows, mask=row_mask, other=0.0).to(tl.float32)

        sum_grad_cond = tl.zeros([BLOCK_M1], dtype=tl.float32)
        sum_grad_cond_norm = tl.zeros([BLOCK_M1], dtype=tl.float32)

        for c_start in range(0, NC, BLOCK_K_NC):
            c_cols = c_start + tl.arange(0, BLOCK_K_NC)
            c_mask = row_mask[:, None] & (c_cols[None, :] < NC)
            c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc

            cond_norm = tl.load(CondNorm + c_offsets, mask=c_mask, other=0.0).to(
                tl.float32
            )
            lnw = tl.load(LnW + c_cols, mask=c_cols < NC, other=0.0).to(tl.float32)
            grad_cond_aff = tl.zeros([BLOCK_M1, BLOCK_K_NC], dtype=tl.float32)

            for x_start in range(0, NX, BLOCK_K_NX):
                x_cols = x_start + tl.arange(0, BLOCK_K_NX)
                x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
                x_offsets = rows[:, None] * stride_yr + x_cols[None, :] * stride_yc

                dy = tl.load(DY + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
                x_hat = tl.load(XHat + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
                gate = tl.load(Gate + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
                dy = tl.where(x_mask, dy, 0.0)
                dscale = dy * tl.where(x_mask, x_hat, 0.0) * gate * (1.0 - gate)

                weight_offsets = (
                    x_cols[:, None] * stride_swr + c_cols[None, :] * stride_swc
                )
                weight_mask = (x_cols[:, None] < NX) & (c_cols[None, :] < NC)
                scale_w = tl.load(
                    ScaleW + weight_offsets,
                    mask=weight_mask,
                    other=0.0,
                ).to(tl.float32)
                bias_w = tl.load(
                    BiasW + (x_cols[:, None] * stride_bwr + c_cols[None, :] * stride_bwc),
                    mask=weight_mask,
                    other=0.0,
                ).to(tl.float32)

                grad_cond_aff = tl.dot(
                    dscale,
                    scale_w,
                    acc=grad_cond_aff,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )
                grad_cond_aff = tl.dot(
                    dy,
                    bias_w,
                    acc=grad_cond_aff,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )

            grad_cond_norm = grad_cond_aff * lnw[None, :]

            sum_grad_cond += tl.sum(tl.where(c_mask, grad_cond_norm, 0.0), axis=1)
            sum_grad_cond_norm += tl.sum(
                tl.where(c_mask, grad_cond_norm * cond_norm, 0.0),
                axis=1,
            )

        c1_cond = sum_grad_cond_norm / NC
        c2_cond = sum_grad_cond / NC
        rstd_cond = tl.load(RstdC + rows, mask=row_mask, other=0.0).to(tl.float32)

        for x_start in range(0, NX, BLOCK_K_NX):
            x_cols = x_start + tl.arange(0, BLOCK_K_NX)
            x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
            x_offsets = rows[:, None] * stride_yr + x_cols[None, :] * stride_yc

            dy = tl.load(DY + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            x_hat = tl.load(XHat + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            gate = tl.load(Gate + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            grad_xhat = dy * gate
            dx = (grad_xhat - (x_hat * c1_x[:, None] + c2_x[:, None])) * rstd_x[:, None]
            tl.store(DX + x_offsets, dx, mask=x_mask)

        for c_start in range(0, NC, BLOCK_K_NC):
            c_cols = c_start + tl.arange(0, BLOCK_K_NC)
            c_mask = row_mask[:, None] & (c_cols[None, :] < NC)
            c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc

            cond_norm = tl.load(CondNorm + c_offsets, mask=c_mask, other=0.0).to(
                tl.float32
            )
            lnw = tl.load(LnW + c_cols, mask=c_cols < NC, other=0.0).to(tl.float32)
            grad_cond_aff = tl.zeros([BLOCK_M1, BLOCK_K_NC], dtype=tl.float32)

            for x_start in range(0, NX, BLOCK_K_NX):
                x_cols = x_start + tl.arange(0, BLOCK_K_NX)
                x_mask = row_mask[:, None] & (x_cols[None, :] < NX)
                x_offsets = rows[:, None] * stride_yr + x_cols[None, :] * stride_yc

                dy = tl.load(DY + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
                x_hat = tl.load(XHat + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
                gate = tl.load(Gate + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
                dy = tl.where(x_mask, dy, 0.0)
                dscale = dy * tl.where(x_mask, x_hat, 0.0) * gate * (1.0 - gate)

                weight_offsets = (
                    x_cols[:, None] * stride_swr + c_cols[None, :] * stride_swc
                )
                weight_mask = (x_cols[:, None] < NX) & (c_cols[None, :] < NC)
                scale_w = tl.load(
                    ScaleW + weight_offsets,
                    mask=weight_mask,
                    other=0.0,
                ).to(tl.float32)
                bias_w = tl.load(
                    BiasW + (x_cols[:, None] * stride_bwr + c_cols[None, :] * stride_bwc),
                    mask=weight_mask,
                    other=0.0,
                ).to(tl.float32)
                grad_cond_aff = tl.dot(
                    dscale,
                    scale_w,
                    acc=grad_cond_aff,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )
                grad_cond_aff = tl.dot(
                    dy,
                    bias_w,
                    acc=grad_cond_aff,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )

            grad_cond_norm = grad_cond_aff * lnw[None, :]
            dcond = (
                grad_cond_norm - (cond_norm * c1_cond[:, None] + c2_cond[:, None])
            ) * rstd_cond[:, None]
            tl.store(DCond + c_offsets, dcond, mask=c_mask)




@triton.autotune(configs=configs_for("adaln_bwd_dw_triton"),
                 # USE_BF16/USE_FP16: tl.dot operand precision, see adaln_fwd_kernel.
                 key=['shape_key', 'NX', 'NC', 'USE_BF16', 'USE_FP16'])
@triton.jit
def adaln_bwd_weight_kernel(
    DY,
    DScaleW,
    DBiasW,
    XHat,
    CondNorm,
    Gate,
    LnW,
    stride_yr,
    stride_yc,
    stride_cr,
    stride_cc,
    stride_swr,
    stride_swc,
    stride_bwr,
    stride_bwc,
    M,
    NX: tl.constexpr,
    NC: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP16: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N_NX: tl.constexpr,
    BLOCK_N_NC: tl.constexpr,
    shape_key,
):
    pid_x = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)

    x_cols = pid_x * BLOCK_N_NX + tl.arange(0, BLOCK_N_NX)
    c_cols = pid_c * BLOCK_N_NC + tl.arange(0, BLOCK_N_NC)
    x_mask = x_cols < NX
    c_mask = c_cols < NC

    lnw = tl.load(LnW + c_cols, mask=c_mask, other=0.0).to(tl.float32)
    acc_scale_w = tl.zeros([BLOCK_N_NX, BLOCK_N_NC], dtype=tl.float32)
    acc_bias_w = tl.zeros([BLOCK_N_NX, BLOCK_N_NC], dtype=tl.float32)

    for row_start in tl.range(0, M, BLOCK_M1):
        rows = row_start + tl.arange(0, BLOCK_M1)
        row_mask = rows < M

        x_offsets = rows[:, None] * stride_yr + x_cols[None, :] * stride_yc
        x_row_mask = row_mask[:, None] & x_mask[None, :]
        dy = tl.load(DY + x_offsets, mask=x_row_mask, other=0.0).to(tl.float32)
        x_hat = tl.load(XHat + x_offsets, mask=x_row_mask, other=0.0).to(tl.float32)
        gate = tl.load(Gate + x_offsets, mask=x_row_mask, other=0.0).to(tl.float32)
        dy = tl.where(x_row_mask, dy, 0.0)
        dscale = dy * tl.where(x_row_mask, x_hat, 0.0) * gate * (1.0 - gate)

        c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc
        c_row_mask = row_mask[:, None] & c_mask[None, :]
        cond_norm = tl.load(CondNorm + c_offsets, mask=c_row_mask, other=0.0).to(
            tl.float32
        )
        cond_aff = tl.where(c_row_mask, cond_norm * lnw[None, :], 0.0)

        dscale_for_dot = dscale
        dy_for_dot = dy
        cond_aff_for_dot = cond_aff
        if USE_BF16:
            dscale_for_dot = dscale.to(tl.bfloat16)
            dy_for_dot = dy.to(tl.bfloat16)
            cond_aff_for_dot = cond_aff.to(tl.bfloat16)
        elif USE_FP16:
            dscale_for_dot = dscale.to(tl.float16)
            dy_for_dot = dy.to(tl.float16)
            cond_aff_for_dot = cond_aff.to(tl.float16)

        acc_scale_w += tl.dot(
            tl.trans(dscale_for_dot),
            cond_aff_for_dot,
            input_precision="ieee",
            out_dtype=tl.float32,
        )
        acc_bias_w += tl.dot(
            tl.trans(dy_for_dot),
            cond_aff_for_dot,
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    scale_offsets = x_cols[:, None] * stride_swr + c_cols[None, :] * stride_swc
    bias_offsets = x_cols[:, None] * stride_bwr + c_cols[None, :] * stride_bwc
    weight_mask = x_mask[:, None] & c_mask[None, :]
    tl.store(DScaleW + scale_offsets, acc_scale_w, mask=weight_mask)
    tl.store(DBiasW + bias_offsets, acc_bias_w, mask=weight_mask)




@triton.autotune(configs=configs_for("adaln_bwd_dlnw_triton"),
                 # USE_BF16/USE_FP16: tl.dot operand precision, see adaln_fwd_kernel.
                 key=['shape_key', 'NX', 'NC', 'USE_BF16', 'USE_FP16'])
@triton.jit
def adaln_bwd_lnw_kernel(
    DY,
    DLnW,
    XHat,
    CondNorm,
    Gate,
    ScaleW,
    BiasW,
    stride_yr,
    stride_yc,
    stride_cr,
    stride_cc,
    stride_swr,
    stride_swc,
    stride_bwr,
    stride_bwc,
    M,
    NX: tl.constexpr,
    NC: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP16: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    shape_key,
):
    pid_c = tl.program_id(0).to(tl.int64)
    c_cols = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = c_cols < NC

    acc_lnw = tl.zeros([BLOCK_N], dtype=tl.float32)

    for row_start in tl.range(0, M, BLOCK_M1):
        rows = row_start + tl.arange(0, BLOCK_M1)
        row_mask = rows < M
        c_offsets = rows[:, None] * stride_cr + c_cols[None, :] * stride_cc
        c_row_mask = row_mask[:, None] & c_mask[None, :]
        cond_norm = tl.load(CondNorm + c_offsets, mask=c_row_mask, other=0.0).to(
            tl.float32
        )
        grad_cond_aff = tl.zeros([BLOCK_M1, BLOCK_N], dtype=tl.float32)

        for x_start in range(0, NX, BLOCK_K):
            x_cols = x_start + tl.arange(0, BLOCK_K)
            x_mask = x_cols < NX
            x_offsets = rows[:, None] * stride_yr + x_cols[None, :] * stride_yc
            x_row_mask = row_mask[:, None] & x_mask[None, :]

            dy = tl.load(DY + x_offsets, mask=x_row_mask, other=0.0).to(tl.float32)
            x_hat = tl.load(XHat + x_offsets, mask=x_row_mask, other=0.0).to(tl.float32)
            gate = tl.load(Gate + x_offsets, mask=x_row_mask, other=0.0).to(tl.float32)
            dy = tl.where(x_row_mask, dy, 0.0)
            dscale = dy * tl.where(x_row_mask, x_hat, 0.0) * gate * (1.0 - gate)

            scale_offsets = x_cols[:, None] * stride_swr + c_cols[None, :] * stride_swc
            weight_mask = x_mask[:, None] & c_mask[None, :]
            scale_w = tl.load(
                ScaleW + scale_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)
            bias_w = tl.load(
                BiasW + (x_cols[:, None] * stride_bwr + c_cols[None, :] * stride_bwc),
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)

            dscale_for_dot = dscale
            dy_for_dot = dy
            scale_w_for_dot = scale_w
            bias_w_for_dot = bias_w
            if USE_BF16:
                dscale_for_dot = dscale.to(tl.bfloat16)
                dy_for_dot = dy.to(tl.bfloat16)
                scale_w_for_dot = scale_w.to(tl.bfloat16)
                bias_w_for_dot = bias_w.to(tl.bfloat16)
            elif USE_FP16:
                dscale_for_dot = dscale.to(tl.float16)
                dy_for_dot = dy.to(tl.float16)
                scale_w_for_dot = scale_w.to(tl.float16)
                bias_w_for_dot = bias_w.to(tl.float16)

            grad_cond_aff = tl.dot(
                dscale_for_dot,
                scale_w_for_dot,
                acc=grad_cond_aff,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            grad_cond_aff = tl.dot(
                dy_for_dot,
                bias_w_for_dot,
                acc=grad_cond_aff,
                input_precision="ieee",
                out_dtype=tl.float32,
            )

        acc_lnw += tl.sum(
            tl.where(c_row_mask, grad_cond_aff * cond_norm, 0.0),
            axis=0,
        )

    tl.store(DLnW + c_cols, acc_lnw, mask=c_mask)


def _adaln_fwd_fake(x_2d, cond_2d, cond_ln_weight, scale_weight, scale_bias, bias_weight,
                    eps_x, eps_cond, use_bf16, use_fp16, shape_key):
    m, nx = x_2d.shape
    nc = cond_2d.shape[1]
    gate_dtype = (torch.bfloat16 if use_bf16 else
                  torch.float16 if use_fp16 else torch.float32)
    return (
        torch.empty_like(x_2d),                              # y
        x_2d.new_empty((m, nx), dtype=torch.float32),         # x_hat
        x_2d.new_empty((m, nc), dtype=torch.float32),         # cond_norm
        x_2d.new_empty((m, nx), dtype=gate_dtype),            # gate
        x_2d.new_empty((m,), dtype=torch.float32),            # rstd_x
        x_2d.new_empty((m,), dtype=torch.float32),            # rstd_cond
    )


@opaque(fake=_adaln_fwd_fake, name="adaln_fwd")
def _adaln_fwd(
    x_2d: torch.Tensor,
    cond_2d: torch.Tensor,
    cond_ln_weight: torch.Tensor,
    scale_weight: torch.Tensor,
    scale_bias: torch.Tensor,
    bias_weight: torch.Tensor,
    eps_x: float,
    eps_cond: float,
    use_bf16: bool,
    use_fp16: bool,
    shape_key: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """The launch -> ``(y, x_hat, cond_norm, gate, rstd_x, rstd_cond)``, all five tail outputs
    being intermediates the backward reuses.

    Split out of ``TritonAdaptiveLayerNormFunction.forward`` so the reshapes and
    ``save_for_backward`` stay traceable -- see ``kernels._compile``. The autocast decision is
    resolved by the CALLER and arrives as ``use_bf16``/``use_fp16``: ``gate``'s dtype depends on
    it, so a fake that could not see it would get the graph's dtypes wrong.
    """
    if not x_2d.is_cuda or not cond_2d.is_cuda:
        msg = "Triton AdaptiveLayerNorm requires CUDA tensors."
        raise RuntimeError(msg)
    if (
        not cond_ln_weight.is_cuda
        or not scale_weight.is_cuda
        or not scale_bias.is_cuda
        or not bias_weight.is_cuda
    ):
        msg = "Triton AdaptiveLayerNorm requires CUDA parameters."
        raise RuntimeError(msg)
    devices = {
        x_2d.device,
        cond_2d.device,
        cond_ln_weight.device,
        scale_weight.device,
        scale_bias.device,
        bias_weight.device,
    }
    if len(devices) != 1:
        msg = "Triton AdaptiveLayerNorm requires all inputs on the same device."
        raise RuntimeError(msg)
    if x_2d.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        msg = f"Unsupported dtype for Triton AdaptiveLayerNorm: {x_2d.dtype}."
        raise TypeError(msg)
    if cond_2d.dtype != x_2d.dtype:
        msg = "x and cond must have the same dtype for Triton AdaptiveLayerNorm."
        raise TypeError(msg)

    m, nx = x_2d.shape
    _, nc = cond_2d.shape
    y = torch.empty_like(x_2d)
    x_hat = torch.empty((m, nx), dtype=torch.float32, device=x_2d.device)
    cond_norm = torch.empty((m, nc), dtype=torch.float32, device=x_2d.device)
    gate_dtype = (torch.bfloat16 if use_bf16 else
                  torch.float16 if use_fp16 else torch.float32)
    gate = torch.empty((m, nx), dtype=gate_dtype, device=x_2d.device)
    rstd_x = torch.empty(m, dtype=torch.float32, device=x_2d.device)
    rstd_cond = torch.empty(m, dtype=torch.float32, device=x_2d.device)

    grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M1"])]
    adaln_fwd_kernel[grid](
        x_2d,
        cond_2d,
        cond_ln_weight,
        scale_weight,
        scale_bias,
        bias_weight,
        y,
        x_hat,
        cond_norm,
        gate,
        rstd_x,
        rstd_cond,
        x_2d.stride(0),
        x_2d.stride(1),
        cond_2d.stride(0),
        cond_2d.stride(1),
        scale_weight.stride(0),
        scale_weight.stride(1),
        bias_weight.stride(0),
        bias_weight.stride(1),
        m,
        NX=nx,
        NC=nc,
        eps_x=eps_x,
        eps_cond=eps_cond,
        USE_BF16=use_bf16,
        USE_FP16=use_fp16,
        shape_key=shape_key,
    )
    return y, x_hat, cond_norm, gate, rstd_x, rstd_cond


def _adaln_bwd_fake(grad_output_2d, x_hat, cond_norm, gate, cond_ln_weight, scale_weight,
                    bias_weight, rstd_x, rstd_cond, use_bf16, use_fp16, shape_key):
    m, nx = grad_output_2d.shape
    nc = cond_norm.shape[1]
    f32 = torch.float32
    return (
        torch.empty_like(grad_output_2d),                        # dx
        grad_output_2d.new_empty((m, nc), dtype=f32),             # dcond
        torch.empty_like(cond_ln_weight, dtype=f32),              # dlnw
        torch.empty_like(scale_weight, dtype=f32),                # dscale_w
        grad_output_2d.new_empty((nx,), dtype=f32),               # dscale_b
        torch.empty_like(bias_weight, dtype=f32),                 # dbias_w
    )


@opaque(fake=_adaln_bwd_fake, name="adaln_bwd")
def _adaln_bwd(
    grad_output_2d: torch.Tensor,
    x_hat: torch.Tensor,
    cond_norm: torch.Tensor,
    gate: torch.Tensor,
    cond_ln_weight: torch.Tensor,
    scale_weight: torch.Tensor,
    bias_weight: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_cond: torch.Tensor,
    use_bf16: bool,
    use_fp16: bool,
    shape_key: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """The three backward launches, every gradient still FLAT and in its accumulation dtype.

    The reshapes back to the caller's shapes and the casts back to the parameters' dtypes are
    plain torch, so they stay outside where the compiler can fuse them.
    """
    m, nx = grad_output_2d.shape
    _, nc = cond_norm.shape

    dx = torch.empty_like(grad_output_2d)
    dcond = torch.empty((m, nc), dtype=torch.float32, device=grad_output_2d.device)
    dlnw = torch.empty_like(cond_ln_weight, dtype=torch.float32)
    dscale_w = torch.empty_like(scale_weight, dtype=torch.float32)
    dscale_b = torch.zeros((nx,), dtype=torch.float32, device=grad_output_2d.device)
    dbias_w = torch.empty_like(bias_weight, dtype=torch.float32)

    input_grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M1"])]
    adaln_bwd_input_kernel[input_grid](
        grad_output_2d,
        dx,
        dcond,
        dscale_b,
        x_hat,
        cond_norm,
        gate,
        cond_ln_weight,
        scale_weight,
        bias_weight,
        rstd_x,
        rstd_cond,
        grad_output_2d.stride(0),
        grad_output_2d.stride(1),
        cond_norm.stride(0),
        cond_norm.stride(1),
        scale_weight.stride(0),
        scale_weight.stride(1),
        bias_weight.stride(0),
        bias_weight.stride(1),
        m,
        NX=nx,
        NC=nc,
        USE_BF16=use_bf16,
        USE_FP16=use_fp16,
        shape_key=shape_key,
    )
    weight_grid = lambda meta: (
        triton.cdiv(nx, meta["BLOCK_N_NX"]),
        triton.cdiv(nc, meta["BLOCK_N_NC"]),
    )
    adaln_bwd_weight_kernel[weight_grid](
        grad_output_2d,
        dscale_w,
        dbias_w,
        x_hat,
        cond_norm,
        gate,
        cond_ln_weight,
        grad_output_2d.stride(0),
        grad_output_2d.stride(1),
        cond_norm.stride(0),
        cond_norm.stride(1),
        scale_weight.stride(0),
        scale_weight.stride(1),
        bias_weight.stride(0),
        bias_weight.stride(1),
        m,
        NX=nx,
        NC=nc,
        USE_BF16=use_bf16,
        USE_FP16=use_fp16,
        shape_key=shape_key,
    )
    lnw_grid = lambda meta: [triton.cdiv(nc, meta["BLOCK_N"])]
    adaln_bwd_lnw_kernel[lnw_grid](
        grad_output_2d,
        dlnw,
        x_hat,
        cond_norm,
        gate,
        scale_weight,
        bias_weight,
        grad_output_2d.stride(0),
        grad_output_2d.stride(1),
        cond_norm.stride(0),
        cond_norm.stride(1),
        scale_weight.stride(0),
        scale_weight.stride(1),
        bias_weight.stride(0),
        bias_weight.stride(1),
        m,
        NX=nx,
        NC=nc,
        USE_BF16=use_bf16,
        USE_FP16=use_fp16,
        shape_key=shape_key,
    )

    return dx, dcond, dlnw, dscale_w, dscale_b, dbias_w


class TritonAdaptiveLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_ln_weight: torch.Tensor,
        scale_weight: torch.Tensor,
        scale_bias: torch.Tensor,
        bias_weight: torch.Tensor,
        eps_x: float,
        eps_cond: float,
    ) -> torch.Tensor:
        compute_dtype = x.dtype
        if torch.is_autocast_enabled():
            compute_dtype = torch.get_autocast_dtype("cuda")

        orig_x_shape = x.shape
        orig_cond_shape = cond.shape
        x_2d = x.reshape(-1, orig_x_shape[-1]).contiguous()
        cond_2d = cond.reshape(-1, orig_cond_shape[-1]).contiguous()

        if x_2d.shape[0] != cond_2d.shape[0]:
            msg = "AdaptiveLayerNorm expects x and cond to share leading dimensions."
            raise ValueError(msg)

        use_bf16 = compute_dtype == torch.bfloat16
        use_fp16 = compute_dtype == torch.float16
        y, x_hat, cond_norm, gate, rstd_x, rstd_cond = _adaln_fwd(
            x_2d, cond_2d, cond_ln_weight, scale_weight, scale_bias, bias_weight,
            eps_x, eps_cond, use_bf16, use_fp16,
            # L for the autotune key: x's pre-flatten shape[-2] (the atom count), not m = B*A.
            atom_key(length_of(orig_x_shape)),
        )

        ctx.save_for_backward(
            x_hat,
            cond_norm,
            gate,
            cond_ln_weight,
            scale_weight,
            bias_weight,
            rstd_x,
            rstd_cond,
        )
        ctx.orig_x_shape = orig_x_shape
        ctx.orig_cond_shape = orig_cond_shape
        ctx.x_dtype = x.dtype
        ctx.cond_dtype = cond.dtype
        ctx.cond_ln_weight_dtype = cond_ln_weight.dtype
        ctx.scale_weight_dtype = scale_weight.dtype
        ctx.scale_bias_dtype = scale_bias.dtype
        ctx.bias_weight_dtype = bias_weight.dtype
        ctx.use_bfloat16 = use_bf16
        ctx.use_float16 = use_fp16
        return y.reshape(orig_x_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_hat,
            cond_norm,
            gate,
            cond_ln_weight,
            scale_weight,
            bias_weight,
            rstd_x,
            rstd_cond,
        ) = ctx.saved_tensors

        dx, dcond, dlnw, dscale_w, dscale_b, dbias_w = _adaln_bwd(
            grad_output.reshape(-1, grad_output.shape[-1]).contiguous(),
            x_hat, cond_norm, gate, cond_ln_weight, scale_weight, bias_weight,
            rstd_x, rstd_cond, ctx.use_bfloat16, ctx.use_float16,
            # Same L as the forward: the pre-flatten atom count, kept on ctx for exactly this.
            atom_key(length_of(ctx.orig_x_shape)),
        )
        return (
            dx.reshape(ctx.orig_x_shape).to(ctx.x_dtype),
            dcond.reshape(ctx.orig_cond_shape).to(ctx.cond_dtype),
            dlnw.to(ctx.cond_ln_weight_dtype),
            dscale_w.to(ctx.scale_weight_dtype),
            dscale_b.to(ctx.scale_bias_dtype),
            dbias_w.to(ctx.bias_weight_dtype),
            None,
            None,
        )


triton_adaptive_layer_norm = TritonAdaptiveLayerNormFunction.apply
