# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/layernorm.py
import os

import torch
import triton
from miniworld_engine.autotune.grids import brute, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_1D
import triton.language as tl

from miniworld_engine.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_mixed
    return bucket_mixed(length)


AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0") == "layernorm"

# Opt-in: route the LN backward (dx/dw/db) through the hand-CUDA warp-per-row kernel for the regime
# where it beats the triton atomic path STANDALONE (bf16, 128<=N<=512, contiguous, no per-row
# scale) — measured ~1.1x over the triton atomic bwd on B200. DEFAULT OFF because the win is
# e2e-NEUTRAL in the trimul training graph: LN_in bwd is ~7% of the step and the CUDA path's second
# (reduce) kernel launch eats the ~12% compute win at these M (verified graph-time delta within
# run-to-run noise at L=384/768/1024). Enabling also forces the one-time nvcc JIT build on first
# `triton_layernorm` import. Set MINIWORLD_LN_IN_CUDA=1 to use it (kernels/layernorm/cuda/ now
# gencodes sm_80/90/100 + PTX so the ext loads and runs on B200).
_LN_CUDA_BWD_ENABLED = os.environ.get("MINIWORLD_LN_IN_CUDA", "0") != "0"


configs = brute({"BLOCK_M": BLOCK_M})

fwd_configs = brute({"BLOCK_M": BLOCK_M})

bwd_configs = brute({"BLOCK_M": BLOCK_M})

fwd_configs = configs
bwd_configs = configs


# fmt: off
_layernorm_main_fwd_prune = make_cache_prune(
    "layernorm_main_fwd", dtype_of=tensor_dtype_of("X"),
    bucket_of=key_bucket_of("N", "GROUP_M"),
)


@triton.autotune(configs=fwd_configs, key=["N", "GROUP_M"],
                 prune_configs_by={"early_config_prune": _layernorm_main_fwd_prune})
@triton.jit
def layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, Rowscale,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr, HAS_ROWSCALE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0).to(tl.int64)
    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    row_mask = offset_row < M

    offset_row = offset_row[:, None] * stride_r
    row_mask = row_mask[:, None]
    offset_col = tl.arange(0, BLOCK_N)
    col_mask = offset_col < N
    offset_col = offset_col[None, :] * stride_c
    col_mask = col_mask[None, :]
    mask = row_mask & col_mask

    Y += offset_row + offset_col
    X += offset_row + offset_col
    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _mean += x
    mean = tl.sum(_mean, axis=1) / N
    # Compute variance
    _var = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    x = x - mean[:, None]
    _var += x * x
    var = tl.sum(_var, axis=1) / N
    var -= (BLOCK_N - N) / N * mean * mean
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    # tl.store(Mean, mean, mask=row_mask)
    mean_offset = tl.arange(0, BLOCK_M) + row * BLOCK_M
    mean_mask = mean_offset < M
    tl.store(Mean + mean_offset, mean, mask=mean_mask)
    tl.store(Rstd + mean_offset, rstd, mask=mean_mask)
    # Normalize and apply linear transformation
    offset_col = tl.arange(0, BLOCK_N)
    col_mask = offset_col < N
    W = W + offset_col * stride_c
    B = B + offset_col * stride_c
    w = tl.load(W, mask=col_mask)
    b = tl.load(B, mask=col_mask)
    x_hat = x * rstd[:, None]
    y = x_hat * w + b
    if HAS_ROWSCALE:  # fold a per-row scale (e.g. AF pair-mask) into the LN epilogue — free
        rs_off = tl.arange(0, BLOCK_M) + row * BLOCK_M
        rs = tl.load(Rowscale + rs_off, mask=rs_off < M, other=0.0).to(tl.float32)
        y = y * rs[:, None]
    tl.store(Y, y, mask=mask)
# fmt: on


# fmt: off
_layernorm_main_fwd_recal_prune = make_cache_prune(
    "layernorm_main_fwd_recal", dtype_of=tensor_dtype_of("X"),
    bucket_of=key_bucket_of("N", "GROUP_M"),
)


@triton.autotune(configs=fwd_configs, key=["N", "GROUP_M"],
                 prune_configs_by={"early_config_prune": _layernorm_main_fwd_recal_prune})
@triton.jit
def layer_norm_fwd_fused_recal(
    X, Y, W, B, Mean, Rstd,
    stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0).to(tl.int64)
    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    row_mask = offset_row < M

    offset_row = offset_row[:, None] * stride_r
    row_mask = row_mask[:, None]
    offset_col = tl.arange(0, BLOCK_N)
    col_mask = offset_col < N
    offset_col = offset_col[None, :] * stride_c
    col_mask = col_mask[None, :]
    mask = row_mask & col_mask

    Y += offset_row + offset_col
    X += offset_row + offset_col
    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)

    # tl.store(Mean, mean, mask=row_mask)
    mean_offset = tl.arange(0, BLOCK_M) + row * BLOCK_M
    mean_mask = mean_offset < M
    mean = tl.load(Mean + mean_offset, mask=mean_mask)
    rstd = tl.load(Rstd + mean_offset, mask=mean_mask)
    x = x - mean[:, None]
    # Normalize and apply linear transformation
    offset_col = tl.arange(0, BLOCK_N)
    col_mask = offset_col < N
    W = W + offset_col * stride_c
    B = B + offset_col * stride_c
    w = tl.load(W, mask=col_mask)
    b = tl.load(B, mask=col_mask)
    x_hat = x * rstd[:, None]
    y = x_hat * w + b
    tl.store(Y, y, mask=mask)
# fmt: on


# fmt: off
_layernorm_main_bwd_dx_prune = make_cache_prune(
    "layernorm_main_bwd_dx", dtype_of=tensor_dtype_of("DY"),
    bucket_of=key_bucket_of("N", "GROUP_M"),
)


@triton.autotune(configs=bwd_configs, key=["N", "GROUP_M"], reset_to_zero=["DW", "DB"],
                 prune_configs_by={"early_config_prune": _layernorm_main_bwd_dx_prune})
@triton.jit
def layer_norm_bwd_dx_fused(
    DX, DY, DW, DB,
    X, W, Mean, Rstd, Rowscale,
    stride_wc, stride_bc, stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr, HAS_ROWSCALE: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0).to(tl.int64)

    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    row_mask = offset_row < M
    offset_row = offset_row[:, None] * stride_r
    row_mask = row_mask[:, None]
    offset_col = tl.arange(0, BLOCK_N)
    col_mask = offset_col < N
    mask = row_mask & col_mask

    offset_col = offset_col[None, :] * stride_c
    X += offset_row + offset_col
    Mean = Mean + (tl.arange(0, BLOCK_M) + row * BLOCK_M)
    Rstd = Rstd + (tl.arange(0, BLOCK_M) + row * BLOCK_M)

    DY += offset_row + offset_col
    DX += offset_row + offset_col
    # Offset locks and weights/biases gradient pointer for parallel reduction
    offset_col = tl.arange(0, BLOCK_N)
    W = W + offset_col
    DW = DW + offset_col
    DB = DB + offset_col
    # Load data to SRAM
    x = tl.load(X, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY, mask=mask, other=0).to(tl.float32)
    if HAS_ROWSCALE:  # bwd of y = LN(x)*rs: scale incoming grad by rs (then dx/dw/db all follow)
        rs_idx = tl.arange(0, BLOCK_M) + row * BLOCK_M
        rs = tl.load(Rowscale + rs_idx, mask=rs_idx < M, other=0.0).to(tl.float32)
        dy = dy * rs[:, None]
    w = tl.load(W).to(tl.float32)
    mean = tl.load(Mean).to(tl.float32)
    rstd = tl.load(Rstd).to(tl.float32)
    # Compute dx
    xhat = (x - mean[:, None]) * rstd[:, None]
    wdy = w[None, :] * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=1) / N
    c2 = tl.sum(wdy, axis=1) / N
    dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    # Write dx
    tl.store(DX, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)

    partial_dw = tl.sum(partial_dw, axis=0)
    partial_db = tl.sum(partial_db, axis=0)

    # tl.store(DW, partial_dw)
    # tl.store(DB, partial_db)

    tl.atomic_add(DW, partial_dw, mask=col_mask)
    tl.atomic_add(DB, partial_db, mask=col_mask)
# fmt: on


class TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
        row_scale: torch.Tensor | None = None,
    ):
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = torch.empty_like(x_2d)
        M, N = x_2d.shape

        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        has_rs = row_scale is not None
        # per-row scale (e.g. AF pair-mask) folded into the LN epilogue — y = LN(x)*rs, FREE
        # (no extra (M,N) multiply / HBM round-trip). rs reshaped to [M], broadcast over N.
        rs = row_scale.reshape(-1).to(x_2d.dtype).contiguous() if has_rs else rstd
        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_fwd_fused[grid](
            x_2d, y_2d, weight, bias, mean, rstd, rs,  # rs (or rstd placeholder when no rowscale)
            x_2d.stride(0), x_2d.stride(1),
            M, N, eps,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M), HAS_ROWSCALE=has_rs,
        )
        # fmt: on

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
    @torch.compiler.disable()
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd, rs = ctx.saved_tensors
        x = x.to(dy.dtype)
        M, N = x.shape
        dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()

        has_rs = ctx.has_rowscale

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
        if (x.dtype == torch.bfloat16 and 128 <= N <= 512 and (has_rs or _LN_CUDA_BWD_ENABLED)):
            try:
                from ..cuda import layer_norm_bwd_cuda
                dx_c, dw_c, db_c = layer_norm_bwd_cuda(
                    dy_2d, x.contiguous(), weight, mean, rstd,
                    row_scale=rs if has_rs else None,
                )
                return (dx_c.view(ctx.input_shape), dw_c.float(), db_c.float(), None, None)
            except Exception:  # noqa: BLE001 - portable triton fallback on any CUDA-path failure
                pass

        # allocate output
        dx_2d = torch.empty_like(dy_2d)
        dw = torch.zeros(N, dtype=torch.float32, device=x.device)
        db = torch.zeros(N, dtype=torch.float32, device=x.device)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_bwd_dx_fused[grid](
            dx_2d, dy_2d, dw, db,
            x, weight, mean, rstd, rs if has_rs else rstd,  # rs folds the mask grad in (free)
            dw.stride(0), db.stride(0), x.stride(0), x.stride(1),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M), HAS_ROWSCALE=has_rs,
        )
        # fmt: on

        return (
            dx_2d.view(ctx.input_shape),
            dw,
            db,
            None,
            None,
        )


def triton_layernorm(x, weight, bias, eps, row_scale=None):
    """LayerNorm (autograd). Optional `row_scale` [M] folds a per-row scale into the LN epilogue
    (fwd) and the grad into the LN backward (bwd) — y = LN(x)*rs, the AF pair-mask applied FREE
    (no separate (M,N) multiply). rs=None -> plain LN."""
    return TritonLayerNormFunction.apply(x, weight, bias, eps, row_scale)


@torch.compiler.disable()
def triton_layernorm_masked(x, weight, bias, eps, row_scale):
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
    grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
    layer_norm_fwd_fused[grid](
        x_2d, y_2d, weight, bias, mean, rstd, rs,
        x_2d.stride(0), x_2d.stride(1),
        M, N, eps,
        BLOCK_N=triton.next_power_of_2(N),
        GROUP_M=get_seq_group(M), HAS_ROWSCALE=True,
    )
    # fmt: on
    return y_2d.view_as(x)
