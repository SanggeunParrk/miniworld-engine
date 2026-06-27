# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/layernorm.py
import os

import torch
import triton
import triton.language as tl


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [
        32 * 32,
        64 * 64,
        128 * 128,
        256 * 256,
        384 * 384,
        512 * 512,
        768 * 768,
        48 * 128,
        48 * 512,
        48 * 256,
        48 * 384,
    ]
    GROUP_LENGTHS = sorted(GROUP_LENGTHS)
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0") == "layernorm"


configs = [
    triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=num_stages)
    for block_m in [1, 2, 4, 8, 16, 32, 64]
    for num_warps in [4, 8, 16]
    for num_stages in [2, 3, 4, 5]
]

fwd_configs = [
    triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=4),  # 128, 256 at H100
    triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=3),  # 384 at H100
]

bwd_configs = [
    triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),  # 128 at H100
    triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=4),  # 256 at H100
    triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),  # 384 at H100
]

fwd_configs = configs
bwd_configs = configs


# fmt: off
@triton.autotune(configs=fwd_configs, key=["N", "GROUP_M"])
@triton.jit
def layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, Rowscale,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr, HAS_ROWSCALE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
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
@triton.autotune(configs=fwd_configs, key=["N", "GROUP_M"])
@triton.jit
def layer_norm_fwd_fused_recal(
    X, Y, W, B, Mean, Rstd,
    stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
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
@triton.autotune(configs=bwd_configs, key=["N", "GROUP_M"], reset_to_zero=["DW", "DB"])
@triton.jit
def layer_norm_bwd_dx_fused(
    DX, DY, DW, DB,
    X, W, Mean, Rstd,
    stride_wc, stride_bc, stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)

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
    ):
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = torch.empty_like(x_2d)
        M, N = x_2d.shape

        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_fwd_fused[grid](
            x_2d, y_2d, weight, bias, mean, rstd, rstd,  # Rowscale=rstd placeholder (unused)
            x_2d.stride(0), x_2d.stride(1),
            M, N, eps,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M), HAS_ROWSCALE=False,
        )
        # fmt: on

        ctx.save_for_backward(
            x_2d.to(torch.bfloat16),
            weight,
            mean,
            rstd,
        )
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        x = x.to(dy.dtype)
        M, N = x.shape
        dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()

        # allocate output
        dx_2d = torch.empty_like(dy_2d)
        dw = torch.zeros(N, dtype=torch.float32, device=x.device)
        db = torch.zeros(N, dtype=torch.float32, device=x.device)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_bwd_dx_fused[grid](
            dx_2d, dy_2d, dw, db,
            x, weight, mean, rstd,
            dw.stride(0), db.stride(0), x.stride(0), x.stride(1),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        return (
            dx_2d.view(ctx.input_shape),
            dw,
            db,
            None,
        )


triton_layernorm = TritonLayerNormFunction.apply


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
