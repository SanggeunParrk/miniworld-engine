from __future__ import annotations

import torch
import triton
import triton.language as tl

from .main import get_seq_group, layer_norm_fwd_fused


def _bwd_block_m(n: int) -> int:
    if n <= 256:
        return 128
    return 64


def _bwd_num_warps(n: int) -> int:
    if n <= 256:
        return 4
    return 8


@triton.jit
def _layer_norm_bwd_dx_partials(
    DX,
    PART_DW,
    PART_DB,
    DY,
    X,
    W,
    Mean,
    Rstd,
    stride_part_row,
    stride_r,
    stride_c,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    col_mask = offs_n < N

    w = tl.load(W + offs_n, mask=col_mask, other=1.0).to(tl.float32)
    partial_dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    partial_db = tl.zeros([BLOCK_N], dtype=tl.float32)

    for ri in range(BLOCK_M):
        row = pid * BLOCK_M + ri
        row_mask = row < M
        offs = row * stride_r + offs_n * stride_c
        mask = row_mask & col_mask

        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + row, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row, mask=row_mask, other=0.0).to(tl.float32)

        xhat = (x - mean) * rstd
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, w * dy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd

        tl.store(DX + offs, dx, mask=mask)
        partial_dw += dy * xhat
        partial_db += dy

    part_offs = pid * stride_part_row + offs_n
    tl.store(PART_DW + part_offs, partial_dw, mask=col_mask)
    tl.store(PART_DB + part_offs, partial_db, mask=col_mask)


class TritonLayerNormPartialReductionFunction(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = torch.empty_like(x_2d)
        m, n = x_2d.shape

        mean = torch.empty(m, dtype=torch.float32, device=x.device)
        rstd = torch.empty(m, dtype=torch.float32, device=x.device)

        grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M"])]
        layer_norm_fwd_fused[grid](
            x_2d,
            y_2d,
            weight,
            bias,
            mean,
            rstd,
            rstd,  # placeholder when HAS_ROWSCALE=False; keep the call shape identical to main.py
            x_2d.stride(0),
            x_2d.stride(1),
            m,
            n,
            eps,
            BLOCK_N=triton.next_power_of_2(n),
            GROUP_M=get_seq_group(m),
            HAS_ROWSCALE=False,
        )

        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
        m, n = x.shape

        block_m = _bwd_block_m(n)
        block_n = triton.next_power_of_2(n)
        num_warps = _bwd_num_warps(n)
        num_partials = triton.cdiv(m, block_m)

        dx_2d = torch.empty_like(dy_2d)
        partial_dw = torch.empty((num_partials, n), dtype=torch.float32, device=x.device)
        partial_db = torch.empty((num_partials, n), dtype=torch.float32, device=x.device)

        _layer_norm_bwd_dx_partials[(num_partials,)](
            dx_2d,
            partial_dw,
            partial_db,
            dy_2d,
            x,
            weight,
            mean,
            rstd,
            partial_dw.stride(0),
            x.stride(0),
            x.stride(1),
            m,
            N=n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

        dw = partial_dw.sum(dim=0).to(weight.dtype)
        db = partial_db.sum(dim=0).to(weight.dtype)
        return dx_2d.view(ctx.input_shape), dw, db, None


triton_layernorm_partial = TritonLayerNormPartialReductionFunction.apply
