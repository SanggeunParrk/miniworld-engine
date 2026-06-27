"""Persistent grid-stride LayerNorm backward (triton port of quack's bwd algorithm).

The shipped partial path (`triton/partial.py`) has three algorithmic weaknesses
that the cute-vs-triton bench (`benchmark/CUTE_INVESTIGATION.md`) pinned as the
reason quack's CuTeDSL backward is 1.2-2.3x faster:

1. a scalar ``for ri in range(BLOCK_M)`` row loop instead of a vectorized 2D tile;
2. ``cdiv(M, block_m)`` partial dw/db rows (~16k at M=1M) -> a large partial
   buffer + final reduce;
3. ``BLOCK_N = next_pow2(N)`` register pressure (shared with the forward).

This kernel fixes (1) and (2): a PERSISTENT grid of ``NUM_SM * waves`` blocks
grid-strides over the row tiles, each block carrying its dw/db accumulators in
fp32 registers across the whole stride loop and writing exactly ONE partial row at
the end. The partial buffer is therefore ``[grid, N]`` (~hundreds of rows), so the
final reduce is tiny. Each tile is a vectorized ``[BLOCK_M, BLOCK_N]`` load (like
the atomic kernel) — no per-row loop, no atomics.

Forward is unchanged (reuses ``layer_norm_fwd_fused``); only the backward differs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .main import get_seq_group, layer_norm_fwd_fused

# Persistent grid = (#SMs * PERSIST_WAVES) blocks. A couple of blocks per SM keeps
# the memory system saturated while keeping the partial buffer (and its final
# reduce) to a few hundred rows instead of ~cdiv(M, BLOCK_M).
PERSIST_WAVES = 2


_bwd_configs = [
    triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=num_stages)
    for block_m in [4, 8, 16, 32, 64]
    for num_warps in [4, 8, 16]
    for num_stages in [1, 2, 3]
]


# fmt: off
@triton.autotune(configs=_bwd_configs, key=["N"])
@triton.jit
def _ln_bwd_persistent(
    DX, PART_DW, PART_DB, DY, X, W, Mean, Rstd,
    stride_part, stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)

    offs_n = tl.arange(0, BLOCK_N)
    col_mask = offs_n < N
    w = tl.load(W + offs_n, mask=col_mask, other=0.0).to(tl.float32)

    acc_dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_N], dtype=tl.float32)

    num_tiles = tl.cdiv(M, BLOCK_M)
    for tile in range(pid, num_tiles, G):
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        p = rows[:, None] * stride_r + offs_n[None, :] * stride_c
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(X + p, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + p, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)

        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        wdy = tl.where(mask, w[None, :] * dy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=1) / N
        c2 = tl.sum(wdy, axis=1) / N
        dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
        tl.store(DX + p, dx, mask=mask)

        acc_dw += tl.sum(dy * xhat, axis=0)
        acc_db += tl.sum(dy, axis=0)

    part = pid * stride_part + offs_n
    tl.store(PART_DW + part, acc_dw, mask=col_mask)
    tl.store(PART_DB + part, acc_db, mask=col_mask)
# fmt: on


def _persistent_grid(device: torch.device) -> int:
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    return sm * PERSIST_WAVES


class TritonLayerNormPersistentFunction(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = torch.empty_like(x_2d)
        m, n = x_2d.shape
        mean = torch.empty(m, dtype=torch.float32, device=x.device)
        rstd = torch.empty(m, dtype=torch.float32, device=x.device)
        grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M"])]
        layer_norm_fwd_fused[grid](
            x_2d, y_2d, weight, bias, mean, rstd, rstd,
            x_2d.stride(0), x_2d.stride(1),
            m, n, eps,
            BLOCK_N=triton.next_power_of_2(n),
            GROUP_M=get_seq_group(m), HAS_ROWSCALE=False,
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

        g = _persistent_grid(x.device)
        block_n = triton.next_power_of_2(n)
        dx_2d = torch.empty_like(dy_2d)
        partial_dw = torch.empty((g, n), dtype=torch.float32, device=x.device)
        partial_db = torch.empty((g, n), dtype=torch.float32, device=x.device)

        _ln_bwd_persistent[(g,)](
            dx_2d, partial_dw, partial_db, dy_2d, x, weight, mean, rstd,
            partial_dw.stride(0), x.stride(0), x.stride(1),
            m, N=n, BLOCK_N=block_n,
        )

        dw = partial_dw.sum(dim=0).to(weight.dtype)
        db = partial_db.sum(dim=0).to(weight.dtype)
        return dx_2d.view(ctx.input_shape), dw, db, None


triton_layernorm_persistent = TritonLayerNormPersistentFunction.apply
