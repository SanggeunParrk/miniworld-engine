"""Persistent grid-stride LayerNorm backward (triton port of quack's bwd algorithm).

The shipped partial path (`triton/partial.py`) has three algorithmic weaknesses
that the archived cute-vs-triton bench report pinned as the
reason quack's CuTeDSL backward is 1.2-2.3x faster:

1. a scalar ``for ri in range(BLOCK_M1)`` row loop instead of a vectorized 2D tile;
2. ``cdiv(M, block_m)`` partial dw/db rows (~16k at M=1M) -> a large partial
   buffer + final reduce;
3. ``BLOCK_N = next_pow2(N)`` register pressure (shared with the forward).

(3) is now fixed as well: ``BLOCK_N`` is a tuned tile from the autotune grid
(a CSV tile), and the column axis is
grid axis 1 so the fp32 dγ/dβ register accumulators stay BLOCK_N wide.

This kernel fixes (1) and (2): a PERSISTENT grid of ``NUM_SM * waves`` blocks
grid-strides over the row tiles, each block carrying its dw/db accumulators in
fp32 registers across the whole stride loop and writing exactly ONE partial row at
the end. The partial buffer is therefore ``[grid, N]`` (~hundreds of rows), so the
final reduce is tiny. Each tile is a vectorized ``[BLOCK_M1, BLOCK_N]`` load (like
the atomic kernel) — no per-row loop, no atomics.

Forward is unchanged (reuses ``layer_norm_fwd_fused``); only the backward differs.
"""

from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of
from .main import get_seq_group, layer_norm_fwd_fused

# Persistent grid = (#SMs * PERSIST_WAVES) blocks on axis 0. A couple of blocks per SM keeps
# the memory system saturated while keeping the partial buffer (and its final
# reduce) to a few hundred rows instead of ~cdiv(M, BLOCK_M1).
PERSIST_WAVES = 2


# BLOCK_N was weakness (3) in this module's own docstring — ``next_pow2(N)`` arriving from the
# launcher. It is a CSV tile; a row at or above the extent keeps the whole-row
# schedule stays in the sweep (N = d_hidden runs 128..1024; the canonical BLOCK_N stops at 256 and
# would have made a multi-tile column pass compulsory).


# fmt: off


# shape_key is in the key: the grid is FIXED at g programs, so BLOCK_M1 alone sets num_tiles and
# whether the persistent grid is even filled -- the row count picks the winner.
@triton.autotune(configs=configs_for("layernorm_bwd_split_triton"), key=['N', 'shape_key'])
@triton.jit
def _ln_bwd_persistent(
    DX, PART_DW, PART_DB, DY, X, W, Mean, Rstd,
    stride_part, stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key,
):
    # Grid is 2-D: axis 0 = the persistent programs that grid-stride over row tiles (this is the
    # axis PART_DW/PART_DB are indexed by), axis 1 = the column tile this program owns. The column
    # axis has to be a GRID axis, not an inner loop, because dγ/dβ are accumulated per COLUMN in
    # fp32 registers across the whole row-tile stride loop -- that register accumulator is the
    # whole point of this kernel (no atomics), and it can only be BLOCK_K wide. Splitting columns
    # across programs keeps it BLOCK_K wide and keeps each program's slice of the [G, N] partial
    # row disjoint, so the partial buffer and the final reduce are unchanged.
    pid = tl.program_id(0).to(tl.int64)
    G = tl.num_programs(0)
    pid_n = tl.program_id(1).to(tl.int64)

    offs_n = pid_n * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = offs_n < N
    w = tl.load(W + offs_n, mask=col_mask, other=0.0).to(tl.float32)

    acc_dw = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_K], dtype=tl.float32)

    num_tiles = tl.cdiv(M, BLOCK_M1)
    for tile in range(pid, num_tiles, G):
        rows = tile * BLOCK_M1 + tl.arange(0, BLOCK_M1)
        row_mask = rows < M
        mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)

        # COVERING TILE (BLOCK_K >= N): the c1/c2 gather loop below re-reads X and DY for the WHOLE
        # row, and the dx/dγ/dβ tail then reads this program's own column tile again. At a covering
        # tile those are the SAME addresses (the launcher's grid is
        # `(G, cdiv(N, BLOCK_K))`, so the column axis is exactly ONE block wide and pid_n == 0), but
        # they are not CSE'd -- the gather sits in its own scf.for region and the DX tl.store lands
        # between the two, and Triton cannot prove the raw pointers do not alias. So the covering
        # config read X and DY twice per row tile. Both N and BLOCK_K are tl.constexpr, so the
        # guard is resolved at TRACE time and only ONE branch is emitted; the grid collapse that
        # makes pid_n provably 0 is the launcher's cdiv, not an assumption -- and offs_n is still
        # written pid_n-relative here, so a wider grid would simply mask everything off rather than
        # compute a wrong row. The fp32 register accumulators acc_dw/acc_db are untouched: the fast
        # path feeds them exactly as the column-grid path does, so this stays atomic-free and the
        # [G, N] partial-buffer contract is unchanged.
        if BLOCK_K >= N:
            p = rows[:, None] * stride_r + offs_n[None, :] * stride_c
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + p, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + p, mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            wdy = tl.where(mask, w[None, :] * dy, 0.0)
            c1 = tl.sum(xhat * wdy, axis=1) / N
            c2 = tl.sum(wdy, axis=1) / N
            dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX + p, dx, mask=mask)

            acc_dw += tl.sum(dy * xhat, axis=0)
            acc_db += tl.sum(dy, axis=0)
        else:
            # c1/c2 reduce over the WHOLE row, so they are gathered over every column tile before
            # any dx element of this program's tile can be written.
            c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
            c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
            for n0 in range(0, N, BLOCK_K):
                cn = n0 + tl.arange(0, BLOCK_K)
                cm = cn < N
                m2 = row_mask[:, None] & cm[None, :]
                q = rows[:, None] * stride_r + cn[None, :] * stride_c
                xr = tl.load(X + q, mask=m2, other=0.0).to(tl.float32)
                dyr = tl.load(DY + q, mask=m2, other=0.0).to(tl.float32)
                wr = tl.load(W + cn, mask=cm, other=0.0).to(tl.float32)
                xhat_r = tl.where(m2, (xr - mean[:, None]) * rstd[:, None], 0.0)
                wdy_r = tl.where(m2, wr[None, :] * dyr, 0.0)
                c1 += tl.sum(xhat_r * wdy_r, axis=1)
                c2 += tl.sum(wdy_r, axis=1)
            c1 = c1 / N
            c2 = c2 / N

            p = rows[:, None] * stride_r + offs_n[None, :] * stride_c
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + p, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + p, mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            wdy = tl.where(mask, w[None, :] * dy, 0.0)
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
    @opaque()
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = torch.empty_like(x_2d)
        m, n = x_2d.shape
        mean = torch.empty(m, dtype=torch.float32, device=x.device)
        rstd = torch.empty(m, dtype=torch.float32, device=x.device)
        grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M1"])]
        layer_norm_fwd_fused[grid](
            x_2d, y_2d, weight, bias, mean, rstd, rstd,
            x_2d.stride(0), x_2d.stride(1),
            m, n, eps,
            shape_key=get_seq_group(m), HAS_ROWSCALE=False,
        )
        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    @opaque()
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
        m, n = x.shape

        g = _persistent_grid(x.device)
        dx_2d = torch.empty_like(dy_2d)
        partial_dw = torch.empty((g, n), dtype=torch.float32, device=x.device)
        partial_db = torch.empty((g, n), dtype=torch.float32, device=x.device)

        # grid axis 1 = column tiles (see the kernel note); the partial buffer stays [g, N].
        grid_bwd = lambda meta: (g, triton.cdiv(n, meta["BLOCK_K"]))  # noqa: E731
        _ln_bwd_persistent[grid_bwd](
            dx_2d, partial_dw, partial_db, dy_2d, x, weight, mean, rstd,
            partial_dw.stride(0), x.stride(0), x.stride(1),
            m, N=n, shape_key=get_seq_group(m),
        )

        dw = partial_dw.sum(dim=0).to(weight.dtype)
        db = partial_db.sum(dim=0).to(weight.dtype)
        return dx_2d.view(ctx.input_shape), dw, db, None


triton_layernorm_persistent = TritonLayerNormPersistentFunction.apply
