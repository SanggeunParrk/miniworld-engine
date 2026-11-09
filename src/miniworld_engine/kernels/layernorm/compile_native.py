"""The Triton LayerNorm forward and the atomic / persistent backward launchers (the edge LayerNorm of ProteinMPNN and the fp32 LayerNormLinear forward use them)."""

from __future__ import annotations

import torch
import triton
from torch import Tensor

from miniworld_engine.kernels._shape_key import both_key, pack, rows_of
from miniworld_engine.kernels.layernorm.triton.main import (
    layer_norm_bwd_dx_fused,
    layer_norm_fwd_fused,
)
from miniworld_engine.kernels.layernorm.triton.persistent import (
    _ln_bwd_persistent,
    _persistent_grid,
)


def _fwd_impl(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    y_2d = torch.empty_like(x_2d)
    m, n = x_2d.shape
    mean = torch.empty(m, dtype=torch.float32, device=x.device)
    rstd = torch.empty(m, dtype=torch.float32, device=x.device)
    grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M1"])]
    layer_norm_fwd_fused[grid](
        x_2d,
        y_2d,
        weight,
        bias,
        mean,
        rstd,
        rstd,
        x_2d.stride(0),
        x_2d.stride(1),
        m,
        n,
        eps,
        # L = x.shape[-2], read before the reshape above (pair (B,L,L,D) / token (B,L,D)),
        # not the row count m. The kernel parameter is `shape_key`; `GROUP_M` here was a
        # stale name from before the rename.
        shape_key=both_key(rows_of(x.shape), N=n),
        HAS_ROWSCALE=False,
    )
    return y_2d.view_as(x), mean, rstd


def _bwd_atomic_impl(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor,
                     row_bucket: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Atomic-reduction LayerNorm backward.

    `row_bucket` is the ROW BUCKET, not a finished key, and it is for a caller whose activation is
    genuinely flat -- mpnn's edge LayerNorm runs on (edge rows, 128) and has no (B, L, D) to hand
    over, so `rows_of` refuses its shape by design. `N` is folded on BOTH paths, which is the point
    of taking a bucket rather than a key: a launcher that folds an axis on one path and not the
    other writes two different keys for one shape, and that is what the key-gap audit catches.
    """
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
    m, n = x_2d.shape
    dx_2d = torch.empty_like(dy_2d)
    dw = torch.zeros(n, dtype=torch.float32, device=x.device)
    db = torch.zeros(n, dtype=torch.float32, device=x.device)
    grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M1"])]
    layer_norm_bwd_dx_fused[grid](
        dx_2d,
        dy_2d,
        dw,
        db,
        x_2d,
        weight,
        mean,
        rstd,
        rstd,
        dw.stride(0),
        db.stride(0),
        x_2d.stride(0),
        x_2d.stride(1),
        m,
        n,
        shape_key=both_key(rows_of(x.shape), N=n) if row_bucket is None else pack(row_bucket, N=n),
        HAS_ROWSCALE=False,
    )
    return dx_2d.view_as(x), dw.to(weight.dtype), db.to(weight.dtype)



def _bwd_persistent_impl(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor,
                         row_bucket: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Persistent grid-stride backward: ~NUM_SM*waves partial rows, vectorized 2D
    tiles (see triton/persistent.py). Wins at d >= 384, matches quack cute at d=768.

    `row_bucket` is the ROW BUCKET, not a finished key, and it is for a caller whose activation is
    genuinely flat -- mpnn's edge LayerNorm runs on (edge rows, 128) and has no (B, L, D) to hand
    over, so `rows_of` refuses its shape by design. `N` is folded on BOTH paths, which is the point
    of taking a bucket rather than a key: a launcher that folds an axis on one path and not the
    other writes two different keys for one shape, and that is what the key-gap audit catches.
    """
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
    m, n = x_2d.shape
    g = _persistent_grid(x.device)

    dx_2d = torch.empty_like(dy_2d)
    partial_dw = torch.empty((g, n), dtype=torch.float32, device=x.device)
    partial_db = torch.empty((g, n), dtype=torch.float32, device=x.device)
    # grid axis 1 = feature tiles; BLOCK_N is tuned now (see triton/persistent.py).
    grid = lambda meta: (g, triton.cdiv(n, meta["BLOCK_K"]))
    _ln_bwd_persistent[grid](
        dx_2d,
        partial_dw,
        partial_db,
        dy_2d,
        x_2d,
        weight,
        mean,
        rstd,
        partial_dw.stride(0),
        x_2d.stride(0),
        x_2d.stride(1),
        m,
        N=n,
        shape_key=both_key(rows_of(x.shape), N=n) if row_bucket is None else pack(row_bucket, N=n),
    )
    dw = partial_dw.sum(dim=0).to(weight.dtype)
    db = partial_db.sum(dim=0).to(weight.dtype)
    return dx_2d.view_as(x), dw, db


