"""Recompute helpers for the LayerNormLinear backward: x_normed and x_hat from saved stats."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for

# Both recompute kernels are level=both in kernels/registry.csv, so the key is `both_key(L)` --
# L is the token/atom count, NOT the row count M these launchers receive (x reaches them already
# flattened, and M = L*L for a pair view). Both launchers take it from their caller.
from miniworld_engine.autotune.shape_key import both_key


@triton.autotune(configs=configs_for("layernorm_fwd_recompute_triton"), key=['shape_key', 'K'])
@triton.jit
def _xnormed_kernel(x_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr, y_ptr, M, K, sx0, sx1, sy0, sy1,
                    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, shape_key):
    # Pure elementwise (mean/rstd are read, never reduced here), so the K axis loops in
    # BLOCK_K tiles instead of the old whole-row BLOCK_K=next_power_of_2(K): the tile is now
    # a tuned config value on both axes, and a wide K no longer forces one giant register tile.
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rm = rows < M
    mean = tl.load(mean_ptr + rows, mask=rm, other=0.0)[:, None]
    rstd = tl.load(rstd_ptr + rows, mask=rm, other=0.0)[:, None]
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        cm = cols < K
        x = tl.load(x_ptr + rows[:, None] * sx0 + cols[None, :] * sx1,
                    mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
        b = tl.load(b_ptr + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
        y = (x - mean) * rstd * g + b
        tl.store(y_ptr + rows[:, None] * sy0 + cols[None, :] * sy1,
                 y.to(y_ptr.dtype.element_ty), mask=rm[:, None] & cm[None, :])

def _recompute_xnormed(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       mean: torch.Tensor, rstd: torch.Tensor, *,
                       shape_key: int | None = None):
    """x_normed = (x-mean)*rstd*γ + β, one fused bf16 pass using the SAVED mean/rstd (no stats
    recompute). Reads x at its own strides (strided/transposed view OK — no pre-copy) and writes
    a CONTIGUOUS (M,K) output.

    ``shape_key`` is ``both_key(L)`` from the caller (the saved x is (M, K); M alone cannot say
    which L produced it). None -> smallest bucket (bench/driver entry only)."""
    M, K = x.shape
    y = torch.empty(M, K, device=x.device, dtype=x.dtype)   # contiguous out
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    _xnormed_kernel[grid](
        x, gamma, beta, mean, rstd, y, M, K, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        shape_key=both_key(0) if shape_key is None else shape_key,
    )
    return y

@triton.autotune(configs=configs_for("layernorm_fwd_recompute_noaffine_triton"), key=['shape_key', 'K'])
@triton.jit
def _xhat_kernel(x_ptr, mean_ptr, rstd_ptr, y_ptr, M, K, sx0, sx1, sy0, sy1,
                 BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, shape_key):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rm = rows < M
    mean = tl.load(mean_ptr + rows, mask=rm, other=0.0)[:, None]
    rstd = tl.load(rstd_ptr + rows, mask=rm, other=0.0)[:, None]
    for k0 in range(0, K, BLOCK_K):     # K tiled, was a whole-row next_power_of_2(K) constant
        cols = k0 + tl.arange(0, BLOCK_K)
        cm = cols < K
        x = tl.load(x_ptr + rows[:, None] * sx0 + cols[None, :] * sx1,
                    mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        tl.store(y_ptr + rows[:, None] * sy0 + cols[None, :] * sy1,
                 y.to(y_ptr.dtype.element_ty), mask=rm[:, None] & cm[None, :])

def _recompute_xhat(x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, *,
                    shape_key: int | None = None):
    """x̂ = (x-mean)·rstd (no affine), one fused bf16 pass using the SAVED mean/rstd.
    Reads x at its own strides (so a transposed/strided view is fine — NO pre-copy) and
    writes a CONTIGUOUS (M,K) x̂. This lets the caller feed a strided x (e.g. a bmm
    output viewed channel-major) without a .contiguous() transpose copy.

    ``shape_key`` is ``both_key(L)`` from the caller (see ``_recompute_xnormed``)."""
    M, K = x.shape
    y = torch.empty(M, K, device=x.device, dtype=x.dtype)   # contiguous out
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    _xhat_kernel[grid](x, mean, rstd, y, M, K, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                       shape_key=both_key(0) if shape_key is None else shape_key)
    return y
