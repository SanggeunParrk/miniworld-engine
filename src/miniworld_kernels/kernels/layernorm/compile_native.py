"""custom_op wrappers for standalone LayerNorm methods."""

from __future__ import annotations

import os

import torch
import triton
from torch import Tensor

from .triton.main import (
    get_seq_group,
    layer_norm_bwd_dx_fused,
    layer_norm_fwd_fused,
)
from .triton.partial import (
    _bwd_block_m,
    _bwd_num_warps,
    _layer_norm_bwd_dx_partials,
)
from .triton.persistent import _ln_bwd_persistent, _persistent_grid
from . import dispatch_cache


def _use_partial_reduction(m: int, n: int) -> bool:
    if n < 256:
        return False
    if n >= 768:
        return True
    return m >= 262144


# --- Backward path selection --------------------------------------------------
# The three backward impls are all correct + autotuned on ANY CUDA arch (plain
# triton; the persistent kernel reads the live SM count and uses no Hopper-only
# features). Only the *crossover thresholds* in `_static_bwd_path` were MEASURED on
# H100 (sm_90). On other GPUs we don't guess: `_resolve_bwd_path` times the three
# paths once per (d, M-bucket) on the real tensors and caches the winner per GPU
# (see dispatch_cache). Escape hatch: env `MINIWORLD_LN_BWD=persistent|partial|
# atomic` forces one path (debug / manual override), bypassing cache + heuristic.
_LN_BWD_OVERRIDE = (os.environ.get("MINIWORLD_LN_BWD") or "").strip().lower() or None
_VALID_BWD_PATHS = {"persistent", "partial", "atomic"}
_HOPPER = (9, 0)


def _static_bwd_path(m: int, n: int) -> str:
    """H100-measured heuristic; also the fallback when calibration is off/unavailable."""
    if n >= 384:
        return "persistent"
    if _use_partial_reduction(m, n):
        return "partial"
    return "atomic"


def _time_bwd_path(impl, dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> float:
    try:
        return triton.testing.do_bench(lambda: impl(dy, x, weight, mean, rstd), warmup=10, rep=30)
    except Exception:  # noqa: BLE001 - a path that won't compile/run on this shape is just skipped
        return float("inf")


def _resolve_bwd_path(
    m: int, n: int, dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor
) -> str:
    if _LN_BWD_OVERRIDE in _VALID_BWD_PATHS:
        return _LN_BWD_OVERRIDE

    mode = dispatch_cache.autotune_mode()
    if mode == "off":
        return _static_bwd_path(m, n)

    device = x.device
    cc = torch.cuda.get_device_capability(device)
    # H100 is already measured -> trust the static heuristic (unless explicitly forced).
    if mode != "force" and cc == _HOPPER:
        return _static_bwd_path(m, n)

    mb = dispatch_cache.mbucket(m)
    cached = dispatch_cache.lookup(device, n, mb)
    if cached in _VALID_BWD_PATHS:
        return cached

    # Don't run a timing sweep while a CUDA graph is capturing.
    if torch.cuda.is_current_stream_capturing():
        return _static_bwd_path(m, n)

    # Calibrate: time the three correct paths on the real tensors, cache the winner.
    impls = {"atomic": _bwd_atomic_impl, "partial": _bwd_partial_impl, "persistent": _bwd_persistent_impl}
    times = {name: _time_bwd_path(fn, dy, x, weight, mean, rstd) for name, fn in impls.items()}
    best = min(times, key=times.get)
    if times[best] == float("inf"):  # nothing ran (shouldn't happen) -> heuristic
        return _static_bwd_path(m, n)
    dispatch_cache.store(device, n, mb, best, times)
    return best


def _fwd_impl(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
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
        rstd,
        x_2d.stride(0),
        x_2d.stride(1),
        m,
        n,
        eps,
        BLOCK_N=triton.next_power_of_2(n),
        GROUP_M=get_seq_group(m),
        HAS_ROWSCALE=False,
    )
    return y_2d.view_as(x), mean, rstd


def _bwd_atomic_impl(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
    m, n = x_2d.shape
    dx_2d = torch.empty_like(dy_2d)
    dw = torch.zeros(n, dtype=torch.float32, device=x.device)
    db = torch.zeros(n, dtype=torch.float32, device=x.device)
    grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M"])]
    layer_norm_bwd_dx_fused[grid](
        dx_2d,
        dy_2d,
        dw,
        db,
        x_2d,
        weight,
        mean,
        rstd,
        dw.stride(0),
        db.stride(0),
        x_2d.stride(0),
        x_2d.stride(1),
        m,
        n,
        BLOCK_N=triton.next_power_of_2(n),
        GROUP_M=get_seq_group(m),
    )
    return dx_2d.view_as(x), dw.to(weight.dtype), db.to(weight.dtype)


def _bwd_partial_impl(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
    m, n = x_2d.shape
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
        x_2d,
        weight,
        mean,
        rstd,
        partial_dw.stride(0),
        x_2d.stride(0),
        x_2d.stride(1),
        m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    dw = partial_dw.sum(dim=0).to(weight.dtype)
    db = partial_db.sum(dim=0).to(weight.dtype)
    return dx_2d.view_as(x), dw, db


def _bwd_persistent_impl(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Persistent grid-stride backward: ~NUM_SM*waves partial rows, vectorized 2D
    tiles (see triton/persistent.py). Wins at d >= 384, matches quack cute at d=768."""
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
    m, n = x_2d.shape
    g = _persistent_grid(x.device)
    block_n = triton.next_power_of_2(n)

    dx_2d = torch.empty_like(dy_2d)
    partial_dw = torch.empty((g, n), dtype=torch.float32, device=x.device)
    partial_db = torch.empty((g, n), dtype=torch.float32, device=x.device)
    _ln_bwd_persistent[(g,)](
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
        BLOCK_N=block_n,
    )
    dw = partial_dw.sum(dim=0).to(weight.dtype)
    db = partial_db.sum(dim=0).to(weight.dtype)
    return dx_2d.view_as(x), dw, db


@torch.library.custom_op("miniworld_layernorm::atomic_bwd", mutates_args=())
def _atomic_bwd(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return _bwd_atomic_impl(dy, x, weight, mean, rstd)


@_atomic_bwd.register_fake
def _(dy, x, weight, mean, rstd):
    return x.new_empty(x.shape), weight.new_empty(weight.shape), weight.new_empty(weight.shape)


@torch.library.custom_op("miniworld_layernorm::partial_bwd", mutates_args=())
def _partial_bwd(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return _bwd_partial_impl(dy, x, weight, mean, rstd)


@_partial_bwd.register_fake
def _(dy, x, weight, mean, rstd):
    return x.new_empty(x.shape), weight.new_empty(weight.shape), weight.new_empty(weight.shape)


@torch.library.custom_op("miniworld_layernorm::dispatch_bwd", mutates_args=())
def _dispatch_bwd(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    m = x.numel() // x.shape[-1]
    n = x.shape[-1]
    path = _resolve_bwd_path(m, n, dy, x, weight, mean, rstd)
    if path == "persistent":
        return _bwd_persistent_impl(dy, x, weight, mean, rstd)
    if path == "partial":
        return _bwd_partial_impl(dy, x, weight, mean, rstd)
    return _bwd_atomic_impl(dy, x, weight, mean, rstd)


@_dispatch_bwd.register_fake
def _(dy, x, weight, mean, rstd):
    return x.new_empty(x.shape), weight.new_empty(weight.shape), weight.new_empty(weight.shape)


def _setup_context(ctx, inputs, output) -> None:
    x, weight, bias, eps = inputs
    y, mean, rstd = output
    del bias, eps, y
    ctx.save_for_backward(x, weight, mean, rstd)


@torch.library.custom_op("miniworld_layernorm::atomic_fwd", mutates_args=())
def _atomic_fwd(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    return _fwd_impl(x, weight, bias, eps)


@_atomic_fwd.register_fake
def _(x, weight, bias, eps):
    m = x.numel() // x.shape[-1]
    return x.new_empty(x.shape), x.new_empty((m,), dtype=torch.float32), x.new_empty((m,), dtype=torch.float32)


def _atomic_backward(ctx, dy, dmean, drstd):
    del dmean, drstd
    x, weight, mean, rstd = ctx.saved_tensors
    dx, dw, db = _atomic_bwd(dy, x, weight, mean, rstd)
    return dx, dw, db, None


_atomic_fwd.register_autograd(_atomic_backward, setup_context=_setup_context)


@torch.library.custom_op("miniworld_layernorm::partial_fwd", mutates_args=())
def _partial_fwd(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    return _fwd_impl(x, weight, bias, eps)


@_partial_fwd.register_fake
def _(x, weight, bias, eps):
    m = x.numel() // x.shape[-1]
    return x.new_empty(x.shape), x.new_empty((m,), dtype=torch.float32), x.new_empty((m,), dtype=torch.float32)


def _partial_backward(ctx, dy, dmean, drstd):
    del dmean, drstd
    x, weight, mean, rstd = ctx.saved_tensors
    dx, dw, db = _partial_bwd(dy, x, weight, mean, rstd)
    return dx, dw, db, None


_partial_fwd.register_autograd(_partial_backward, setup_context=_setup_context)


@torch.library.custom_op("miniworld_layernorm::dispatch_fwd", mutates_args=())
def _dispatch_fwd(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    return _fwd_impl(x, weight, bias, eps)


@_dispatch_fwd.register_fake
def _(x, weight, bias, eps):
    m = x.numel() // x.shape[-1]
    return x.new_empty(x.shape), x.new_empty((m,), dtype=torch.float32), x.new_empty((m,), dtype=torch.float32)


def _dispatch_backward(ctx, dy, dmean, drstd):
    del dmean, drstd
    x, weight, mean, rstd = ctx.saved_tensors
    dx, dw, db = _dispatch_bwd(dy, x, weight, mean, rstd)
    return dx, dw, db, None


_dispatch_fwd.register_autograd(_dispatch_backward, setup_context=_setup_context)


def layernorm_atomic_compile(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    return _atomic_fwd(x, weight, bias, eps)[0]


def layernorm_partial_compile(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    return _partial_fwd(x, weight, bias, eps)[0]


def layernorm_dispatch_compile(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    return _dispatch_fwd(x, weight, bias, eps)[0]

