"""custom_op wrappers for standalone LayerNorm methods."""

from __future__ import annotations

from miniworld_engine.kernels._compile import opaque


import torch
import triton
from torch import Tensor

from miniworld_engine.autotune.shape_key import both_key, length_of

from .triton.main import (
    layer_norm_bwd_dx_fused,
    layer_norm_fwd_fused,
)
from .triton.partial import _bwd_block_m
from .triton.persistent import _ln_bwd_persistent, _persistent_grid
from . import dispatch as dispatch_cache
from miniworld_engine import settings


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
# (see dispatch_cache). Escape hatch: env `settings.layernorm_bwd_path
# atomic` forces one path (debug / manual override), bypassing cache + heuristic.
def _ln_bwd_override() -> str | None:
    """Read at call time: a module-level constant would freeze whatever was set at import."""
    return settings.current().layernorm_bwd_path
_VALID_BWD_PATHS = {"persistent", "partial", "atomic", "cuda"}
_HOPPER = (9, 0)


def _static_bwd_path(m: int, n: int, is_bf16: bool = False) -> str:
    """H100-measured heuristic; also the fallback when calibration is off/unavailable."""
    # Hand-CUDA warp-per-row bwd beats triton 1.2-1.46x for bf16 128<=N<=512 (measured H100);
    # N>=768 / fp32 stay on triton (persistent is already near HBM roofline there).
    if is_bf16 and 128 <= n <= 512:
        return "cuda"
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
    override = _ln_bwd_override()
    if override is not None and override in _VALID_BWD_PATHS:
        return override

    # The hand-CUDA fast bwd path requires x AND weight to share one dtype (its kernel rejects a
    # mixed pair); the triton paths promote internally and tolerate a mixed pair. Some pairformer
    # LNs run bf16 activations against an fp32 weight, so gate the cuda path on BOTH being bf16 —
    # otherwise fall to a tolerant triton path (same path the mixed case already used elsewhere).
    is_bf16 = x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    mode = dispatch_cache.autotune_mode()
    if mode == "off":
        return _static_bwd_path(m, n, is_bf16)

    device = x.device
    cc = torch.cuda.get_device_capability(device)
    # H100 is already measured -> trust the static heuristic (unless explicitly forced).
    if mode != "force" and cc == _HOPPER:
        return _static_bwd_path(m, n, is_bf16)

    mb = dispatch_cache.mbucket(m)
    cached = dispatch_cache.lookup(device, n, mb)
    if cached in _VALID_BWD_PATHS:
        return cached

    # Don't run a timing sweep while a CUDA graph is capturing.
    if torch.cuda.is_current_stream_capturing():
        return _static_bwd_path(m, n, is_bf16)

    # Calibrate: time the three correct paths on the real tensors, cache the winner.
    impls = {"atomic": _bwd_atomic_impl, "partial": _bwd_partial_impl, "persistent": _bwd_persistent_impl}
    if is_bf16 and 128 <= n <= 512:
        impls["cuda"] = _bwd_cuda_impl
    times = {name: _time_bwd_path(fn, dy, x, weight, mean, rstd) for name, fn in impls.items()}
    best = min(times, key=lambda name: times[name])  # not `times.get`: it is `float | None`
    if times[best] == float("inf"):  # nothing ran (shouldn't happen) -> heuristic
        return _static_bwd_path(m, n, is_bf16)
    dispatch_cache.store(device, n, mb, best, times)
    return best


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
        shape_key=both_key(length_of(x.shape)),
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
        shape_key=both_key(length_of(x.shape)),
        HAS_ROWSCALE=False,
    )
    return dx_2d.view_as(x), dw.to(weight.dtype), db.to(weight.dtype)


def _bwd_partial_impl(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
    m, n = x_2d.shape
    # _bwd_block_m now sizes the PARTIAL BUFFER only; BLOCK_M1/BLOCK_N/num_warps are tuned
    # (see triton/partial.py). Grid axis 1 is the feature tile.
    num_partials = triton.cdiv(m, _bwd_block_m(n))

    dx_2d = torch.empty_like(dy_2d)
    partial_dw = torch.empty((num_partials, n), dtype=torch.float32, device=x.device)
    partial_db = torch.empty((num_partials, n), dtype=torch.float32, device=x.device)
    grid = lambda meta: (num_partials, triton.cdiv(n, meta["BLOCK_K"]))  # noqa: E731
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
        shape_key=both_key(length_of(x.shape)),
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

    dx_2d = torch.empty_like(dy_2d)
    partial_dw = torch.empty((g, n), dtype=torch.float32, device=x.device)
    partial_db = torch.empty((g, n), dtype=torch.float32, device=x.device)
    # grid axis 1 = feature tiles; BLOCK_N is tuned now (see triton/persistent.py).
    grid = lambda meta: (g, triton.cdiv(n, meta["BLOCK_K"]))  # noqa: E731
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
        shape_key=both_key(length_of(x.shape)),
    )
    dw = partial_dw.sum(dim=0).to(weight.dtype)
    db = partial_db.sum(dim=0).to(weight.dtype)
    return dx_2d.view_as(x), dw, db


def _bwd_cuda_impl(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Hand-CUDA warp-per-row backward (register column-partials, no atomics/no spill). Beats triton
    1.2-1.46x for bf16 128<=N<=512 on H100. Lazy import so `compile_native` never triggers the nvcc
    build unless this path is actually selected."""
    from .cuda import layer_norm_bwd_cuda

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
    dx_2d, dw, db = layer_norm_bwd_cuda(dy_2d, x_2d, weight, mean, rstd)
    return dx_2d.view_as(x), dw.to(weight.dtype), db.to(weight.dtype)










@opaque(fake=lambda dy, x, weight, mean, rstd: (
    x.new_empty(x.shape), weight.new_empty(weight.shape), weight.new_empty(weight.shape)),
    name="layernorm_dispatch_bwd")
def _dispatch_bwd(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor,
                  rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Backward through whichever reduction path this shape and card resolve to."""
    m = x.numel() // x.shape[-1]
    n = x.shape[-1]
    path = _resolve_bwd_path(m, n, dy, x, weight, mean, rstd)
    if path == "cuda":
        return _bwd_cuda_impl(dy, x, weight, mean, rstd)
    if path == "persistent":
        return _bwd_persistent_impl(dy, x, weight, mean, rstd)
    if path == "partial":
        return _bwd_partial_impl(dy, x, weight, mean, rstd)
    return _bwd_atomic_impl(dy, x, weight, mean, rstd)


@opaque(fake=lambda x, weight, bias, eps: (
    x.new_empty(x.shape),
    x.new_empty((x.numel() // x.shape[-1],), dtype=torch.float32),
    x.new_empty((x.numel() // x.shape[-1],), dtype=torch.float32)),
    name="layernorm_dispatch_fwd")
def _dispatch_fwd(x: Tensor, weight: Tensor, bias: Tensor,
                  eps: float) -> tuple[Tensor, Tensor, Tensor]:
    return _fwd_impl(x, weight, bias, eps)


class _LayerNormDispatch(torch.autograd.Function):
    """Carries the backward-path choice through autograd.

    An autograd.Function rather than ``custom_op.register_autograd`` so the same code works under
    both ``compile_wrap`` modes: the launchers above are wrapped by ``opaque`` and this only has to
    save the tensors the chosen backward path needs.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, mean, rstd = _dispatch_fwd(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = _dispatch_bwd(dy, x, weight, mean, rstd)
        return dx, dw, db, None


def layernorm_dispatch_compile(x: Tensor, weight: Tensor, bias: Tensor,
                               eps: float = 1e-5) -> Tensor:
    """LayerNorm whose backward reduction path is resolved per shape and card."""
    return _LayerNormDispatch.apply(x, weight, bias, eps)
