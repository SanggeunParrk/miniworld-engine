"""Experimental native CUDA norms. Explicit opt-in until qualification is complete.

Low precision accumulates in FP32, FP64 in FP64. Affine parameter casts preserve
leaf gradient dtypes. Arbitrary leading dimensions/strides are normalized through
contiguous inputs; this copy is included in end-to-end timings.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable


@lru_cache(None)
def extension():
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    source = Path(__file__).with_name("norm.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load_extension(
        name=f"mw_norm_cuda_{digest}",
        sources=[str(source)],
        extra_cuda_cflags=[*host_flags(), "-O3", "-lineinfo"],
        verbose=False,
    )


def _extension(backend):
    if backend == "cta":
        from .wide import extension as wide_extension

        return wide_extension()
    return extension()


def resolve_config(x, rms, backend="auto", threads=None, rows=None):
    """Conservative native defaults; callers can force either schedule and config.

    CTA rows distribute large widths across threads. The row grouping targets
    roughly 512 CTAs, while the warp path preserves the existing bounded buffer.
    Exact measured H100 overrides are separate from these portable defaults.
    """
    if backend not in ("auto", "warp", "cta"):
        raise ValueError("backend must be auto, warp, or cta")
    d = x.shape[-1]
    m = x.numel() // d
    if backend == "auto":
        backend = "cta" if 1024 <= d <= 16384 else "warp"
    if backend == "cta" and d > 16384:
        raise ValueError("CTA normalization supports widths up to 16384")
    if threads is None:
        if backend == "cta":
            threads = 256 if d > 1024 else 128
        elif d in (256, 384, 512):
            threads = 128 if x.dtype == torch.float64 and d == 512 else 256
        elif d <= 128 and (
            m <= 32768
            or x.dtype in (torch.float32, torch.float64)
            or (rms and d == 128)
        ):
            threads = 256
        else:
            threads = 128
    if rows is None:
        if backend == "cta":
            rows = min(4096, max(1, m // 512))
        elif d in (64, 128):
            rows = (
                1
                if m <= 32768
                else (
                    4
                    if x.dtype in (torch.float32, torch.float64)
                    else (24 if rms and d == 128 else 16)
                )
            )
        elif d in (256, 384, 512):
            rows = (
                (64 if rms and x.dtype in (torch.float32, torch.float64) else 16)
                if m > 32768
                else 4
            )
        else:
            rows = 64 if m > 32768 else 4
    return backend, threads, rows


@torch.library.custom_op("miniworld_norm_cuda::forward", mutates_args=())
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    rms: bool,
    threads: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(_extension(backend).forward(x, weight, bias, eps, rms, threads))


@_forward.register_fake
def _forward_fake(x, weight, bias, eps, rms, threads, backend):
    opts = dict(
        device=x.device,
        dtype=torch.float64 if x.dtype == torch.float64 else torch.float32,
    )
    return (
        torch.empty_like(x),
        torch.empty((x.numel() // x.shape[-1],), **opts),
        torch.empty((x.numel() // x.shape[-1],), **opts),
    )


@torch.library.custom_op("miniworld_norm_cuda::backward", mutates_args=())
def _backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor | None,
    mean: torch.Tensor,
    inv: torch.Tensor,
    has_bias: bool,
    rms: bool,
    threads: int,
    rows: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        _extension(backend).backward(
            x, dy, weight, mean, inv, has_bias, rms, threads, rows
        )
    )


@_backward.register_fake
def _backward_fake(x, dy, weight, mean, inv, has_bias, rms, threads, rows, backend):
    return (
        torch.empty_like(x),
        mean.new_empty((x.shape[-1] if weight is not None else 0,)),
        mean.new_empty((x.shape[-1] if has_bias else 0,)),
    )


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, rms, threads, rows, backend):
        y, mean, inv = _forward(x, weight, bias, float(eps), rms, threads, backend)
        ctx.save_for_backward(x, weight, mean, inv)
        ctx.args = (bias is not None, rms, threads, rows, backend)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, mean, inv = ctx.saved_tensors
        dx, dw, db = _backward(x, dy.contiguous(), weight, mean, inv, *ctx.args)
        return (
            dx,
            dw if weight is not None else None,
            db if ctx.args[0] else None,
            None,
            None,
            None,
            None,
            None,
        )


def _norm(x, weight, bias, eps, rms, threads, rows, backend):
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("normalization needs a nonempty last dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("normalization requires floating-point input")
    for p in (weight, bias):
        if p is not None and (
            p.shape != (x.shape[-1],)
            or p.device != x.device
            or not p.is_floating_point()
        ):
            raise ValueError("affine parameters must match last dimension and device")
    backend, threads, rows = resolve_config(x, rms, backend, threads, rows)
    if threads not in (128, 256) or not 1 <= rows <= 4096:
        raise ValueError("invalid CUDA norm configuration")
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    weight = None if weight is None else weight.to(acc).contiguous()
    bias = None if bias is None else bias.to(acc).contiguous()
    if not x.is_cuda:
        xf = x.to(acc)
        if rms:
            y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
            if weight is not None:
                y = y * weight
        else:
            y = torch.nn.functional.layer_norm(xf, (x.shape[-1],), weight, bias, eps)
        return y.to(x.dtype)
    return _Norm.apply(x.contiguous(), weight, bias, eps, rms, threads, rows, backend)


def cuda_layernorm(
    x, weight=None, bias=None, eps=1e-5, *, threads=None, rows=None, backend="auto"
):
    return _norm(x, weight, bias, eps, False, threads, rows, backend)


def cuda_rmsnorm(x, weight=None, eps=1e-5, *, threads=None, rows=None, backend="auto"):
    # eps=None follows torch's input-dtype epsilon, before the accumulation cast.
    eps = torch.finfo(x.dtype).eps if eps is None else eps
    return _norm(x, weight, None, eps, True, threads, rows, backend)


def cuda_layernorm_linear(
    x,
    ln_weight,
    ln_bias,
    weight,
    bias=None,
    eps=1e-5,
    *,
    threads=None,
    rows=None,
    backend="auto",
    linear_backend="auto",
):
    """Native norm/linear; selected narrow projections use LN/WMMA fusion.

    The normalized activation rounds to x.dtype before the GEMM, matching the
    existing PyTorch contract (no weight folding / changed intermediate rounding).
    """
    if backend not in ("auto", "warp", "cta"):
        raise ValueError("backend must be auto, warp, or cta")
    if linear_backend not in ("auto", "composed", "fused"):
        raise ValueError("linear_backend must be auto, composed, or fused")
    narrow = (
        x.is_cuda
        and x.ndim >= 1
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.shape[-1] == 128
        and x.numel() // 128 == 147456
        and tuple(weight.shape) == (16, 128)
        and weight.dtype == x.dtype
        and ln_weight is not None
        and ln_bias is not None
        and (bias is None or bias.dtype == x.dtype)
        and backend in ("auto", "warp")
    )
    if linear_backend == "fused" or (linear_backend == "auto" and narrow):
        from .linear import fused_layernorm_linear

        return fused_layernorm_linear(
            x, ln_weight, ln_bias, weight, bias, eps, threads=threads, rows=rows
        )
    xn = cuda_layernorm(
        x, ln_weight, ln_bias, eps, threads=threads, rows=rows, backend=backend
    )
    return torch.nn.functional.linear(xn, weight, bias)
