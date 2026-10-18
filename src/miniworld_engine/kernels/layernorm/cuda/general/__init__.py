"""General LayerNorm on CUDA: any width (1 .. 65 536 bf16 / fp16, 32 768 fp32), any row count, bf16 / fp16 / fp32 activations, fp32
statistics, optional weight and bias (cast to fp32), forward and backward (dx and the affine gradients without atomics: a replay is
bit-identical). Plain CUDA (sm_80+), built on first use; every launch is an opaque op with a fake, so it traces under torch.compile
and captures in a CUDA graph.

Layout (``ln_common.cuh``): TPR threads per row, each holding CHUNKS vectors of VEC elements in registers; :func:`config` picks them
from the width: the fewest idle lanes, at most 32 values per thread, and TPR as close to 128 as that allows (measured on H100: 128
threads x 2-4 vectors beat 256-1024 threads x 1 vector by 10-40% at widths 2048-8192; 8 vectors per thread spill).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

_dir = Path(__file__).parent

#: threads per row the builds have (4 / 8 / 16 only on the vector path)
TPRS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
CHUNKS = (1, 2, 4, 8)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def max_width(dtype: torch.dtype) -> int:
    return 1024 * 8 * (16 // torch.tensor([], dtype=dtype).element_size())


@functools.lru_cache(maxsize=4096)
def config(n: int, elem_bytes: int) -> tuple[int, int, int]:
    """(tpr, chunks, vec) for a row of ``n`` elements of ``elem_bytes`` bytes."""
    vmax = 16 // elem_bytes
    vec = vmax if n % vmax == 0 else 1
    nvec = math.ceil(n / vec)
    best = None
    for tpr in TPRS:
        if vec == 1 and tpr < 32:
            continue
        for ch in CHUNKS:
            if tpr * ch < nvec:
                continue
            waste = tpr * ch - nvec
            fits = ch * vec <= 32 or ch == 1
            key = (not fits, waste / (tpr * ch), abs(tpr.bit_length() - 8))
            if best is None or key < best[0]:
                best = (key, (tpr, ch, vec))
            break
    if best is None:
        raise ValueError(f"LayerNorm width {n} exceeds the general CUDA kernel's builds")
    return best[1]


@functools.lru_cache(maxsize=None)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="layernorm_general_cuda_v2",
        sources=[str(_dir / "bind.cpp"), str(_dir / "ln_fwd.cu"), str(_dir / "ln_bwd.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "--expt-relaxed-constexpr", *gencodes("80", "90", "100")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def supported(x: torch.Tensor) -> bool:
    """Whether the kernels take this activation (CUDA, a listed dtype, a width within the builds)."""
    return (
        x.is_cuda
        and x.dtype in DTYPES
        and x.dim() >= 1
        and 1 <= x.shape[-1] <= max_width(x.dtype)
        and (x.dtype != torch.float32 or x.shape[-1] <= 1024 * 8 * 4)
    )


def _affine(t: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    if t is None:
        return x.new_empty((0,), dtype=torch.float32)
    return t.detach().to(device=x.device, dtype=torch.float32).contiguous()


def _fwd_fake(x, w, b, eps, save):
    m = x.shape[0] if save else 0
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty_like(x), torch.empty((m,), **f32), torch.empty((m,), **f32)


@opaque(fake=_fwd_fake, name="layernorm_general_fwd")
def _fwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tpr, ch, vec = config(x.shape[1], x.element_size())
    y, mean, rstd = _ext().fwd(x, w, b, float(eps), bool(save), tpr, ch, vec)
    return y, mean, rstd


def _bwd_fake(dy, x, w, mean, rstd, wgrad):
    f32 = dict(dtype=torch.float32, device=x.device)
    n = x.shape[1] if wgrad else 0
    return torch.empty_like(x), torch.empty((n,), **f32), torch.empty((n,), **f32)


@opaque(fake=_bwd_fake, name="layernorm_general_bwd")
def _bwd(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor,
         wgrad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tpr, ch, vec = config(x.shape[1], x.element_size())
    dx, dw, db = _ext().bwd(dy, x, w, mean, rstd, bool(wgrad), tpr, ch, vec, 0)
    return dx, dw, db


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, weight, bias, eps):
        w, b = _affine(weight, x2), _affine(bias, x2)
        y, mean, rstd = _fwd(x2, w, b, eps, True)
        ctx.save_for_backward(x2, w, mean, rstd)
        ctx.has = (weight is not None, bias is not None)
        ctx.dtypes = (None if weight is None else weight.dtype, None if bias is None else bias.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, w, mean, rstd = ctx.saved_tensors
        has_w, has_b = ctx.has
        wgrad = (has_w and ctx.needs_input_grad[1]) or (has_b and ctx.needs_input_grad[2])
        dx, dw, db = _bwd(dy.contiguous(), x2, w, mean, rstd, bool(wgrad))
        gw = dw.to(ctx.dtypes[0]) if has_w and ctx.needs_input_grad[1] else None
        gb = db.to(ctx.dtypes[1]) if has_b and ctx.needs_input_grad[2] else None
        return dx, gw, gb, None


def layernorm_cuda(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension; ``y`` in ``x``'s dtype, fp32 statistics; weight / bias optional, any float dtype (fp32
    gradients handed back in the parameters' dtype)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if not needs_grad:
        y, _, _ = _fwd(x2, _affine(weight, x2), _affine(bias, x2), float(eps), False)
        return y.reshape(shape)
    return _LayerNorm.apply(x2, weight, bias, float(eps)).reshape(shape)


__all__ = ["config", "layernorm_cuda", "supported"]
