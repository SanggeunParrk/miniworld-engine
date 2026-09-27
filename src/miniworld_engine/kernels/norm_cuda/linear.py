"""Experimental native LN/WMMA fusion, with exact saved-activation backward."""

import hashlib
from functools import lru_cache
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
    source = Path(__file__).with_name("linear.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load_extension(
        name=f"mw_lnlinear_{digest}",
        sources=[str(source)],
        extra_cuda_cflags=[*host_flags(), "-O3", "-lineinfo"],
    )


@torch.library.custom_op("miniworld_norm_cuda::linear_forward", mutates_args=())
def _forward(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    eps: float,
    save: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(extension().forward(x, gamma, beta, w, b, eps, save))


@_forward.register_fake
def _fake(x, gamma, beta, w, b, eps, save):
    m, k = x.shape
    return (
        x.new_empty((m, w.shape[0])),
        x.new_empty((m if save else 0, k)),
        gamma.new_empty((m,)),
        gamma.new_empty((m,)),
    )


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, w, b, eps, threads, rows):
        y, xn, mean, inv = _forward(x, gamma, beta, w, b, float(eps), True)
        ctx.save_for_backward(x, gamma, w, xn, mean, inv)
        ctx.args = threads, rows, b is not None
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        from . import _backward

        x, gamma, w, xn, mean, inv = ctx.saved_tensors
        threads, rows, has_bias = ctx.args
        dy = dy.contiguous()
        dxn = dy @ w
        dw = dy.t() @ xn
        dx, dg, dbeta = _backward(
            x, dxn, gamma, mean, inv, True, False, threads, rows, "warp"
        )
        db = dy.sum(0) if has_bias else None
        return dx, dg, dbeta, dw, db, None, None, None


def fused_layernorm_linear(
    x, gamma, beta, weight, bias=None, eps=1e-5, *, threads=None, rows=None
):
    """Explicit experimental path; fallback retains unrestricted dtype/shape support."""
    from . import cuda_layernorm_linear, resolve_config

    supported = (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim >= 1
        and x.shape[-1] in (64, 128, 256, 384, 512)
        and weight.ndim == 2
        and weight.shape[1] == x.shape[-1]
        and weight.shape[0] > 0
        and weight.shape[0] % 16 == 0
        and gamma is not None
        and beta is not None
        and weight.dtype == x.dtype
        and (bias is None or bias.dtype == x.dtype)
        and torch.cuda.get_device_capability(x.device)[0] >= 9
    )
    if not supported:
        return cuda_layernorm_linear(
            x,
            gamma,
            beta,
            weight,
            bias,
            eps,
            threads=threads,
            rows=rows,
            linear_backend="composed",
        )
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("linear weight dtype/device must match input")
    _, threads, rows = resolve_config(x, False, "warp", threads, rows)
    if threads not in (128, 256) or not 1 <= rows <= 4096:
        raise ValueError("invalid CUDA norm configuration")
    shape = x.shape[:-1]
    xc = x.reshape(-1, x.shape[-1]).contiguous()
    wc = weight.contiguous()
    gamma, beta = gamma.float().contiguous(), beta.float().contiguous()
    b = None if bias is None else bias.contiguous()
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, gamma, beta, weight, b) if t is not None
    ):
        y = _Linear.apply(xc, gamma, beta, wc, b, eps, threads, rows)
    else:
        y = _forward(xc, gamma, beta, wc, b, float(eps), False)[0]
    return y.reshape(*shape, weight.shape[0])
