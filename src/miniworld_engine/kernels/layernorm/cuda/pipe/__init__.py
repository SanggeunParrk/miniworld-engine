"""Pipelined LayerNorm / RMSNorm on CUDA (warp-per-row-group, cp.async ring per warp; see ``ln_pipe.cu``).

Rows of 16-byte-aligned width N (``N * itemsize`` a multiple of 16, at most 4096 bytes), bf16 / fp16 / fp32 activations, fp32 statistics,
optional weight / bias of one dtype (bf16, fp16 or fp32; read in place, gradients handed back in that dtype). The forward stores only ``y``
(the backward recomputes the statistics from the x it re-reads); the backward is one kernel: dx and the affine gradients (a last-CTA reduction
over atomically added per-CTA partials, converted to the weight dtype in place). The atomics add in an unspecified order: the affine
gradients agree to fp32 rounding between replays, not bit for bit (the general kernels are bit-reproducible). Plain CUDA sm_80+.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

_dir = Path(__file__).parent
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_WCODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
#: widest and narrowest rows (bytes) the builds have: 16-chunk half warps up to 32 lanes x 4 chunks of 16 bytes
MIN_ROW_BYTES = 144
MAX_ROW_BYTES = 2048
#: resident CTAs per SM to launch (0: as many as fit)
PER_SM = 0


@functools.lru_cache(maxsize=None)
def _ext():  # noqa: D103
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="layernorm_pipe_cuda_v1",
        sources=[str(_dir / "ln_pipe.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "--expt-relaxed-constexpr", *gencodes("80", "90", "100")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def supported(x: torch.Tensor) -> bool:
    """Whether the pipelined kernels take this activation: CUDA, a listed dtype, 16-byte rows of at most ``MAX_ROW_BYTES``."""
    if not x.is_cuda or x.dtype not in _DTYPES or x.dim() < 1 or x.numel() == 0:
        return False
    row = x.shape[-1] * x.element_size()
    return row % 16 == 0 and MIN_ROW_BYTES <= row <= MAX_ROW_BYTES


def _affine_args(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None):
    """(w, b, wcode): both in one dtype the kernel reads (bf16 / fp16 / fp32), 16-byte aligned, empty tensors when absent."""
    ref = weight if weight is not None else bias
    dt = ref.dtype if ref is not None and ref.dtype in _WCODE else torch.float32
    if (weight is not None and weight.dtype != dt) or (bias is not None and bias.dtype != dt):
        dt = torch.float32
    empty = x.new_empty((0,), dtype=dt)

    def prep(t):
        t = t.detach().to(dt).contiguous()
        return t if t.data_ptr() % 16 == 0 else t.clone()

    return (empty if weight is None else prep(weight)), (empty if bias is None else prep(bias)), _WCODE[dt]


def _fwd_fake(x, w, b, eps, rms, train, with_ws):
    n, m = x.shape[1], x.shape[0]
    f32 = dict(dtype=torch.float32, device=x.device)
    return (torch.empty_like(x), torch.empty((2 * n + 1) if with_ws else 0, **f32), torch.empty((2 * m) if train else 0, **f32))


@opaque(fake=_fwd_fake, name="layernorm_pipe_fwd")
def _fwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, rms: bool, train: bool,
         with_ws: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward of contiguous x [M, N]; (y, ws, stats). ``train``: stats = fp32 [M, 2] (mean, rstd) for the backward; ``with_ws``: ws = the
    backward's fp32 workspace [2N + 1], zeroed here (empty when not asked)."""
    f32 = dict(dtype=torch.float32, device=x.device)
    ws = torch.empty(2 * x.shape[1] + 1 if with_ws else 0, **f32)
    stats = torch.empty(2 * x.shape[0] if train else 0, **f32)
    y = _ext().fwd(x, w, b, float(eps), bool(rms), ws, stats, PER_SM)
    return y, ws, stats


def _bwd_fake(dy, x, w, stats, ws, rms, wcode):
    wdt = {v: k for k, v in _WCODE.items()}[wcode]
    n = x.shape[1] if ws.numel() else 0
    return (torch.empty_like(x), torch.empty((n,), dtype=wdt, device=x.device),
            torch.empty((0 if rms else n,), dtype=wdt, device=x.device))


@opaque(fake=_bwd_fake, name="layernorm_pipe_bwd")
def _bwd(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, stats: torch.Tensor, ws: torch.Tensor, rms: bool,
         wcode: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward: (dx, dw, db); dw / db (``wcode`` dtype) only when ``ws`` is non-empty, db not for RMSNorm. Clears ``ws``."""
    dx, dw, db = _ext().bwd(dy, x, w, stats, ws, bool(rms), int(wcode), PER_SM)
    return dx, dw, db


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, weight, bias, eps, rms):
        w, b, wcode = _affine_args(x2, weight, bias)
        gw = weight is not None and ctx.needs_input_grad[1]
        gb = bias is not None and ctx.needs_input_grad[2]
        y, ws, stats = _fwd(x2, w, b, eps, rms, True, bool(gw or gb))
        ctx.save_for_backward(x2, w, stats, ws)
        ctx.cfg = (rms, wcode, gw, gb, None if weight is None else weight.dtype, None if bias is None else bias.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, w, stats, ws = ctx.saved_tensors
        rms, wcode, gw, gb, wdt, bdt = ctx.cfg
        dx, dw, db = _bwd(dy.contiguous(), x2, w, stats, ws, rms, wcode)
        return dx, (dw.to(wdt) if gw else None), (db.to(bdt) if gb else None), None, None


def layernorm_pipe(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension (``supported(x)``); weight / bias optional."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    needs = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if not needs:
        w, b, _ = _affine_args(x2, weight, bias)
        return _fwd(x2, w, b, float(eps), False, False, False)[0].reshape(shape)
    return _Norm.apply(x2, weight, bias, float(eps), False).reshape(shape)


def rmsnorm_pipe(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm over the last dimension (``supported(x)``); weight optional."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    needs = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight))
    if not needs:
        w, b, _ = _affine_args(x2, weight, None)
        return _fwd(x2, w, b, float(eps), True, False, False)[0].reshape(shape)
    return _Norm.apply(x2, weight, None, float(eps), True).reshape(shape)


__all__ = ["layernorm_pipe", "rmsnorm_pipe", "supported"]
