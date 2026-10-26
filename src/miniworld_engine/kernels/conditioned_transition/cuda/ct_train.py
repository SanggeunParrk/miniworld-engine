"""H100 ConditionedTransition tail, training: forward and backward (``ct_rows.cu`` + cuBLAS).

    ab = x [Wa; Wb]^T        (cuBLAS)       h = silu(a) b                 (``swiglu_fwd``)
    out = h Ws^T             (cuBLAS)       sp = cond Wsc^T + b           (cuBLAS, bias epilogue)
    y = res + sigmoid(sp) out               (``sigmul_fwd``)

and the backward as the mirror (``sigmul_bwd``, ``swiglu_bwd`` and one cuBLAS GEMM per operand gradient). ``x`` is the output of the module's AdaLN
(the kernel is the post-AdaLN tail, as the Triton family's). bf16 activations (fp32 master weights are rounded to bf16 per call and their gradients
come back in fp32) or fp32 on TF32 GEMMs (the process's ``allow_tf32``). Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.adaln.cuda.infer import colsum

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="ct_rows_sm90",
        sources=[str(_dir / "ct_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(), "-O3", "-std=c++17", *gencodes("90a"), "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def supported(x: torch.Tensor, cond: torch.Tensor, nd: int) -> bool:
    """sm_90, bf16, or fp32 under TF32 (``allow_tf32``); the hidden width a multiple of 8."""
    if not (x.is_cuda and cond.is_cuda and x.dtype == cond.dtype):
        return False
    if x.dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return False
    if x.dtype not in (torch.bfloat16, torch.float32) or nd % 8:
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx) == (9, 0)


def _swiglu_fwd_fake(ab):
    return ab.new_empty((ab.shape[0], ab.shape[1] // 2))


@opaque(fake=_swiglu_fwd_fake, name="ct_cuda_swiglu_fwd")
def swiglu_fwd(ab: torch.Tensor) -> torch.Tensor:
    return ext().swiglu_fwd(ab)


@opaque(fake=lambda dh, ab: torch.empty_like(ab), name="ct_cuda_swiglu_bwd")
def swiglu_bwd(dh: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    return ext().swiglu_bwd(dh, ab)


@opaque(fake=lambda out, sp, res: torch.empty_like(out), name="ct_cuda_sigmul_fwd")
def sigmul_fwd(out: torch.Tensor, sp: torch.Tensor, res: torch.Tensor | None) -> torch.Tensor:
    return ext().sigmul_fwd(out, sp, res)


@opaque(fake=lambda dy, out, sp: (torch.empty_like(dy), torch.empty_like(dy), dy.new_empty((1056, dy.shape[-1]), dtype=torch.float32)), name="ct_cuda_sigmul_bwd")
def sigmul_bwd(dy: torch.Tensor, out: torch.Tensor, sp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, dsc, part = ext().sigmul_bwd(dy, out, sp)
    return dout, dsc, part


class _Tail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc, res):  # noqa: D102
        dt = x.dtype
        wab = torch.cat([wa, wb]).to(dt)
        wsb, wscb, bscb = ws.to(dt), wsc.to(dt), bsc.to(dt)
        ab = torch.mm(x, wab.t())
        h = swiglu_fwd(ab)
        out = torch.mm(h, wsb.t())
        sp = torch.addmm(bscb, cond, wscb.t())
        y = sigmul_fwd(out, sp, res)
        ctx.save_for_backward(x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc)
        ctx.has_res = res is not None
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc = ctx.saved_tensors
        dy = dy.contiguous()
        dout, dsc, part = sigmul_bwd(dy, out, sp)
        f32 = torch.float32
        o = lambda w: f32 if (w.dtype == f32 and x.dtype != f32) else None  # noqa: E731  fp32 master weights: fp32 gradients, unrounded
        dws = torch.mm(dout.t(), h, out_dtype=f32) if o(ws) else torch.mm(dout.t(), h)
        dh = torch.mm(dout, wsb)
        dab = swiglu_bwd(dh, ab)
        dwab = torch.mm(dab.t(), x, out_dtype=f32) if o(wa) else torch.mm(dab.t(), x)
        dx = torch.mm(dab, wab)
        dwsc = torch.mm(dsc.t(), cond, out_dtype=f32) if o(wsc) else torch.mm(dsc.t(), cond)
        dcond = torch.mm(dsc, wscb)
        dbsc = colsum(part).to(dsc.dtype if not o(wsc) else f32)   # the block partials of sum(dscale), summed in a fixed order
        nd = wab.shape[0] // 2
        return dx, dcond, dwab[:nd], dwab[nd:], dws, dwsc, dbsc, (dy if ctx.has_res else None)


def ct_tail_train(x, cond, wa, wb, ws, wsc, bsc, residual=None):
    """``residual + sigmoid(cond Wsc^T + b) * (silu(x Wa^T) * (x Wb^T)) Ws^T`` with a backward; ``x`` (M, D), ``cond`` (M, DC) of one dtype, the weights in
    any dtype (rounded to the activations' dtype per call; gradients in the weights' dtype)."""
    return _Tail.apply(x.contiguous(), cond.contiguous(), wa, wb, ws, wsc, bsc, residual)


__all__ = ["ct_tail_train", "ext", "supported"]
