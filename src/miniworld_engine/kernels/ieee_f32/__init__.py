"""IEEE fp32 CUDA GEMM and the fp32 Transition built on it (A100 sm_80; the same sources built for sm_120a on the RTX 5090).

``torch.set_float32_matmul_precision("highest")`` / ``allow_tf32 = False`` means IEEE fp32 products: no tensor core, no TF32 rounding.  The
engine's fp32 paths used to hand such a call to cuBLAS's simt kernels (or to Triton).  ``ieee_f32.cu`` is the engine's own kernel: a 128 x 128 x 8
FFMA tile with fused epilogues (SwiGLU, residual, SwiGLU backward) plus the LayerNorm row kernels, fixed accumulation order (bitwise repeatable).

``mm(a, b)``: ``a @ b`` for fp32 2-D operands in any of the four transposition layouts (a view ``t.t()`` of a contiguous tensor is the transposed
layout, no copy); every dimension a multiple of 4.  ``transition_ieee(x, gamma, beta, wa, wb, ws, eps)``: ``x + transition(x)`` with a backward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import build_failed, ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels._sm120_builds import per_arch, served

_dir = Path(__file__).parent

#: the widest LayerNorm row the row kernels hold in registers (D <= 1024)
MAX_D = 1024


def _build(suffix: str, gencode: str):
    ensure_cuda_home()
    return load_extension(
        name=f"ieee_f32_{suffix}",
        sources=[str(_dir / "ieee_f32.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", gencode],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


_ext, _ext_sm80, _ext_sm120 = per_arch(_build)


def _x() -> Any:
    """The current card's extension (typed Any: the pybind functions)."""
    return _ext()


@torch.compiler.assume_constant_result
def loads() -> bool:
    """The extension builds / loads (cached); a failed build raises ``_nvcc.KernelBuildError``."""
    try:
        _ext()
    except Exception as exc:
        build_failed("ieee f32", exc)
        raise
    return True


def available(x: torch.Tensor) -> bool:
    """fp32 CUDA tensor on a card the sm_80 sources serve (A100, RTX 5090)."""
    return x.is_cuda and x.dtype is torch.float32 and served(x.device) and loads()


# ---------------------------------------------------------------------------------------------------------------------------- generic mm
def _mm_fake(a, lda, la, b, ldb, lb, m, n, k, bias, out):
    return a.new_empty(m, n) if out is None else out.new_empty(0)


@opaque(fake=_mm_fake, name="ieee_f32_mm", mutates_args=("out",))
def _mm(a: torch.Tensor, lda: int, la: int, b: torch.Tensor, ldb: int, lb: int, m: int, n: int, k: int, bias: torch.Tensor | None, out: torch.Tensor | None) -> torch.Tensor:
    r = _x().mm(a, lda, la, b, ldb, lb, m, n, k, bias, out)
    return r if out is None else out.new_empty(0)


def _reduce_fake(t):
    return t.new_empty(t.shape[1:])


@opaque(fake=_reduce_fake, name="ieee_f32_reduce_slabs")
def _reduce(t: torch.Tensor) -> torch.Tensor:
    return _x().reduce_slabs(t)


def _operand(t: torch.Tensor, a_side: bool) -> tuple[torch.Tensor, int, int]:
    """(tensor, leading dimension, layout code) of a 2-D operand; the A side [M, K] is code 0 (row major) / 1 (a view of a [K, M] tensor), the B side [K, N] code 1 (row major) / 0
    (a view of a [N, K] tensor)."""
    if t.stride(1) == 1 and t.stride(0) % 4 == 0:
        return t, t.stride(0), 0 if a_side else 1
    if t.stride(0) == 1 and t.stride(1) % 4 == 0:
        return t, t.stride(1), 1 if a_side else 0
    t = t.contiguous()
    return t, t.stride(0), 0 if a_side else 1


def mm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a @ b`` (+ ``bias`` [N]), IEEE fp32 (a [M, K], b [K, N], contiguous or a transposed view of a contiguous tensor), a contiguous [M, N]."""
    m, k = a.shape
    n = b.shape[1]
    ao, lda, la = _operand(a, True)
    bo, ldb, lb = _operand(b, False)
    r = _mm(ao, lda, la, bo, ldb, lb, m, n, k, None if bias is None else bias.to(torch.float32).contiguous(), out)
    return r if out is None else out


def supported_mm(a: torch.Tensor, b: torch.Tensor) -> bool:
    m, k = a.shape
    n = b.shape[1]
    return m % 4 == 0 and n % 4 == 0 and k % 4 == 0 and available(a) and b.dtype is torch.float32 and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0 and a.device == b.device


# ---------------------------------------------------------------------------------------------------------------------------- Transition
def _ln_fwd_fake(x, gamma, beta, eps, save):
    return torch.empty_like(x), x.new_empty(x.shape[0], 2) if save else x.new_empty(0)


@opaque(fake=_ln_fwd_fake, name="ieee_f32_ln_fwd")
def _ln_fwd(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor]:
    xn, stats = _x().ln_fwd(x, gamma, beta, float(eps), bool(save))
    return xn, stats


def _swiglu_fake(x, wa, wb, save):
    m, h = x.shape[0], wa.shape[0]
    return x.new_empty(m, h), x.new_empty(m, 2 * h) if save else x.new_empty(0)


@opaque(fake=_swiglu_fake, name="ieee_f32_swiglu_fwd")
def _swiglu(x: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, save: bool) -> tuple[torch.Tensor, torch.Tensor]:
    h, ab = _x().swiglu_fwd(x, wa, wb, bool(save))
    return h, ab


def _res_fake(h, ws, x):
    return torch.empty_like(x)


@opaque(fake=_res_fake, name="ieee_f32_gemm_res")
def _gemm_res(h: torch.Tensor, ws: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _x().gemm_res(h, ws, x)


def _dab_fake(dy, ws, ab):
    return torch.empty_like(ab)


@opaque(fake=_dab_fake, name="ieee_f32_dab")
def _dab(dy: torch.Tensor, ws: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    return _x().dab_gemm(dy, ws, ab)


def _stacked_fake(a, b0, b1):
    return a.new_empty(a.shape[0], b0.shape[1])


@opaque(fake=_stacked_fake, name="ieee_f32_mm_stacked")
def _stacked(a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    return _x().mm_stacked(a, b0, b1)


def _ln_bwd_fake(dxn, x, stats, gamma, dy):
    return torch.empty_like(x), x.new_empty(2 * x.shape[1])


@opaque(fake=_ln_bwd_fake, name="ieee_f32_ln_bwd")
def _ln_bwd(dxn: torch.Tensor, x: torch.Tensor, stats: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx, part = _x().ln_bwd(dxn, x, stats, gamma, dy)
    return dx, _x().reduce_slabs(part)


def _wgrad_fake(a, b):
    return a.new_empty(a.shape[1], b.shape[1])


@opaque(fake=_wgrad_fake, name="ieee_f32_wgrad")
def _wgrad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a^T b`` for a [K, Mo], b [K, No] (contiguous): the weight gradient, split along K."""
    k, mo = a.shape
    no = b.shape[1]
    return _x().mm(a, mo, 1, b, no, 1, mo, no, k, None, None)


def supported_transition(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """fp32 activations [..., D] with D a multiple of 4 up to ``MAX_D``, hidden H a multiple of 4."""
    d = x.shape[-1]
    h = wa.shape[0]
    return (
        available(x)
        and d % 4 == 0
        and d <= MAX_D
        and h % 4 == 0
        and tuple(wa.shape) == (h, d)
        and tuple(ws.shape) == (d, h)
        and x.numel() > 0
    )


class _IeeeTransition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        xn, stats = _ln_fwd(flat, gamma, beta, eps, True)
        h, ab = _swiglu(xn, wa, wb, True)
        out = _gemm_res(h, ws, flat)
        ctx.save_for_backward(flat, xn, stats, gamma, wa, wb, ws, ab, h)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, stats, gamma, wa, wb, ws, ab, h = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1]).contiguous()
        dab = _dab(dy, ws, ab)
        dwab = _wgrad(dab, xn)  # [2H, D]
        dws = _wgrad(dy, h)  # [D, H]
        dxn = _stacked(dab, wa, wb)
        dx, dgb = _ln_bwd(dxn, flat, stats, gamma, dy)
        d = flat.shape[1]
        hid = wa.shape[0]
        return dx.reshape(ctx.shape), dgb[:d], dgb[d:], dwab[:hid], dwab[hid:], dws, None


def transition_ieee(x, gamma, beta, wa, wb, ws, eps):
    """``x + ws (silu(wa LN(x)) * (wb LN(x)))`` in IEEE fp32 (call ``supported_transition`` first); all operands fp32. Differentiable."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws)):
        return _IeeeTransition.apply(x, gamma, beta, wa, wb, ws, eps)
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).contiguous()
    xn, _ = _ln_fwd(flat, gamma, beta, eps, False)
    h, _ = _swiglu(xn, wa, wb, False)
    return _gemm_res(h, ws, flat).reshape(shape)


# ---------------------------------------------------------------------------------------------------------------------------- drop-in for the engine's own fp32 torch.mm / torch.addmm calls
def tf32_allowed() -> bool:
    """torch lets fp32 products run on TF32 tensor cores (``allow_tf32`` or a matmul precision other than "highest"): else fp32 means IEEE."""
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def _layout_ok(t: torch.Tensor) -> bool:
    return t.dim() == 2 and ((t.stride(1) == 1 and t.stride(0) % 4 == 0) or (t.stride(0) == 1 and t.stride(1) % 4 == 0))


def _out_ok(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0


def _owned(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The call is the engine kernel's: fp32 operands on a served card, TF32 off (IEEE: torch would run cuBLAS's simt kernels), no autograd record to keep, and a layout the kernel indexes."""
    if a.dtype is not torch.float32 or b.dtype is not torch.float32 or tf32_allowed():
        return False
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return False
    return a.dim() == 2 and b.dim() == 2 and _layout_ok(a) and _layout_ok(b) and supported_mm(a, b)


def torch_mm(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """``torch.mm(a, b[, out=out])``; an fp32 call with TF32 off runs on the engine's IEEE fp32 GEMM instead of cuBLAS's simt kernels."""
    if _owned(a, b) and (out is None or (out.dtype is torch.float32 and _out_ok(out))):
        return mm(a, b, out=out)
    return torch.mm(a, b) if out is None else torch.mm(a, b, out=out)


def torch_addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.addmm(bias, a, b)`` for a ``bias`` [N]; fp32 with TF32 off: the engine's IEEE GEMM with the bias in its epilogue."""
    if bias.dim() == 1 and bias.dtype is torch.float32 and not (torch.is_grad_enabled() and bias.requires_grad) and _owned(a, b):
        return mm(a, b, bias=bias)
    return torch.addmm(bias, a, b)


class _Linear(torch.autograd.Function):
    """``x w^T + b`` in IEEE fp32 on the engine's GEMM: forward ``mm`` (bias in the epilogue), backward ``dx = dy w``, ``dw = dy^T x`` (split-K), ``db = sum dy`` (a row GEMM
    against ones would round differently: the column sum is its own deterministic reduction)."""

    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        ctx.has_bias = b is not None
        return mm(x, w.t(), bias=b)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy = dy.contiguous()
        dx = mm(dy, w) if ctx.needs_input_grad[0] else None
        dw = _wgrad(dy, x) if ctx.needs_input_grad[1] else None
        db = _reduce(dy.unsqueeze(0)) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``F.linear(x, weight, bias)``: fp32 with TF32 off on the engine's IEEE GEMM (autograd through ``_Linear``), else torch's. ``x`` [..., K], ``weight`` [N, K], ``bias`` [N]."""
    import torch.nn.functional as F

    flat = x.reshape(-1, x.shape[-1])
    ok = (
        x.dtype is torch.float32
        and weight.dtype is torch.float32
        and (bias is None or bias.dtype is torch.float32)
        and not tf32_allowed()
        and flat.is_contiguous()
        and weight.is_contiguous()
        and supported_mm(flat, weight.t())
    )
    if not ok:
        return F.linear(x, weight, bias)
    y = _Linear.apply(flat, weight, bias) if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad) else mm(flat, weight.t(), bias=bias)
    return y.reshape(*x.shape[:-1], weight.shape[0])
