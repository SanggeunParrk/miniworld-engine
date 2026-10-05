"""A100 (sm_80) hand-CUDA LayerNorm / RMSNorm rows, inference and training (``sm80/norm_rows.cu``).

Both norms normalise the last axis of a bf16 (or fp32) activation with fp32 statistics, an fp32 or bf16 affine (or none), an optional per-row scale (the AF pair mask folded
into the epilogue).  A row is read once and written once (the byte floor of the op):

* **vector path** (width a multiple of 8 bf16 / 4 fp32, 16-byte aligned): a row is NV 16-byte chunks owned by G lanes (a power of two dividing NV, at most 32), so a warp serves
  32 / G rows with coalesced loads and the statistics are a shuffle over the lane group; a one-shot grid of 128-thread CTAs, a warp per row group (or per two or four of them for
  a mid-sized bf16 problem, where one narrow group a warp keeps too few bytes in flight).
* **scalar path** (any other width, e.g. 267 / 451 / 831 / 833, or an unaligned view) and a **staged path** for the odd widths with many rows (tiles copied to shared memory with
  ``cp.async``, normalised there, copied back).
* **backward**: ``dx`` row by row (the row sums over the same lane groups), ``dw`` / ``db`` as register column partials over a persistent loop, folded through shared memory into
  one fp32 partial row per CTA and added in a fixed order by a second small kernel (no atomics).  A small M has static row assignment and is bit-reproducible; a large M takes
  chunks of rows from a work counter, so which rows land in which partial row varies and ``dw`` / ``db`` agree between runs to fp32 rounding (``dx`` and the forward are always
  bit-reproducible).

``supports`` is the whole gate (capability 8.0, engine backend not forced to Triton, ``MINIWORLD_NORMS_SM80`` not 0, a bf16 / fp32 activation, widths up to 4096); everything it
refuses keeps the Triton path.  A failed extension build warns once and keeps it too.  Training is an autograd function over two opaque ops (the forward saves x and the
row statistics; the backward needs no recomputation); parameter gradients come back in the parameters' dtype.  ``kernels/rmsnorm/cuda/sm80.py`` drives the RMSNorm side.
"""

from __future__ import annotations

import functools
import hashlib
import os
import warnings
from pathlib import Path

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels._compile import device_constant, opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

#: widest row served: the scalar backward keeps 4 warps x 2 x N fp32 of shared memory
MAX_N = 4096
_FAILED = False


@functools.lru_cache(maxsize=1)
def ext():
    """The row-kernel extension, built on first use (``MINIWORLD_NORMS_SM80_FLAGS``: extra nvcc flags for A/B experiments, their own build)."""
    ensure_cuda_home()
    extra = os.environ.get("MINIWORLD_NORMS_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"norm_rows_sm80{tag}",
        sources=[str(_dir / "norm_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


@torch.compiler.assume_constant_result
def loads() -> bool:
    """Builds (first call) or loads the extension; False, with one warning, when the toolchain fails (the Triton path then serves).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _FAILED
    if _FAILED:
        return False
    try:
        ext()
    except Exception as exc:  # noqa: BLE001 -- any toolchain problem keeps the Triton path
        _FAILED = True
        warnings.warn(f"sm_80 norm kernels unavailable, keeping the Triton path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


@device_constant
@functools.lru_cache(maxsize=8)
def _is_a100(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (8, 0)


def enabled(device: torch.device) -> bool:
    """The switches every norm kernel of this card shares: ``MINIWORLD_NORMS_SM80`` not 0, the engine backend not forced to Triton, capability exactly 8.0."""
    if os.environ.get("MINIWORLD_NORMS_SM80", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    return _is_a100(device.index if device.index is not None else torch.cuda.current_device())


def _params_ok(n: int, *params: Tensor | None) -> bool:
    """Each parameter None or a CUDA vector of the row width in fp32 / bf16, all of one dtype (the kernels read them as one)."""
    seen = {p.dtype for p in params if p is not None}
    if len(seen) > 1 or not seen <= {torch.float32, torch.bfloat16}:
        return False
    return all(p is None or (p.is_cuda and p.ndim == 1 and p.shape[0] == n) for p in params)


#: Training steps this small are launch-bound: the forward, the backward and the dw / db reduction are three kernels of ~3 us each, and the Triton path's forward + one atomic backward wins by 3-35 % there
#: (measured 2026-10-04, one process, CUDA graph: token_single L128 at 384 / 768 wide, the MSA stream at L128 / 64 wide; no cell above 100 K elements, none below 20 K, was slower).  Those calls keep the Triton path;
#: ``MINIWORLD_NORMS_SM80=force`` runs the CUDA rows at every size (the kernels are reachable, and tested, for them).
_TINY_TRAIN_ELEMS = (20_000, 100_000)
_TINY_TRAIN_WIDTHS = (64, 768)


def _triton_wins(x: Tensor, weight: Tensor | None, bias: Tensor | None) -> bool:
    """A training call (a gradient is needed) of a vector-path width over so few elements that the Triton path is faster; False otherwise.  Shapes and flags only: ``torch.compile`` traces it."""
    n = x.shape[-1]
    if n % 8 or not _TINY_TRAIN_WIDTHS[0] <= n <= _TINY_TRAIN_WIDTHS[1] or not _TINY_TRAIN_ELEMS[0] <= x.numel() <= _TINY_TRAIN_ELEMS[1]:
        return False
    return torch.is_grad_enabled() and (x.requires_grad or (weight is not None and weight.requires_grad) or (bias is not None and bias.requires_grad))


def supports(x: Tensor, weight: Tensor | None, bias: Tensor | None, row_scale: Tensor | None = None) -> bool:
    """Whether the kernels run this call: a CUDA bf16 / fp32 activation of width 1..4096, an fp32 / bf16 affine (or none), A100, switches on, extension built, and not one of the tiny training
    steps the Triton path wins (:func:`_triton_wins`; ``MINIWORLD_NORMS_SM80=force`` serves those too)."""
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32) or x.ndim < 1 or not 1 <= x.shape[-1] <= MAX_N or x.numel() == 0:
        return False
    if not _params_ok(x.shape[-1], weight, bias) or (row_scale is not None and row_scale.numel() != x.numel() // x.shape[-1]):
        return False
    if _triton_wins(x, weight, bias) and os.environ.get("MINIWORLD_NORMS_SM80", "1") != "force":
        return False
    return enabled(x.device) and loads()


# ------------------------------------------------------------------------------------------------------------------ forward ops
def _layernorm_fwd_fake(x: Tensor, weight: Tensor | None, bias: Tensor | None, row_scale: Tensor | None, eps: float) -> Tensor:
    """y: the normalised rows, like x."""
    return torch.empty_like(x)


@opaque(fake=_layernorm_fwd_fake, name="layernorm_sm80_fwd")
def _layernorm_fwd(x: Tensor, weight: Tensor | None, bias: Tensor | None, row_scale: Tensor | None, eps: float) -> Tensor:
    """Inference forward over the rows of x [M, N] (contiguous): ``y = LN(x) w + b`` (times ``row_scale`` [M] in x's dtype when given), no statistics saved."""
    y = torch.empty_like(x)
    ext().norm_fwd(x, weight, bias, row_scale, y, None, None, eps, False)
    return y


def _layernorm_train_fwd_fake(x: Tensor, weight: Tensor | None, bias: Tensor | None, row_scale: Tensor | None, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    """(y like x, mean [M], rstd [M]): the statistics are fp32 whatever x's dtype is."""
    m = x.shape[0]
    return torch.empty_like(x), x.new_empty((m,), dtype=torch.float32), x.new_empty((m,), dtype=torch.float32)


@opaque(fake=_layernorm_train_fwd_fake, name="layernorm_sm80_train_fwd")
def _layernorm_train_fwd(x: Tensor, weight: Tensor | None, bias: Tensor | None, row_scale: Tensor | None, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    """Training forward: ``(y, mean, rstd)`` of the rows of x [M, N]; the statistics are what the backward reads."""
    m = x.shape[0]
    y = torch.empty_like(x)
    mean = x.new_empty((m,), dtype=torch.float32)
    rstd = x.new_empty((m,), dtype=torch.float32)
    ext().norm_fwd(x, weight, bias, row_scale, y, mean, rstd, eps, False)
    return y, mean, rstd


def _layernorm_train_bwd_fake(dy: Tensor, x: Tensor, weight: Tensor | None, bias: Tensor | None, row_scale: Tensor | None, mean: Tensor, rstd: Tensor,
                              need_dw: bool, need_db: bool) -> tuple[Tensor, Tensor, Tensor]:
    """(dx like x, dweight, dbias): the parameter gradients are [N] in the parameters' dtype, and empty when not asked for."""
    n = x.shape[1]
    dw = weight.new_empty((n,)) if need_dw and weight is not None else x.new_empty((0,))
    db = bias.new_empty((n,)) if need_db and bias is not None else x.new_empty((0,))
    return torch.empty_like(x), dw, db


@opaque(fake=_layernorm_train_bwd_fake, name="layernorm_sm80_train_bwd")
def _layernorm_train_bwd(dy: Tensor, x: Tensor, weight: Tensor | None, bias: Tensor | None, row_scale: Tensor | None, mean: Tensor, rstd: Tensor, need_dw: bool,
                         need_db: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Backward of the rows of x [M, N] from dy [M, N] and the forward's statistics: ``(dx, dweight, dbias)``; one persistent kernel and a fixed-order reduction of its
    per-CTA partial sums (``dx`` is bit-reproducible; ``dweight`` / ``dbias`` agree between runs to fp32 rounding when the rows are handed out dynamically, at large M)."""
    n = x.shape[1]
    dx = torch.empty_like(x)
    dw = weight.new_empty((n,)) if need_dw and weight is not None else x.new_empty((0,))
    db = bias.new_empty((n,)) if need_db and bias is not None else x.new_empty((0,))
    ext().norm_bwd(dy, x, weight, row_scale, mean, rstd, dx, dw if need_dw and weight is not None else None, db if need_db and bias is not None else None, False)
    return dx, dw, db


# ------------------------------------------------------------------------------------------------------------------ autograd
class _LayerNormTrain(torch.autograd.Function):
    """Forward and backward are one opaque op each; the saved set is x, the statistics, the weight (for dx) and the row scale."""

    @staticmethod
    def forward(ctx, x, weight, bias, row_scale, eps):
        y, mean, rstd = _layernorm_train_fwd(x, weight, bias, row_scale, eps)
        ctx.save_for_backward(x, weight, bias, row_scale, mean, rstd)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, bias, row_scale, mean, rstd = ctx.saved_tensors
        need_dw = weight is not None and ctx.needs_input_grad[1]
        need_db = bias is not None and ctx.needs_input_grad[2]
        dx, dw, db = _layernorm_train_bwd(dy.contiguous().to(x.dtype), x, weight, bias, row_scale, mean, rstd, need_dw, need_db)
        return dx, (dw if need_dw else None), (db if need_db else None), None, None


def layernorm(x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float = 1e-5, row_scale: Tensor | None = None) -> Tensor:
    """``LayerNorm(x) * weight + bias`` over the last axis (optionally times ``row_scale``, one factor per row), autograd-aware.  The caller checked :func:`supports`."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    rs = None if row_scale is None else row_scale.reshape(-1).to(x2.dtype).contiguous()
    grad = torch.is_grad_enabled() and (x.requires_grad or (weight is not None and weight.requires_grad) or (bias is not None and bias.requires_grad))
    y = _LayerNormTrain.apply(x2, weight, bias, rs, eps) if grad else _layernorm_fwd(x2, weight, bias, rs, eps)
    return y.view(shape)


def layernorm_backward(dy: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor, row_scale: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """``(dx, dweight, dbias)`` of the rows of x from dy and the statistics (``mean`` / ``rstd`` [M] fp32): the kernel the training backward runs, without autograd (the
    kernel-level bench).  ``weight`` fixes the dtype of the two parameter gradients."""
    shape = x.shape
    x2, dy2 = x.reshape(-1, shape[-1]).contiguous(), dy.reshape(-1, shape[-1]).contiguous()
    dx, dw, db = _layernorm_train_bwd(dy2, x2, weight, weight, row_scale, mean, rstd, True, True)
    return dx.view(shape), dw, db
