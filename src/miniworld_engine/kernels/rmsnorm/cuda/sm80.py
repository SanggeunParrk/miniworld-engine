"""A100 (sm_80) hand-CUDA RMSNorm and RMSNorm + adaLN modulation, inference and training.

**RMSNorm**: ``y = x rstd w`` over the last axis, ``rstd = 1 / sqrt(mean(x^2) + eps)`` in fp32: the RMS mode of the LayerNorm family's row kernels
(``kernels/layernorm/cuda/sm80/norm_rows.cu``: one 16-byte chunk per lane, the statistic a shuffle over the lanes of a row; the backward keeps ``dw`` as register column partials
over the persistent loop and adds one partial row per CTA in a fixed order).  The widths of the q / k head norms (32 / 48 / 64 / 128) and of the DiT block (128) are all vector
widths.  Same gate as the LayerNorm rows (``layernorm.cuda.sm80.supports``: capability 8.0, engine backend not Triton, ``MINIWORLD_NORMS_SM80`` not 0); training is an autograd
function over two opaque ops (the forward saves x and the row statistic ``rstd``).

**RMSNorm + adaLN modulation** (``rms_norm_modulation``, the atom stream: d_hidden = d_cond = 128, bf16; ``sm80/adamod_sm80.cu``): ``y = rmsnorm(q) (1 + c Wsc^T) + c Wsh^T``,
``gate = c Wg^T`` in one pass -- a persistent CTA keeps the three weight matrices in shared memory and runs the products on the tensor cores over tiles of 64 rows of c and q.
The backward recomputes the scale on the tensor cores and writes the stacked ``[dscale | dy | dgate]`` gradient the caller's two cuBLAS GEMMs consume (the weight gradients and dc).
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

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels.layernorm.cuda import sm80 as _rows

_dir = Path(__file__).parent / "sm80"
_common = Path(_rows.__file__).parent / "sm80"       # norm_common.cuh: the helpers shared with the row kernels
_ADAMOD_FAILED = False
D_MOD = 128                                           # d_hidden = d_cond of the modulation kernel


def supports(x: Tensor, weight: Tensor | None) -> bool:
    """Whether the row kernels run ``rmsnorm(x, weight)``: see ``layernorm.cuda.sm80.supports`` (an RMSNorm has no bias)."""
    return _rows.supports(x, weight, None)


def _rmsnorm_fwd_fake(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """y: the normalised rows, like x."""
    return torch.empty_like(x)


@opaque(fake=_rmsnorm_fwd_fake, name="rmsnorm_sm80_fwd")
def _rmsnorm_fwd(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """Inference forward over the rows of x [M, N] (contiguous): ``x rstd w``, no statistic saved."""
    y = torch.empty_like(x)
    _rows.ext().norm_fwd(x, weight, None, None, y, None, None, eps, True)
    return y


def _rmsnorm_train_fwd_fake(x: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor]:
    """(y like x, rstd [M] fp32)."""
    return torch.empty_like(x), x.new_empty((x.shape[0],), dtype=torch.float32)


@opaque(fake=_rmsnorm_train_fwd_fake, name="rmsnorm_sm80_train_fwd")
def _rmsnorm_train_fwd(x: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor]:
    """Training forward: ``(y, rstd)`` of the rows of x [M, N]; ``rstd`` (fp32) is what the backward reads."""
    y = torch.empty_like(x)
    rstd = x.new_empty((x.shape[0],), dtype=torch.float32)
    _rows.ext().norm_fwd(x, weight, None, None, y, None, rstd, eps, True)
    return y, rstd


def _rmsnorm_train_bwd_fake(dy: Tensor, x: Tensor, weight: Tensor | None, rstd: Tensor, need_dw: bool) -> tuple[Tensor, Tensor]:
    """(dx like x, dweight [N] in the weight's dtype -- empty when not asked for)."""
    dw = weight.new_empty((x.shape[1],)) if need_dw and weight is not None else x.new_empty((0,))
    return torch.empty_like(x), dw


@opaque(fake=_rmsnorm_train_bwd_fake, name="rmsnorm_sm80_train_bwd")
def _rmsnorm_train_bwd(dy: Tensor, x: Tensor, weight: Tensor | None, rstd: Tensor, need_dw: bool) -> tuple[Tensor, Tensor]:
    """Backward of the rows of x [M, N]: ``(dx, dweight)`` from dy and the saved ``rstd``; ``dweight`` is the fixed-order sum of per-CTA partial rows."""
    dx = torch.empty_like(x)
    want = need_dw and weight is not None
    dw = weight.new_empty((x.shape[1],)) if want else x.new_empty((0,))
    _rows.ext().norm_bwd(dy, x, weight, None, None, rstd, dx, dw if want else None, None, True)
    return dx, dw


class _RMSNormTrain(torch.autograd.Function):
    """Forward and backward are one opaque op each; the saved set is x, the row statistic and the weight."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        y, rstd = _rmsnorm_train_fwd(x, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        need_dw = weight is not None and ctx.needs_input_grad[1]
        dx, dw = _rmsnorm_train_bwd(dy.contiguous().to(x.dtype), x, weight, rstd, need_dw)
        return dx, (dw if need_dw else None), None


def rmsnorm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    """``x / sqrt(mean(x^2) + eps) * weight`` over the last axis, autograd-aware.  The caller checked :func:`supports`."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    grad = torch.is_grad_enabled() and (x.requires_grad or (weight is not None and weight.requires_grad))
    y = _RMSNormTrain.apply(x2, weight, eps) if grad else _rmsnorm_fwd(x2, weight, eps)
    return y.view(shape)


def rmsnorm_backward(dy: Tensor, x: Tensor, weight: Tensor | None, rstd: Tensor) -> tuple[Tensor, Tensor | None]:
    """``(dx, dweight)`` from dy, x and the saved ``rstd`` [M] fp32: the kernel the training backward runs, without autograd (the kernel-level bench)."""
    shape = x.shape
    x2, dy2 = x.reshape(-1, shape[-1]).contiguous(), dy.reshape(-1, shape[-1]).contiguous()
    dx, dw = _rmsnorm_train_bwd(dy2, x2, weight, rstd, weight is not None)
    return dx.view(shape), (dw if weight is not None else None)


# ------------------------------------------------------------------------------------------------------------------ RMSNorm + adaLN modulation
@functools.lru_cache(maxsize=1)
def adamod_ext():
    """The modulation extension, built on first use (``MINIWORLD_NORMS_SM80_FLAGS``: extra nvcc flags for A/B experiments, their own build)."""
    ensure_cuda_home()
    extra = os.environ.get("MINIWORLD_NORMS_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"adamod_sm80{tag}",
        sources=[str(_dir / "adamod_sm80.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_common}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


@torch.compiler.assume_constant_result
def adamod_loads() -> bool:
    """Builds (first call) or loads the extension; False, with one warning, when the toolchain fails (the Triton path then serves).  A process-level constant for ``torch.compile``."""
    global _ADAMOD_FAILED
    if _ADAMOD_FAILED:
        return False
    try:
        adamod_ext()
    except Exception as exc:  # noqa: BLE001 -- any toolchain problem keeps the Triton path
        _ADAMOD_FAILED = True
        warnings.warn(f"sm_80 RMSNorm-modulation kernel unavailable, keeping the Triton path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _chunk_ok(t: Tensor) -> bool:
    return t.is_cuda and t.dtype == torch.bfloat16


def _aligned(t: Tensor) -> Tensor:
    """``t`` itself when its base address is 16-byte aligned, else a fresh contiguous copy.  Runs inside the op bodies, on real tensors (the gate cannot look at a storage offset:
    ``torch.compile`` traces it)."""
    return t if t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def supports_adamod(q: Tensor, c: Tensor, w_scale: Tensor, w_shift: Tensor, w_gate: Tensor, weight: Tensor | None = None) -> bool:
    """Whether the kernels run ``rms_norm_modulation``: bf16 q, c of width 128 over the same rows, the three [128, 128] bf16 weights (views of the adaLN projection are fine), an
    RMSNorm weight of 128 in fp32 / bf16 or none; A100, switches on, extension built."""
    if not (_chunk_ok(q) and _chunk_ok(c)) or q.shape[-1] != D_MOD or c.shape[-1] != D_MOD or q.shape[:-1] != c.shape[:-1] or q.numel() == 0:
        return False
    for w in (w_scale, w_shift, w_gate):
        if not _chunk_ok(w) or tuple(w.shape) != (D_MOD, D_MOD) or w.stride() != (D_MOD, 1):
            return False
    if weight is not None and (weight.shape != (D_MOD,) or weight.dtype not in (torch.float32, torch.bfloat16) or not weight.is_cuda):
        return False
    return _rows.enabled(q.device) and adamod_loads()


def _adamod_fwd_fake(q: Tensor, c: Tensor, wsc: Tensor, wsh: Tensor, wg: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor]:
    """(y, gate): like q."""
    return torch.empty_like(q), torch.empty_like(q)


@opaque(fake=_adamod_fwd_fake, name="rmsnorm_adamod_sm80_fwd")
def _adamod_fwd(q: Tensor, c: Tensor, wsc: Tensor, wsh: Tensor, wg: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor]:
    """Inference forward over the rows of q, c [M, 128]: ``(rmsnorm(q) (1 + c Wsc^T) + c Wsh^T, c Wg^T)``, one launch, nothing saved."""
    y, gate = torch.empty_like(q), torch.empty_like(q)
    adamod_ext().adamod_fwd(_aligned(q), _aligned(c), _aligned(wsc), _aligned(wsh), _aligned(wg), weight, y, gate, None, eps)
    return y, gate


def _adamod_train_fwd_fake(q: Tensor, c: Tensor, wsc: Tensor, wsh: Tensor, wg: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    """(y like q, rstd [M] fp32, gate like q)."""
    return torch.empty_like(q), q.new_empty((q.shape[0],), dtype=torch.float32), torch.empty_like(q)


@opaque(fake=_adamod_train_fwd_fake, name="rmsnorm_adamod_sm80_train_fwd")
def _adamod_train_fwd(q: Tensor, c: Tensor, wsc: Tensor, wsh: Tensor, wg: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    """Training forward: ``(y, rstd, gate)``; the row statistic is what the backward reads."""
    y, gate = torch.empty_like(q), torch.empty_like(q)
    rstd = q.new_empty((q.shape[0],), dtype=torch.float32)
    adamod_ext().adamod_fwd(_aligned(q), _aligned(c), _aligned(wsc), _aligned(wsh), _aligned(wg), weight, y, gate, rstd, eps)
    return y, rstd, gate


def _adamod_train_bwd_fake(dy: Tensor, dgate: Tensor, q: Tensor, c: Tensor, wsc: Tensor, weight: Tensor | None, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dq like q, the stacked [dscale | dy | dgate] [M, 384], dweight [128] fp32 -- empty without a weight)."""
    return torch.empty_like(q), q.new_empty((q.shape[0], 3 * q.shape[1])), q.new_empty((D_MOD if weight is not None else 0,), dtype=torch.float32)


@opaque(fake=_adamod_train_bwd_fake, name="rmsnorm_adamod_sm80_train_bwd")
def _adamod_train_bwd(dy: Tensor, dgate: Tensor, q: Tensor, c: Tensor, wsc: Tensor, weight: Tensor | None, rstd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The elementwise half of the backward: ``(dq, [dscale | dy | dgate], dweight)``; ``scale`` is recomputed on the tensor cores, ``dweight`` (fp32) is a few atomic adds."""
    dq = torch.empty_like(q)
    dsd = q.new_empty((q.shape[0], 3 * q.shape[1]))
    dw = torch.zeros((D_MOD,), device=q.device, dtype=torch.float32) if weight is not None else q.new_empty((0,), dtype=torch.float32)
    adamod_ext().adamod_bwd(_aligned(dy), _aligned(dgate), _aligned(q), _aligned(c), _aligned(wsc), weight, rstd, dq, dsd, dw if weight is not None else None)
    return dq, dsd, dw


class _RMSNormAdaMod(torch.autograd.Function):
    """The Triton function's contract (same saved set, same GEMMs over the stacked gradient), the two kernels of the stage replaced."""

    @staticmethod
    def forward(ctx, q2, c2, wsc, wsh, wg, weight, eps):
        y, rstd, gate = _adamod_train_fwd(q2, c2, wsc, wsh, wg, weight, eps)
        ctx.save_for_backward(q2, c2, wsc, wsh, wg, weight, rstd)
        return y, gate

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, dgate):
        q2, c2, wsc, wsh, wg, weight, rstd = ctx.saved_tensors
        dq, dsd, dw = _adamod_train_bwd(dy.contiguous(), dgate.contiguous(), q2, c2, wsc, weight, rstd)
        n = q2.shape[1]
        w_stack = torch.cat((wsc, wsh, wg), dim=0)                 # [3N, K]
        dw_stack = dsd.mT @ c2                                     # [3N, K]  one GEMM
        dc = dsd @ w_stack                                         # [M, K]   one GEMM
        dweight = dw.to(weight.dtype) if weight is not None else None
        return dq, dc, dw_stack[:n].to(wsc.dtype), dw_stack[n:2 * n].to(wsh.dtype), dw_stack[2 * n:].to(wg.dtype), dweight, None


def rmsnorm_adamod(q: Tensor, c: Tensor, w_scale: Tensor, w_shift: Tensor, w_gate: Tensor, weight: Tensor | None = None, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """``rmsnorm(q) * (1 + c @ w_scale^T) + c @ w_shift^T`` and ``c @ w_gate^T``, autograd-aware (``c`` is the activated conditioning, as the Triton entry's).  The caller checked
    :func:`supports_adamod`."""
    shape = q.shape
    q2, c2 = q.reshape(-1, shape[-1]), c.reshape(-1, c.shape[-1])
    if not q2.is_contiguous():
        q2 = q2.contiguous()
    if not c2.is_contiguous():
        c2 = c2.contiguous()
    w_scale, w_shift, w_gate = w_scale.contiguous(), w_shift.contiguous(), w_gate.contiguous()
    grad = torch.is_grad_enabled() and (q.requires_grad or c.requires_grad or w_scale.requires_grad or w_shift.requires_grad or w_gate.requires_grad
                                        or (weight is not None and weight.requires_grad))
    if grad:
        y, gate = _RMSNormAdaMod.apply(q2, c2, w_scale, w_shift, w_gate, weight, eps)
    else:
        y, gate = _adamod_fwd(q2, c2, w_scale, w_shift, w_gate, weight, eps)
    return y.view(shape), gate.view(shape)
