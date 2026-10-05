"""A100 (sm_80) hand-CUDA 3D RoPE and fused Q/K RMSNorm + RoPE, inference and training (``sm80/rope_sm80.cu``).

The fused op is what ``modules/swa_atom_attention`` runs on the q / k views of the interleaved QKV projection: per (position, head) row an RMSNorm over the head dim (no weight, fp32
statistic, the normalised row rounded to the activation dtype as the Triton kernel does), then the rotation of the leading ``2 HALF`` channels by the position's cos / sin; the
backward is one fused kernel as well (rotate the gradient back, round, the RMSNorm input gradient).  A row is owned by ``D / 8`` lanes (bf16), so one warp serves a position's q AND
k rows at 4 heads x 32 channels and the cos / sin are read once; the rotation partner chunk comes through a shuffle.  Inputs are the strided views (unit channel stride, 16-byte
aligned rows), outputs contiguous [N, S, H, D].  ``rope_3d`` is the standalone rotation (its backward is the same kernel with the angle negated).

``supports_*`` are the gates (A100, switches on, bf16 / fp32, head dim 32 / 64 / 128, a rotary half that is a multiple of one 16-byte chunk, fp32 angle tables of [N or 1, S, HALF]);
everything they refuse keeps the Triton kernels.  Training is an autograd function over two opaque ops; the angle tables are constants.
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
_FAILED = False
HEAD_DIMS = (32, 64, 128)


@functools.lru_cache(maxsize=1)
def ext():
    """The RoPE extension, built on first use (``MINIWORLD_NORMS_SM80_FLAGS``: extra nvcc flags for A/B experiments, their own build)."""
    ensure_cuda_home()
    extra = os.environ.get("MINIWORLD_NORMS_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"rope_sm80{tag}",
        sources=[str(_dir / "rope_sm80.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_common}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


@torch.compiler.assume_constant_result
def loads() -> bool:
    """Builds (first call) or loads the extension; False, with one warning, when the toolchain fails (the Triton path then serves).  A process-level constant for ``torch.compile``."""
    global _FAILED
    if _FAILED:
        return False
    try:
        ext()
    except Exception as exc:  # noqa: BLE001 -- any toolchain problem keeps the Triton path
        _FAILED = True
        warnings.warn(f"sm_80 RoPE kernels unavailable, keeping the Triton path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _view_ok(t: Tensor, ce: int) -> bool:
    """A [N, S, H, D] view the kernel can read with 16-byte loads, as far as the gate can tell (shape and strides only: unit channel stride, strides in whole chunks); the base address
    is checked where the op runs (:func:`_readable`) -- ``torch.compile`` traces the gate, and a storage offset is not something it can trace."""
    s = t.stride()
    return s[3] == 1 and s[0] % ce == 0 and s[1] % ce == 0 and s[2] % ce == 0


def _readable(t: Tensor) -> Tensor:
    """``t`` itself when its base address is 16-byte aligned (its strides were checked by the gate), else a fresh contiguous copy.  Runs inside the op bodies, on real tensors."""
    return t if t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def _angles_ok(cos: Tensor, sin: Tensor, n: int, s: int, d: int, ce: int) -> bool:
    if cos.dtype != torch.float32 or sin.dtype != torch.float32 or cos.ndim != 3 or cos.shape != sin.shape or cos.stride() != sin.stride():
        return False
    if cos.shape[0] not in (1, n) or cos.shape[1] != s or cos.requires_grad or sin.requires_grad:
        return False
    half = cos.shape[2]
    st = cos.stride()
    return half % ce == 0 and 2 * half <= d and st[2] == 1 and st[0] % 4 == 0 and st[1] % 4 == 0 and cos.is_cuda and sin.device == cos.device


def _shape_ok(x: Tensor) -> bool:
    return (x.is_cuda and x.ndim == 4 and x.dtype in (torch.bfloat16, torch.float32) and x.shape[3] in HEAD_DIMS and x.numel() > 0
            and 2 * x.shape[0] * x.shape[1] * x.shape[2] < 2**31)


def supports_qk(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> bool:
    """Whether the kernels run ``qk_norm_rope_3d(q, k, cos, sin)``: q, k [N, S, H, D] views of one shape and dtype, the angle tables constants fp32 [N or 1, S, HALF]."""
    if not _shape_ok(q) or q.shape != k.shape or q.dtype != k.dtype or k.device != q.device:
        return False
    ce = 16 // q.element_size()
    if not (_view_ok(q, ce) and _view_ok(k, ce) and _angles_ok(cos, sin, q.shape[0], q.shape[1], q.shape[3], ce)):
        return False
    return _rows.enabled(q.device) and loads()


def supports_rope(x: Tensor, cos: Tensor, sin: Tensor) -> bool:
    """Whether the kernels run ``rope_3d(x, cos, sin)`` (x [N, S, H, D])."""
    if not _shape_ok(x):
        return False
    ce = 16 // x.element_size()
    if not (_view_ok(x, ce) and _angles_ok(cos, sin, x.shape[0], x.shape[1], x.shape[3], ce)):
        return False
    return _rows.enabled(x.device) and loads()


# ------------------------------------------------------------------------------------------------------------------ the fused norm + rotation
def _qk_norm_rope_fwd_fake(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """The normalised and rotated q and k: contiguous, like the inputs' shape and dtype."""
    return torch.empty_like(q, memory_format=torch.contiguous_format), torch.empty_like(k, memory_format=torch.contiguous_format)


@opaque(fake=_qk_norm_rope_fwd_fake, name="qk_norm_rope_sm80_fwd")
def _qk_norm_rope_fwd(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """Paired q / k RMSNorm (over the head dim, no weight) and 3D RoPE: ``(oq, ok)``, one launch."""
    oq = torch.empty_like(q, memory_format=torch.contiguous_format)
    ok = torch.empty_like(k, memory_format=torch.contiguous_format)
    ext().qk_norm_rope_fwd(_readable(q), _readable(k), _readable(cos), _readable(sin), oq, ok, eps)
    return oq, ok


def _qk_norm_rope_bwd_fake(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, gq: Tensor, gk: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """The input gradients of q and k: contiguous, with the forward inputs' shape and dtype."""
    return torch.empty_like(q, memory_format=torch.contiguous_format), torch.empty_like(k, memory_format=torch.contiguous_format)


@opaque(fake=_qk_norm_rope_bwd_fake, name="qk_norm_rope_sm80_bwd")
def _qk_norm_rope_bwd(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, gq: Tensor, gk: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """The fused input backward of :func:`_qk_norm_rope_fwd`: ``(dq, dk)`` from the inputs and the gradients of the rotated outputs, one launch.  The gradients may arrive in any
    layout (an expanded or transposed view): they are made readable here."""
    dq = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk = torch.empty_like(k, memory_format=torch.contiguous_format)
    ce = 16 // q.element_size()
    gq = gq if _view_ok(gq, ce) else gq.contiguous()
    gk = gk if _view_ok(gk, ce) else gk.contiguous()
    ext().qk_norm_rope_bwd(_readable(q), _readable(k), _readable(cos), _readable(sin), _readable(gq), _readable(gk), dq, dk, eps)
    return dq, dk


class _QKNormRoPE(torch.autograd.Function):
    """Forward and backward are one opaque op each; the saved set is q, k and the (constant) angle tables."""

    @staticmethod
    def forward(ctx, q, k, cos, sin, eps):
        oq, ok = _qk_norm_rope_fwd(q, k, cos, sin, eps)
        ctx.save_for_backward(q, k, cos, sin)
        ctx.eps = eps
        return oq, ok

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gk):
        q, k, cos, sin = ctx.saved_tensors
        dq, dk = _qk_norm_rope_bwd(q, k, cos, sin, gq, gk, ctx.eps)
        return dq, dk, None, None, None


def qk_norm_rope_3d(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, eps: float = torch.finfo(torch.float32).eps) -> tuple[Tensor, Tensor]:
    """Normalise (RMSNorm over the head dim, no weight) and rotate q and k, autograd-aware.  The caller checked :func:`supports_qk`."""
    return _QKNormRoPE.apply(q, k, cos, sin, eps)


# ------------------------------------------------------------------------------------------------------------------ the standalone rotation
def _rope_rotate_fake(x: Tensor, cos: Tensor, sin: Tensor, sign: float) -> Tensor:
    """The rotated activation: contiguous, like x's shape and dtype."""
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@opaque(fake=_rope_rotate_fake, name="rope_sm80_rotate")
def _rope_rotate(x: Tensor, cos: Tensor, sin: Tensor, sign: float) -> Tensor:
    """Rotate the leading ``2 HALF`` channels of every head of x [N, S, H, D] by the position's angles (``sign`` = -1: by the opposite angle, the backward)."""
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    x = x if _view_ok(x, 16 // x.element_size()) else x.contiguous()       # a gradient may arrive in any layout
    ext().rope_apply(_readable(x), _readable(cos), _readable(sin), y, sign)
    return y


class _RoPE3D(torch.autograd.Function):
    """RoPE is a rotation: its backward is the same kernel with the angle negated (nothing saved but the tables)."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rope_rotate(x, cos, sin, 1.0)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return _rope_rotate(dy, cos, sin, -1.0), None, None


def rope_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate x [N, S, H, D] by ``cos`` / ``sin`` [N or 1, S, HALF], autograd-aware.  The caller checked :func:`supports_rope`."""
    return _RoPE3D.apply(x, cos, sin)
