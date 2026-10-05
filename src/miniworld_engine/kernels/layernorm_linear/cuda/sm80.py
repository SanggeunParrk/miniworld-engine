"""A100 (sm_80) hand-CUDA fused LayerNorm + projection to a few outputs: ``ops.layer_norm_linear`` (the pair-bias projection ``LN(pair) -> n_head``, the atom-pair windows
``LN(pair) -> 4 / 12 heads`` and the atom coordinate projection ``LN(single) -> 3``), inference and training.

    out[..., h] = sum_c ((x[..., c] - mean) rstd gamma_c) W[h, c]          (no LayerNorm beta, no bias; ``W`` [n_head, d], 1 .. 16 outputs, d = 16 or 64 .. 512)

``sm80/lnl_sm80.cuh``: the forward is one warp per tile of 16 rows with the rows read straight into the ``mma.sync`` A fragments (the LayerNorm on the registers, bf16(n) as the
operand, fp32 accumulation, bf16 output; training also saves the statistics and the fp32 result ``u``); the backward is slice-local given the saved statistics (``dxn = dout W`` on the
tensor cores, ``dx`` from the two row sums ``dout . (W gamma)`` and ``dout . u``, the weight and scale gradients from ``Gn = dout^T xhat`` accumulated on the tensor cores and summed in
a fixed order): one pass over ``x`` in each direction.  ``MINIWORLD_LNLINEAR_SM80=0`` keeps the Triton kernels.
"""

import functools
import os
import warnings
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"
BF = torch.bfloat16
F32 = torch.float32
#: the widths with a kernel instantiation (d_norm of the registry rows: 16, 64, 128, 256, 384; 512 is the benchmark width)
WIDTHS = (16, 64, 128, 256, 384, 512)
MAX_OUT = 16


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="layernorm_linear_sm80",
        sources=[str(_dir / "lnl_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


_BUILD_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the extension; False, with one warning, when the toolchain fails (the Triton kernels then serve).  A process-level constant: ``torch.compile``
    evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _BUILD_FAILED
    if _BUILD_FAILED:
        return False
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"sm80 layernorm_linear unavailable, keeping the Triton kernels: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


@functools.lru_cache(maxsize=8)
def _is_ampere(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (8, 0)


def serves(x: torch.Tensor, ln_weight: torch.Tensor, proj_weight: torch.Tensor) -> bool:
    """The path's whole gate: sm_80, bf16 activations ``[..., d]`` (d = 16 or 64 .. 512, see ``WIDTHS``), a bf16 projection weight ``[n_head, d]`` with 1 .. 16 rows, a LayerNorm scale
    ``[d]`` in fp32 or bf16, fewer than 2^31 rows, the engine's backend not forced to Triton, ``MINIWORLD_LNLINEAR_SM80`` not 0."""
    if os.environ.get("MINIWORLD_LNLINEAR_SM80", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    if not x.is_cuda or x.dtype is not BF or x.ndim < 2 or x.numel() == 0:
        return False
    d = x.shape[-1]
    if d not in WIDTHS or proj_weight.ndim != 2 or proj_weight.shape[1] != d or not 1 <= proj_weight.shape[0] <= MAX_OUT or proj_weight.dtype is not BF:
        return False
    if ln_weight.shape != (d,) or ln_weight.dtype not in (F32, BF) or ln_weight.device != x.device or proj_weight.device != x.device:
        return False
    if x.numel() // d >= 2**31:
        return False
    return _is_ampere(x.device.index if x.device.index is not None else torch.cuda.current_device()) and _loads()


def _forward_fake(x, ln_w, w, eps, save):
    """[out [M, n_head] bf16, stats [M, 2] fp32 and u [M, n_head] fp32 when ``save`` (else empty)]."""
    m, nh = x.shape[0], w.shape[0]
    return [x.new_empty((m, nh)), x.new_empty((m, 2), dtype=F32) if save else x.new_empty((0,), dtype=F32),
            x.new_empty((m, nh), dtype=F32) if save else x.new_empty((0,), dtype=F32)]


@opaque(fake=_forward_fake, name="layernorm_linear_sm80_forward")
def _forward(x: torch.Tensor, ln_w: torch.Tensor, w: torch.Tensor, eps: float, save: bool) -> list[torch.Tensor]:
    """The kernel launch: ``x`` ``[M, d]`` bf16 contiguous, ``ln_w`` ``[d]`` fp32, ``w`` ``[n_head, d]`` bf16; returns [out, stats, u] (the last two empty unless ``save``)."""
    m, nh = x.shape[0], w.shape[0]
    out = torch.empty((m, nh), dtype=BF, device=x.device)
    stats = torch.empty((m, 2), dtype=F32, device=x.device) if save else x.new_empty((0,), dtype=F32)
    u = torch.empty((m, nh), dtype=F32, device=x.device) if save else x.new_empty((0,), dtype=F32)
    _ext().lnl_forward(x, w, ln_w, out, stats, u, float(eps))
    return [out, stats, u]


def _backward_fake(x, ln_w, w, dout, stats, u, ln_bf16):
    """[dx [M, d] bf16, dw [n_head, d] bf16, dgamma [d] fp32 (bf16 when ``ln_bf16``)]."""
    return [x.new_empty(x.shape), w.new_empty(w.shape), x.new_empty((x.shape[1],), dtype=BF if ln_bf16 else F32)]


@opaque(fake=_backward_fake, name="layernorm_linear_sm80_backward")
def _backward(x: torch.Tensor, ln_w: torch.Tensor, w: torch.Tensor, dout: torch.Tensor, stats: torch.Tensor, u: torch.Tensor, ln_bf16: bool) -> list[torch.Tensor]:
    """The kernel launches (one pass over ``x`` plus a fixed-order reduction of the weight gradients): returns [dx, dw, dgamma]."""
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    dgamma = torch.empty((x.shape[1],), dtype=BF if ln_bf16 else F32, device=x.device)
    _ext().lnl_backward(x, w, ln_w, dout, stats, u, dx, dw, dgamma, int(ln_bf16))
    return [dx, dw, dgamma]


def _scale32(ln_weight: torch.Tensor) -> torch.Tensor:
    return ln_weight.detach().float().contiguous()


class _LayerNormLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, ln_weight, proj_weight, eps):
        out, stats, u = _forward(x2, _scale32(ln_weight), proj_weight.contiguous(), eps, True)
        ctx.save_for_backward(x2, ln_weight, proj_weight, stats, u)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        x2, ln_weight, proj_weight, stats, u = ctx.saved_tensors
        dx, dw, dgamma = _backward(x2, _scale32(ln_weight), proj_weight.contiguous(), dout.reshape(x2.shape[0], proj_weight.shape[0]).to(BF).contiguous(), stats, u,
                                   ln_weight.dtype is BF)
        return dx, dgamma, dw, None


def layer_norm_linear(x: torch.Tensor, ln_weight: torch.Tensor, proj_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """``Linear(LayerNorm(x))`` over the last dim (LayerNorm scale only, no bias): ``x`` ``[..., d]`` bf16, ``ln_weight`` ``[d]``, ``proj_weight`` ``[n_head, d]``; returns
    ``[..., n_head]`` bf16.  Autograd-aware (``dx``, the scale's and the projection's gradients).  Call ``serves()`` first."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    grad = torch.is_grad_enabled() and (x.requires_grad or ln_weight.requires_grad or proj_weight.requires_grad)
    out = _LayerNormLinear.apply(x2, ln_weight, proj_weight, float(eps)) if grad else _forward(x2, _scale32(ln_weight), proj_weight.contiguous(), float(eps), False)[0]
    return out.reshape(*shape[:-1], proj_weight.shape[0])
