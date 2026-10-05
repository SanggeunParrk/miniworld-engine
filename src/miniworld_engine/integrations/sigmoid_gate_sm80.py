"""The A100 (sm_80) sigmoid gate ``sigmoid(gate) * x`` -- forward and backward as one hand-CUDA pass each (``kernels/augmented_attention/cuda/sm80/glue_sm80.cuh``) -- for the
modules that gate an attention output (``SWA3DRoPEAttention``: the output gate before ``out_proj``).

``serves()`` is the whole gate: A100 (sm_80), the engine's kernel backend not forced to Triton, ``gate`` and ``x`` of one shape and one dtype (bf16 or fp32), a last dimension
that is a multiple of 8. ``MINIWORLD_SIGMOID_GATE_SM80=0`` turns it off (the module then takes the Triton ``sigmoid_gate_fused``). A failed extension build warns once and keeps
the old path. One autograd Function whose forward and backward are each one opaque op (``kernels/_compile.opaque``): no Triton, no Inductor kernel, no eager chain.
"""

from __future__ import annotations

import os
import warnings

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

_FAILED = False


def _sm80():
    from miniworld_engine.kernels.augmented_attention.cuda import sm80
    return sm80


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the gate extension; False, with one warning, when the toolchain fails (the caller then keeps its old path).  A process-level constant, so
    ``torch.compile`` evaluates it once at trace time."""
    global _FAILED
    if _FAILED:
        return False
    try:
        _sm80()._glue_ext()
    except Exception as exc:  # a toolchain problem keeps the old path
        _FAILED = True
        warnings.warn(f"sm_80 sigmoid-gate kernels unavailable, keeping the Triton gate: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def serves(gate: torch.Tensor, x: torch.Tensor) -> bool:
    if os.environ.get("MINIWORLD_SIGMOID_GATE_SM80", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    if not gate.is_cuda or gate.dtype not in (torch.bfloat16, torch.float32) or x.dtype is not gate.dtype or gate.shape != x.shape or gate.ndim < 2 or gate.shape[-1] % 8:
        return False
    if torch.cuda.get_device_capability(gate.device) != (8, 0):
        return False
    return _loads()


def _fwd_fake(gate, x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@opaque(fake=_fwd_fake, name="sigmoid_gate_sm80_fwd")
def _fwd(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``sigmoid(gate) x`` over [..., d] tensors (the same dtype and shape), a fresh contiguous tensor."""
    d = x.shape[-1]
    g2, x2 = gate.reshape(-1, d), x.reshape(-1, d)
    out = torch.empty_like(x2)
    _sm80().gate_rows(x2, g2, out)
    return out.view(x.shape)


def _bwd_fake(dout, gate, x):
    return [torch.empty_like(gate, memory_format=torch.contiguous_format), torch.empty_like(x, memory_format=torch.contiguous_format)]


@opaque(fake=_bwd_fake, name="sigmoid_gate_sm80_bwd")
def _bwd(dout: torch.Tensor, gate: torch.Tensor, x: torch.Tensor) -> list[torch.Tensor]:
    """``[d gate, d x]`` of ``sigmoid(gate) x``: ``dx = dout sigmoid(gate)``, ``d gate = dout x sigmoid(gate) (1 - sigmoid(gate))``."""
    d = x.shape[-1]
    dg = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _sm80().gate_bwd(dout.reshape(-1, d), x.reshape(-1, d), gate.reshape(-1, d), dx.view(-1, d), dg.view(-1, d))
    return [dg, dx]


class _SigmoidGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, x):
        ctx.save_for_backward(gate, x)
        return _fwd(gate, x)

    @staticmethod
    def backward(ctx, dout):
        gate, x = ctx.saved_tensors
        dg, dx = _bwd(dout.contiguous(), gate, x)
        return dg if ctx.needs_input_grad[0] else None, dx if ctx.needs_input_grad[1] else None


def sigmoid_gate(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``sigmoid(gate) * x`` on the hand-CUDA pass (call ``serves`` first); differentiable in both."""
    if torch.is_grad_enabled() and (gate.requires_grad or x.requires_grad):
        return _SigmoidGate.apply(gate, x)
    return _fwd(gate, x)


__all__ = ["serves", "sigmoid_gate"]
