"""Sigmoid gate and output linear, using the engine's measured dispatch policy."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gated_linear(
    gate: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Linear(sigmoid(gate) * value); weights use (out, in) layout.

    This op has no residual and no conditioning gate after the output projection.
    Callers retain those model-specific operations. Unsupported dtypes/broadcasts
    preserve the PyTorch equation rather than quantizing the caller's tensors.
    """
    if (
        not value.is_cuda
        or value.dtype != torch.bfloat16
        or gate.dtype != value.dtype
        or weight.dtype != value.dtype
        or gate.shape != value.shape
    ):
        return F.linear(torch.sigmoid(gate) * value, weight, bias)
    shape = value.shape
    # Released AF3 exposes an unbatched [L,D] token stream. Restore its semantic
    # batch before the existing primitives compute their pre-flatten shape key.
    if value.ndim == 2:
        gate, value = gate.unsqueeze(0), value.unsqueeze(0)
    from miniworld_engine.kernels.bias_only_attention.dispatch import gate_use_fused
    from miniworld_engine.kernels.bias_only_attention.interface import (
        fused_gate_out,
        sigmoid_gate_fused,
    )

    if gate_use_fused(
        value.shape[-1],
        weight.shape[0],
        math.prod(value.shape[:-1]),
        value.device,
        value.dtype,
    ):
        out = fused_gate_out(gate, value, weight)
    else:
        out = F.linear(sigmoid_gate_fused(gate, value), weight)
    out = out.reshape(*shape[:-1], weight.shape[0])
    return out if bias is None else out + bias
