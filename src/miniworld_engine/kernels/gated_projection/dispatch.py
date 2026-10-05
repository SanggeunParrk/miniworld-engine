"""Which implementation serves the gated projections: the A100 hand-CUDA kernels (``cuda/sm80.py``) where their gate takes the call, the Triton kernels otherwise.

``fused_gate_out`` (``(sigmoid(gate) * out_r) @ wo^T``, the registry's ``gated_linear``), ``sigmoid_gate_fused`` (the one-pass gate) and ``gated_residual`` keep their names and
contracts; ``MINIWORLD_GATED_SM80=0`` or an engine backend forced to Triton keeps the Triton kernels.
"""

from __future__ import annotations

import torch


def fused_gate_out(gate: torch.Tensor, out_r: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """sigmoid(gate) * out_r, then @ wo^T. gate / out_r [..., DH], wo [N, DH] -> [..., N]."""
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    if gate.shape == out_r.shape and out_r.dtype is gate.dtype and wo.dtype is gate.dtype and sm80.serves(gate.shape[-1], wo.shape[0], gate.device, gate.dtype):
        return sm80.fused_gate_out(gate, out_r, wo)

    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import (
        fused_gate_out as triton_fused_gate_out,
    )

    return triton_fused_gate_out(gate, out_r, wo)


def sigmoid_gate_fused(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """sigmoid(gate) * out in ONE pass (same shape)."""
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    if sm80.serves_elementwise(gate, out):
        return sm80.sigmoid_gate_fused(gate, out)

    from miniworld_engine.kernels.gated_projection.triton.main import (
        sigmoid_gate_fused as triton_sigmoid_gate_fused,
    )

    return triton_sigmoid_gate_fused(gate, out)


def gated_residual(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """Same-shape, same-dtype residual gate ``x + gate * branch``; no sigmoid, normalization, or projection."""
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    if x.shape == gate.shape == branch.shape and sm80.serves_elementwise(x, gate, branch):
        return sm80.gated_residual(x, gate, branch)

    from miniworld_engine.kernels.gated_projection.triton.residual import (
        gated_residual as triton_gated_residual,
    )

    return triton_gated_residual(x, gate, branch)
