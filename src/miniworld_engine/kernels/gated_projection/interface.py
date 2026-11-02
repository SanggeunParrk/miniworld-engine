"""Public entry point for the gated-projection family.

Gated projection is ``sigmoid(gate) * x`` followed by the output projection, fused so the
gated activation never round-trips through HBM; the exported name is the autograd-aware
entry (fwd+bwd), not the ``torch.autograd.Function`` behind it.

``sigmoid_gate_fused`` is the standalone one-pass gate: the autograd-aware entry over the
``_sigmul_fwd``/``_sigmul_bwd`` kernels this family owns. ``bias_only_attention`` re-exports
it, because its ``dispatch.py`` is what chooses between it and ``fused_gate_out``.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.gated_projection.triton.main import (
    sigmoid_gate_fused,
    triton_gated_projection,
)

__all__ = ["sigmoid_gate_fused", "triton_gated_projection"]


class _GatedResidual(torch.autograd.Function):
    """``x + gate * branch``: one ``addcmul`` kernel forward; backward the two products (autograd's ``addcmul`` backward runs four kernels)."""

    @staticmethod
    def forward(ctx, x, gate, branch):  # noqa: D102
        ctx.save_for_backward(gate, branch)
        return torch.addcmul(x, gate, branch)

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        gate, branch = ctx.saved_tensors
        return g, (g * branch if ctx.needs_input_grad[1] else None), (g * gate if ctx.needs_input_grad[2] else None)


def _addcmul_residual(x, gate, branch):
    """``x + gate * branch`` as one fused elementwise kernel (``addcmul``: one launch, one rounding) instead of a multiply and an add."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, gate, branch)):
        return _GatedResidual.apply(x, gate, branch)
    return torch.addcmul(x, gate, branch)


def gated_residual(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """``x + gate * branch``, routed per card -- see ``kernels.gated_projection.triton.residual.gated_residual`` (op ``gated_residual``:
    ``gated_projection_fwd_residual_triton`` and its backward). H100 / B200 launch no Triton: one fused ``addcmul`` kernel (the PyTorch statement
    for mixed dtypes). On a card without their record: the same PyTorch statement (the kernel rounds the product to the activation dtype before
    the add, as eager does), with one warning."""
    from miniworld_engine.kernels._device import no_triton
    from miniworld_engine.modules.dispatch import portable_blocked

    if no_triton(x) or (x.is_cuda and portable_blocked("gated_residual", x.device, x, gate, branch)):
        if x.shape != gate.shape or x.shape != branch.shape:
            raise ValueError("gated_residual requires identical shapes")
        if no_triton(x) and x.dtype == gate.dtype == branch.dtype:
            return _addcmul_residual(x, gate, branch)   # one fused elementwise kernel
        return x + gate * branch
    from miniworld_engine.kernels.gated_projection.triton.residual import (
        gated_residual as _gated_residual,
    )

    return _gated_residual(x, gate, branch)
