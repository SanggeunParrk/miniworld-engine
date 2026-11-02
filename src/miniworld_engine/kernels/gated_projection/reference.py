"""PyTorch reference for the fused gated-projection kernel.

Mirrors ``triton_gated_projection`` (``TritonGatedProjectionFunction``), which
takes ``(gate, x, out_weight)`` in that order and returns one tensor.

Formula::

    y = (sigmoid(gate) * x) @ out_weight

with ``gate``/``x`` of shape ``(..., hd)`` and ``out_weight`` of shape
``(hd, d)`` -- a right-multiplied matrix, not an ``nn.Linear``-style
``(out, in)`` weight -- so ``y`` has shape ``(..., d)``.

The reference is the plain functional form (:func:`gated_projection_pytorch`): a kernel is checked against it on both the forward output and the backward gradients.

Note that the kernel's ``backward`` returns fp32 gradients regardless of the
input dtype; autograd through this reference produces gradients in the input
dtype, so a gradient comparison should cast one side.
"""

from __future__ import annotations

import torch


def gated_projection_pytorch(
    gate: torch.Tensor,  # (..., hd)
    x: torch.Tensor,  # (..., hd)
    out_weight: torch.Tensor,  # (hd, d)
) -> torch.Tensor:
    """Compute ``(sigmoid(gate) * x) @ out_weight``."""
    op_dtype = x.dtype
    # The kernel casts gate to x's dtype before the elementwise pass, evaluates the
    # sigmoid and the product in fp32, then rounds the gated tensor back to x's dtype
    # before the GEMM. In fp32 all three casts are no-ops.
    gate = gate.to(op_dtype)
    gated = (x.float() * torch.sigmoid(gate.float())).to(op_dtype)
    return gated @ out_weight.to(op_dtype)
