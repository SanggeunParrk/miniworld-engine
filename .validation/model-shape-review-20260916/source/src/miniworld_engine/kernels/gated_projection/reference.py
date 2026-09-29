"""PyTorch reference for the fused gated-projection kernel.

Mirrors ``triton_gated_projection`` (``TritonGatedProjectionFunction``), which
takes ``(gate, x, out_weight)`` in that order and returns one tensor.

Formula::

    y = (sigmoid(gate) * x) @ out_weight

with ``gate``/``x`` of shape ``(..., hd)`` and ``out_weight`` of shape
``(hd, d)`` -- a right-multiplied matrix, not an ``nn.Linear``-style
``(out, in)`` weight -- so ``y`` has shape ``(..., d)``.

Provided as an ``nn.Module`` (:class:`GatedProjectionReference`, owning
``out_weight``) so a kernel can be checked against it on both the forward output
and the backward gradients, e.g.::

    ref = GatedProjectionReference(hd, d).cuda().to(torch.bfloat16)
    y = ref(gate, x)
    yk = triton_gated_projection(gate, x, ref.out_weight)
    y.sum().backward()

A plain functional form (:func:`gated_projection_pytorch`) is kept for callers
that already hold the weight tensor.

Note that the kernel's ``backward`` returns fp32 gradients regardless of the
input dtype; autograd through this reference produces gradients in the input
dtype, so a gradient comparison should cast one side.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


class GatedProjectionReference(nn.Module):
    """nn.Module reference for gated projection (forward + backward ground truth)."""

    def __init__(
        self,
        d_hidden: int,
        d_out: int,
        *,
        device=None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        scale = d_hidden**-0.5
        self.out_weight = nn.Parameter(
            torch.randn(d_hidden, d_out, device=device, dtype=dtype) * scale
        )

    def forward(self, gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; differentiable in ``gate``, ``x`` and ``out_weight``."""
        return gated_projection_pytorch(gate, x, self.out_weight)
