"""Bidirectional triangular multiplicative update — outgoing + incoming in one block.

A single module shares one input LayerNorm and projects the pair to ``2 * d_hidden``
channels; the hidden channels split in half — first ``d_hidden`` compute the
**outgoing** product (``bikd,bjkd->bijd``), the second ``d_hidden`` the **incoming**
product (``bkid,bkjd->bijd``). The two are concatenated to ``2 * d_hidden`` and
projected down to ``d_pair``.

PYTORCH is the reference. The fused path (CUTE) reuses the trimul_inproj pipeline
with bidirectional dims: one wider gated GEMM front (left/right each ``2*d_hidden``),
two einsums (outgoing on the first half, incoming on the second), then the split
back (cute LayerNormLinear over ``2*d_hidden`` + triton GateElem). See
``kernels/trimul_inproj/cute/bidirectional.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.ops import sigmoid_gate
from miniworld_kernels.modules.primitives import LayerNorm, Linear


class BidirectionalTriangleMultiplication(nn.Module):
    """Triangular multiplicative update computing outgoing+incoming in one block."""

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int | None = None,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.implementation = implementation
        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair
        d2 = 2 * self.d_hidden

        self.ln_pair = LayerNorm(d_pair, implementation=implementation)
        # Doubled-width left/right projections: [outgoing | incoming] channels.
        self.to_left = Linear(d_pair, d2, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d2, bias=False, init="zero")
        self.to_right = Linear(d_pair, d2, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d2, bias=False, init="zero")

        self.ln_out = LayerNorm(d2, implementation=implementation)
        self.to_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_out = Linear(d2, d_pair, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        if self.implementation != ImplementationType.PYTORCH:
            raise InvalidImplementationError(self.implementation)

        pair = self.ln_pair(pair)
        left = sigmoid_gate(self.to_left_gate(pair), self.to_left(pair))
        right = sigmoid_gate(self.to_right_gate(pair), self.to_right(pair))

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            left = left * mask_2d[..., None]
            right = right * mask_2d[..., None]

        # Split hidden channels: first half -> outgoing, second half -> incoming.
        h = self.d_hidden
        left_out, left_in = left[..., :h], left[..., h:]
        right_out, right_in = right[..., :h], right[..., h:]

        out_outgoing = torch.einsum("bikd,bjkd->bijd", left_out, right_out)
        out_incoming = torch.einsum("bkid,bkjd->bijd", left_in, right_in)
        out = torch.cat([out_outgoing, out_incoming], dim=-1)

        out = self.ln_out(out)
        return sigmoid_gate(self.to_gate(pair), self.to_out(out))
