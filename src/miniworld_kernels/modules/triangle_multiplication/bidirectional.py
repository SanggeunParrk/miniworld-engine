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
        if self.implementation == ImplementationType.CUTE:
            return self._forward_cute(pair, mask)
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

    def _forward_cute(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CUTE bidirectional path: compose the single-direction tm1 ``bdll_sm100``
        gate-GEMM+einsum for BOTH directions (outgoing on the first ``d_hidden``
        channels, incoming on the second), then the SHARED ln_out(2h) + to_out + gate
        back. Avoids the broken quack gated-M-major front of trimul_inproj/cute.

        incoming = outgoing with the k<->contraction index flipped, handled directly
        by the incoming einsum (no input transpose needed since we control the einsum).
        Same math as the pytorch reference; bf16 in / fp32 acc / bf16 out.
        """
        from miniworld_kernels.modules.triangle_multiplication.module import _load_cute_fns

        tm1_cute_forward, _tm2, fused_ln_mask, layer_norm_transpose = _load_cute_fns()
        b, l1, l2, d = pair.shape
        h = self.d_hidden
        M = b * l1 * l2

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            x = fused_ln_mask(pair, self.ln_pair.weight, self.ln_pair.bias, mask_2d)
        else:
            o = layer_norm_transpose(
                pair.reshape(M, d), self.ln_pair.weight, self.ln_pair.bias,
                eps=self.ln_pair.eps, layout="nd->nd")
            x = (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

        def _front(sl: slice):
            return tm1_cute_forward(
                x,
                self.to_left.weight[sl].T.contiguous(),
                self.to_left_gate.weight[sl].T.contiguous(),
                self.to_right.weight[sl].T.contiguous(),
                self.to_right_gate.weight[sl].T.contiguous(),
                out_layout="bdll_sm100",
            )

        left_out, right_out = _front(slice(0, h))          # outgoing half, [B,h,L,L]
        left_in, right_in = _front(slice(h, 2 * h))        # incoming half, [B,h,L,L]
        out_o = torch.einsum("bdik,bdjk->bdij", left_out, right_out)   # outgoing
        out_i = torch.einsum("bdki,bdkj->bdij", left_in, right_in)     # incoming
        tri = torch.cat([out_o, out_i], dim=1)             # [B, 2h, L, L]

        tri_dbn = tri.permute(1, 0, 2, 3).reshape(2 * h, b, l1 * l2)
        oo = layer_norm_transpose(
            tri_dbn, self.ln_out.weight, self.ln_out.bias,
            eps=self.ln_out.eps, layout="dbn->bnd")
        out_normed = (oo[0] if isinstance(oo, tuple) else oo).view(b, l1, l2, 2 * h)

        # shared back: sigmoid(x @ to_gate.T) * (out_normed @ to_out.T)  (gate K=d, out K=2h)
        gate = torch.sigmoid(x.reshape(M, d) @ self.to_gate.weight.T)
        proj = out_normed.reshape(M, 2 * h) @ self.to_out.weight.T
        return (gate * proj).view(b, l1, l2, d)
