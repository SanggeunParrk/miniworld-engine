# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/triangle_updates.py
"""Triangle (gated self-)attention — model-level op connecting the fused
triangle-attention kernel (and a cuequivariance baseline)."""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from cuequivariance_torch import triangle_attention
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_kernels import kernels
from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.ops import sigmoid_gate
from miniworld_kernels.modules.primitives import LayerNorm, Linear


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


class TriangleAttention(nn.Module):
    """Unified implementation of triangular gated self-attention.

    Parameters
    ----------
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of attention heads.
    d_hidden : int | None
        Total hidden dimension for QKV projections.  Defaults to ``d_pair`` when *None*.
        Must be divisible by ``n_head``.
    starting : bool
        Whether the attention is around the "starting" node.
    use_self_attention : bool
        Whether to use self-attention.
    use_qk_norm : bool
        Whether to apply RMSNorm to query and key projections.
    implementation : ImplementationType
        Implementation to use.

    """

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        use_self_attention: bool = True,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.starting = starting
        self.use_self_attention = use_self_attention
        self.use_qk_norm = use_qk_norm
        self.implementation = implementation
        position = "starting" if starting else "ending"
        self.nvtx_enabled = False
        self.nvtx_name = f"triangle_attention/{position}"

        if d_hidden is None:
            d_hidden = d_pair

        if d_hidden % n_head != 0:
            msg = f"d_hidden ({d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        self.ln_pair = LayerNorm(d_pair)
        if use_self_attention:
            self.to_query = Linear(d_pair, d_hidden, bias=False, init="glorot")
            self.to_key = Linear(d_pair, d_hidden, bias=False, init="glorot")

            if use_qk_norm:
                d_head = d_hidden // n_head
                self.norm_query = nn.RMSNorm(d_head)
                self.norm_key = nn.RMSNorm(d_head)

        self.to_value = Linear(d_pair, d_hidden, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d_hidden, bias=False, init="gating")
        self.to_out = Linear(d_hidden, d_pair, bias=False, init="zero")

    def _kernel_triangle_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if self.implementation == ImplementationType.PYTORCH:
            query.mul_(query.shape[-1] ** -0.5)
            attention = torch.einsum("bhijd,bhikd->bhijk", query, key)
            attention = attention + bias[:, :, None, :, :]
            attention = F.softmax(attention, dim=-1)
            return torch.einsum("bhijk,bhikd->bhijd", attention, value)

        if self.implementation == ImplementationType.TRITON:
            return kernels.triton_triangle_attention_pair_bias(
                query,
                key,
                value,
                bias,
            )

        if self.implementation == ImplementationType.CUEQUIVARIANCE:
            q = rearrange(query, "B H L1 L2 D -> B L1 H L2 D")
            k = rearrange(key, "B H L1 L2 D -> B L1 H L2 D")
            v = rearrange(value, "B H L1 L2 D -> B L1 H L2 D")
            out = triangle_attention(q, k, v, bias.unsqueeze(1).float())
            return rearrange(out, "B L1 H L2 D -> B H L1 L2 D")  # ty: ignore[invalid-return-type]

        raise InvalidImplementationError(self.implementation)

    def _kernel_bias_only_attention(
        self,
        value: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        attention = F.softmax(bias, dim=-1)
        return torch.einsum("bhjk,bhikd->bhijd", attention, value)

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
            if not self.starting:
                pair = rearrange(pair, "B I J D -> B J I D").contiguous()
            assert pair.is_contiguous()
            pair = self.ln_pair(pair)
            value = self.to_value(pair)
            bias = self.to_bias(pair)

            value = rearrange(
                value, "B L L2 (H D) -> B H L L2 D", H=self.n_head
            ).contiguous()
            bias = rearrange(bias, "B L L2 H -> B H L L2").contiguous()
            if mask is not None:
                bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))

            if self.use_self_attention:
                query = self.to_query(pair)
                key = self.to_key(pair)

                query = rearrange(
                    query, "B L L2 (H D) -> B H L L2 D", H=self.n_head
                ).contiguous()
                key = rearrange(
                    key, "B L L2 (H D) -> B H L L2 D", H=self.n_head
                ).contiguous()

                if self.use_qk_norm:
                    query = self.norm_query(query)
                    key = self.norm_key(key)

                out = self._kernel_triangle_attention(query, key, value, bias)
            else:
                out = self._kernel_bias_only_attention(value, bias)

            out = rearrange(out, "B H L L2 D -> B L L2 (H D)").contiguous()
            out = sigmoid_gate(self.to_gate(pair), out)
            out = self.to_out(out)

            if not self.starting:
                out = rearrange(out, "B J I D -> B I J D").contiguous()
            return out


class TrianglePairAttention(nn.Module):
    """Triangular gated self-attention with a pair-attention contraction."""

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        use_self_attention: bool = True,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.starting = starting
        self.use_self_attention = use_self_attention
        self.use_qk_norm = use_qk_norm
        self.implementation = implementation
        position = "starting" if starting else "ending"
        self.nvtx_enabled = False
        self.nvtx_name = f"triangle_attention/{position}"

        if d_hidden is None:
            d_hidden = d_pair

        if d_hidden % n_head != 0:
            msg = f"d_hidden ({d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        self.ln_pair = LayerNorm(d_pair)

        self.to_value = Linear(d_pair, d_hidden, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d_hidden, bias=False, init="gating")
        self.to_out = Linear(d_hidden, d_pair, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        if not self.starting:
            pair = pair.transpose(1, 2).contiguous()
        pair = self.ln_pair(pair)
        value = self.to_value(pair)
        bias = self.to_bias(pair)

        value = value.view(
            value.shape[0], value.shape[1], value.shape[2], self.n_head, -1
        )
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, :, None], float("-inf"))

        attention = F.softmax(bias, dim=-2)
        out = torch.einsum("bjkh,bikhd->bijhd", attention, value).contiguous()
        out = out.view(out.shape[0], out.shape[1], out.shape[2], -1)

        out = sigmoid_gate(self.to_gate(pair), out)
        out = self.to_out(out)

        if not self.starting:
            out = out.transpose(1, 2).contiguous()
        return out
