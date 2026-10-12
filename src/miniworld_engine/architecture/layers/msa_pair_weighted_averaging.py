import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine.architecture.checkpoints.backend_attention import gated_projection
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.primitives import LayerNorm, Linear
from miniworld_engine.architecture.typing import typecheck


class MSAPairWeightedAveraging(nn.Module):
    """MSA pair-weighted averaging.

    Parameters
    ----------
    d_msa : int
        Dimension of MSA representation.
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of heads.
    d_hidden : int, optional
        Per-head hidden dimension. Defaults to ``d_msa // n_head``.

    """

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        n_head: int = 8,
        d_hidden: int | None = None,
        *,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.foldforge_implementation = ImplementationType(implementation)

        if d_hidden is None:
            d_hidden = d_msa // n_head

        # Fused miniworld_engine LN (bf16) under MINIWORLD_ENGINE — raw nn.LayerNorm is fp32
        # native under autocast and dominates the MSA module at depth 2048.
        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        self.to_value = Linear(d_msa, d_hidden * n_head, bias=False, init="glorot")

        self.ln_pair = LayerNorm(d_pair, implementation=implementation)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")

        self.to_gate = Linear(d_msa, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_msa, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B M L d_msa"]:
        """Forward pass."""
        msa = self.ln_msa(msa)
        value = self.to_value(msa)
        value = rearrange(value, "B M L (H D) -> B M L H D", H=self.n_head)

        pair = self.ln_pair(pair)
        bias = self.to_bias(pair)
        bias = rearrange(bias, "B L1 L2 H -> B H L1 L2")

        if mask is not None:
            bias = bias.masked_fill(
                ~mask[:, None, None, :], torch.finfo(bias.dtype).min
            )

        attention = F.softmax(bias, dim=-1)
        out = torch.einsum("bhij,bmjhd->bmihd", attention, value)

        gate = self.to_gate(msa)
        out = rearrange(out, "B M L H D -> B M L (H D)")
        return gated_projection(self, self.to_out, gate, out)
