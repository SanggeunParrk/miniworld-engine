# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/augmented_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels import kernels
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.primitives import LayerNorm, Linear

from ..adaptive_layernorm.module import AdaptiveLayerNorm
from ..ops import sigmoid_gate


class AugmentedAttentionPairBias(nn.Module):
    """Augmented attention with pair bias and adaptive conditioning.

    Parameters
    ----------
    d_single : int
        Dimension of single representation.
    d_cond : int
        Dimension of conditioning representation.
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of attention heads.
    use_qk_norm : bool
        Whether to apply RMSNorm to query and key projections.
    implementation : ImplementationType
        Implementation to use.

    """

    def __init__(
        self,
        d_single: int,
        d_cond: int,
        d_pair: int,
        n_head: int,
        *,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.use_qk_norm = use_qk_norm
        self.implementation = implementation

        d_hidden = d_single // n_head

        self.ada_ln_in = AdaptiveLayerNorm(d_single, d_cond)
        self.to_query = Linear(d_single, d_hidden * n_head, bias=True)
        self.to_key = Linear(d_single, d_hidden * n_head, bias=False)
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False)

        if use_qk_norm:
            self.norm_query = nn.RMSNorm(d_hidden)
            self.norm_key = nn.RMSNorm(d_hidden)

        self.ln_pair = LayerNorm(d_pair)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="zero")
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")

        self.to_scale = Linear(d_cond, d_single, bias=True, init="default")
        self.to_scale.bias.data.fill_(-2.0)

    def _kernel_attention_pair_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.implementation == ImplementationType.PYTORCH:
            query.mul_(query.shape[-1] ** -0.5)
            attention = torch.einsum("abihd,abjhd->abhij", query, key)
            bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, L, L)
            attention = attention + bias[None]
            if mask is not None:
                attention = attention.masked_fill(
                    ~mask[:, :, None, None, :],
                    float("-inf"),
                )
            attention = F.softmax(attention, dim=-1)
            return torch.einsum("abhij,abjhd->abihd", attention, value)

        if self.implementation == ImplementationType.TRITON:
            return kernels.triton_augmented_attention_pair_bias(
                query,
                key,
                value,
                bias,
                mask,
            )

        raise InvalidImplementationError(self.implementation)

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Forward pass."""
        single = self.ada_ln_in(single, cond)
        pair = self.ln_pair(pair)
        query = self.to_query(single)
        key = self.to_key(single)
        value = self.to_value(single)
        gate = self.to_gate(single)
        bias = self.to_bias(pair)

        num_aug, batch, len_res = query.shape[:3]
        n_head, hidden_dim = self.n_head, query.shape[-1] // self.n_head

        query, key, value = [
            x.view(num_aug, batch, len_res, n_head, hidden_dim)
            for x in (query, key, value)
        ]

        if mask is not None and mask.ndim == 2:  # noqa: PLR2004
            mask = repeat(mask, "B L -> A B L", A=single.shape[0])

        if self.use_qk_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        out = self._kernel_attention_pair_bias(query, key, value, bias, mask)
        out = rearrange(out, "A B L H D -> A B L (H D)")

        out = sigmoid_gate(gate, out)
        out = self.to_out(out)
        return sigmoid_gate(self.to_scale(cond), out)
