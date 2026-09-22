import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine import kernels
from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules.dispatch import KernelBackend, resolve_augmented_attention
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear, RMSNorm


class AttentionPairBias(nn.Module):
    """Attention with pair bias.

    Parameters
    ----------
    d_single : int
        Dimension of single representation.
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of attention heads.
    d_hidden : int, optional
        Per-head hidden dimension. Defaults to ``d_single // n_head``.
    use_qk_norm : bool
        Whether to apply RMSNorm to query and key projections.

    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        n_head: int = 8,
        d_hidden: int | None = None,
        *,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.use_qk_norm = use_qk_norm
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_augmented_attention(self.implementation)
        if d_hidden is None:
            if d_single % n_head != 0:
                msg = f"{d_single=} must be divisible by {n_head=}"
                raise ValueError(msg)
            d_hidden = d_single // n_head

        self.ln_single = LayerNorm(d_single, implementation=self.implementation)
        self.to_query = Linear(d_single, d_hidden * n_head, bias=True, init="glorot")
        self.to_key = Linear(d_single, d_hidden * n_head, bias=False, init="glorot")
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False, init="glorot")

        if use_qk_norm:
            self.norm_query = RMSNorm(d_hidden, implementation=self.implementation)
            self.norm_key = RMSNorm(d_hidden, implementation=self.implementation)

        self.ln_pair = LayerNorm(d_pair, implementation=self.implementation)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "B L d_single"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L d_single"]:
        """Forward pass. ALWAYS applies the residual: single + attention_pair_bias(single, pair).
        The residual is UNCONDITIONAL (domain standard) and applied EXPLICITLY here (team-gm layer,
        not kernel-fused). The residual is unconditional and has no flag."""
        single_res = single  # residual == the ORIGINAL input (before ln_single rebinds `single`)
        single = self.ln_single(single)
        query = self.to_query(single)
        key = self.to_key(single)
        value = self.to_value(single)

        query = rearrange(query, "B L (H D) -> B H L D", H=self.n_head)
        key = rearrange(key, "B L (H D) -> B H L D", H=self.n_head)
        value = rearrange(value, "B L (H D) -> B H L D", H=self.n_head)

        if self.use_qk_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        pair = self.ln_pair(pair)
        bias = self.to_bias(pair)
        bias = rearrange(bias, "B L L2 H -> B H L L2")
        if mask is not None:
            bias = bias.masked_fill(
                ~mask[:, None, None, :], torch.finfo(bias.dtype).min
            )
        if self._backend == KernelBackend.PYTORCH:
            out = F.scaled_dot_product_attention(query, key, value, bias)
        elif self._backend == KernelBackend.TRITON:
            # Reuse the existing QK + pair-bias kernel with one augmentation.
            # Bias-only attention is a different operation and cannot replace QK.
            q, k, v = (t.transpose(1, 2).unsqueeze(0) for t in (query, key, value))
            out = kernels.triton_augmented_attention_pair_bias(
                q, k, v, bias.permute(0, 2, 3, 1),
                None if mask is None else mask.unsqueeze(0),
            ).squeeze(0).transpose(1, 2)
        else:
            raise InvalidImplementationError(self.implementation)

        gate = self.to_gate(single)
        out = rearrange(out, "B H L D -> B L (H D)")
        out = sigmoid_gate(gate, out)
        out = self.to_out(out)
        return single_res + out
