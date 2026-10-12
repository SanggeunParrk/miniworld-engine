from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float

from miniworld_engine.architecture.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.architecture.layers.adaln import AdaptiveLayerNorm
from miniworld_engine.architecture.layers.ops import sigmoid_gate
from miniworld_engine.architecture.primitives import Linear, MPLinear
from miniworld_engine.architecture.typing import typecheck


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
    d_hidden : int, optional
        Per-head hidden dimension. Defaults to ``d_single // n_head``.
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
        d_hidden: int | None = None,
        *,
        use_qk_norm: bool = False,
        use_bias_norm: bool = False,
        magnitude_preserving: bool = False,
        use_rotation: bool = False,
        mp_full: bool = False,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
        kernel_type: Literal["memory_efficient", "compute_efficient"] = (
            "compute_efficient"
        ),
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.use_qk_norm = use_qk_norm
        self.use_bias_norm = use_bias_norm
        # mp_full extends MP to every linear in the block (incl. zero/gating-init
        # to_out, to_gate, and AdaLN scale/angle); it implies magnitude_preserving.
        magnitude_preserving = magnitude_preserving or mp_full
        self.magnitude_preserving = magnitude_preserving
        self.mp_full = mp_full
        self.implementation = implementation
        self.kernel_type = kernel_type

        if d_hidden is None:
            d_hidden = d_single // n_head

        # EDM2 magnitude preservation: q/k/v projections use forced weight
        # normalization. They emit tensors consumed by the attention kernel
        # (not raw weights), so this composes with the miniworld_engine attention core.
        dense = MPLinear if magnitude_preserving else Linear
        self.ada_ln_in = AdaptiveLayerNorm(
            d_single, d_cond, use_rotation=use_rotation, mp_full=mp_full
        )
        self.to_query = dense(d_single, d_hidden * n_head, bias=True)
        self.to_key = dense(d_single, d_hidden * n_head, bias=False)
        self.to_value = dense(d_single, d_hidden * n_head, bias=False)

        if use_qk_norm:
            self.norm_query = nn.RMSNorm(d_hidden)
            self.norm_key = nn.RMSNorm(d_hidden)

        self.ln_pair = nn.LayerNorm(d_pair, bias=False)
        # The pair->logit-bias projection is one of the top ||w||-growth sites
        # (its weight is scale-invariant under norm_bias, so its norm drifts
        # freely). Under MP it gets forced weight normalization too; otherwise
        # it stays the original zero-init Linear.
        self.to_bias = (
            MPLinear(d_pair, n_head, bias=False, init="normal")
            if magnitude_preserving
            else Linear(d_pair, n_head, bias=False, init="zero")
        )
        # RMSNorm (affine) over the n_head axis of the pair bias, to bound the
        # additive attention-logit bias (otherwise to_bias / ln_pair.gamma grow
        # unbounded -> attention-logit-growth divergence). Mirrors qk-norm.
        if use_bias_norm:
            self.norm_bias = nn.RMSNorm(n_head)
        # Under mp_full, the gate and output projections are MP too (lose their
        # gating/zero init; magnitude stays controlled via MP + mp_sum residual).
        gate_cls = MPLinear if mp_full else Linear
        out_cls = MPLinear if mp_full else Linear
        self.to_gate = gate_cls(
            d_single,
            d_hidden * n_head,
            bias=False,
            init="normal" if mp_full else "gating",
        )
        self.to_out = out_cls(
            d_hidden * n_head,
            d_single,
            bias=False,
            init="normal" if mp_full else "zero",
        )

        scale_cls = MPLinear if mp_full else Linear
        self.to_scale = scale_cls(
            d_cond, d_single, bias=True, init="normal" if mp_full else "default"
        )
        self.to_scale.bias.data.fill_(-2.0)

    # Inference-time chunking constant. The non-chunked path computes
    #   pair = self.ln_pair(pair)          # [B, L, L, d_pair]  — full tensor
    #   bias = self.to_bias(pair)          # [B, L, L, n_head]  — full tensor
    # which doubles peak memory while bias is being allocated. For
    # L ≈ atom-scale (~14k), the ln_pair tensor alone is ~13 GiB on top of
    # the input pair tensor still in scope. Streaming over row chunks lets
    # us materialise bias directly without ever holding the full ln'd pair.
    _PAIR_CHUNK: int = 1024

    def _ln_pair_to_bias_chunked(
        self,
        pair: torch.Tensor,
    ) -> torch.Tensor:
        """LayerNorm + bias projection in row chunks; bit-exact with canonical.

        LayerNorm normalises over the trailing ``d_pair`` axis only, so each
        (b, i, j) cell is independent of the rest of the row dimension —
        chunking along ``i`` is exact, not an approximation.
        """
        batch, length, _, _ = pair.shape
        n_head = self.to_bias.weight.shape[0]
        bias = torch.empty(
            (batch, length, length, n_head),
            dtype=pair.dtype,
            device=pair.device,
        )
        chunk = self._PAIR_CHUNK
        for s in range(0, length, chunk):
            e = min(s + chunk, length)
            bias[:, s:e] = self.to_bias(self.ln_pair(pair[:, s:e]))
        return bias

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
            attention = torch.einsum("abhid,abhjd->abhij", query, key)
            attention = attention + bias[None]
            if mask is not None:
                attention = attention.masked_fill(
                    ~mask[:, :, None, None, :],
                    torch.finfo(attention.dtype).min,
                )
            attention = F.softmax(attention, dim=-1)
            return torch.einsum("abhij,abhjd->abhid", attention, value)

        if self.implementation == ImplementationType.MINIWORLD_ENGINE:
            # Whole-op: the pair-biased attention core (incl. the bf16 cast/restore)
            # lives in miniworld-engine; MP-aware q/k/v projections stay in this layer.
            from miniworld_engine import ops

            return ops.augmented_attention_pair_bias(
                query, key, value, bias, mask, kernel_type=self.kernel_type
            )

        raise InvalidImplementationError(self.implementation)

    @typecheck
    def compute_bias(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
    ) -> Float[torch.Tensor, "B H L L"]:
        """Compute the step-invariant pair-derived attention bias."""
        # Default fallback: chunk the pair->bias build whenever not training (avoids the
        # large [B,H,L,L] materialization at long L); never chunk in training.
        use_chunked_inference = not self.training
        if use_chunked_inference and self.implementation == ImplementationType.PYTORCH:
            bias = self._ln_pair_to_bias_chunked(pair)
        elif self.implementation == ImplementationType.MINIWORLD_ENGINE:
            from miniworld_engine import ops

            bias = ops.layer_norm_linear(
                pair, self.ln_pair.weight, self.to_bias.weight, self.ln_pair.eps
            )
        elif self.implementation == ImplementationType.PYTORCH:
            bias = self.to_bias(self.ln_pair(pair))
        else:
            raise InvalidImplementationError(self.implementation)
        if self.use_bias_norm:
            bias = self.norm_bias(bias)
        return rearrange(bias, "B L L2 H -> B H L L2").contiguous()

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
        *,
        bias: Float[torch.Tensor, "B H L L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Forward pass.

        Exactly one of ``pair`` or ``bias`` must be provided. ``bias`` (precomputed
        via :meth:`compute_bias`) takes precedence over ``pair`` when both are
        given.
        """
        if bias is None:
            if pair is None:
                msg = "either `pair` or `bias` must be provided"
                raise ValueError(msg)
            bias = self.compute_bias(pair)

        single = self.ada_ln_in(single, cond)
        query = self.to_query(single)
        key = self.to_key(single)
        value = self.to_value(single)

        query = rearrange(query, "A B L (H D) -> A B H L D", H=self.n_head)
        key = rearrange(key, "A B L (H D) -> A B H L D", H=self.n_head)
        value = rearrange(value, "A B L (H D) -> A B H L D", H=self.n_head)
        if mask is not None and mask.ndim == 2:
            mask = repeat(mask, "B L -> A B L", A=single.shape[0])

        if self.use_qk_norm:
            # Since torch 2.13, CUDA autocast runs rms_norm in FP32; keep Q/K in V's dtype
            # as before so the attention core never receives mixed-precision operands.
            query = self.norm_query(query).to(value.dtype)
            key = self.norm_key(key).to(value.dtype)

        if bias.dtype != query.dtype:
            bias = bias.to(query.dtype)

        out = self._kernel_attention_pair_bias(query, key, value, bias, mask)

        gate = self.to_gate(single)
        out = rearrange(out, "A B H L D -> A B L (H D)")
        out = sigmoid_gate(gate, out)
        out = self.to_out(out)
        return sigmoid_gate(self.to_scale(cond), out)
