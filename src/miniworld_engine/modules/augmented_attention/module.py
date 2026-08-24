# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/augmented_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float

from miniworld_engine import kernels
from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.dispatch import (
    KernelBackend,
    resolve_augmented_attention,
)
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear


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
        self.implementation = ImplementationType(implementation)
        # 'miniworld' (auto) -> the TRITON attention kernel (the only fused one).
        self._backend = resolve_augmented_attention(self.implementation)

        d_hidden = d_single // n_head

        self.ada_ln_in = AdaptiveLayerNorm(d_single, d_cond)
        self.to_query = Linear(d_single, d_hidden * n_head, bias=True)
        self.to_key = Linear(d_single, d_hidden * n_head, bias=False)
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False)

        if use_qk_norm:
            self.norm_query = nn.RMSNorm(d_hidden)
            self.norm_key = nn.RMSNorm(d_hidden)

        # No offset: this LayerNorm's bias reaches the loss only through `to_bias`, as a per-head
        # constant added to every attention logit, and softmax(z + c) == softmax(z) exactly. Its
        # gradient is therefore identically zero -- measured, not assumed: perturbing it in fp64
        # moves the loss by ~1e-9 regardless of step size, while the same probe on the weight
        # gives a directional derivative that converges to 4 digits. Matches MiniWorld upstream
        # (`nn.LayerNorm(d_pair, bias=False)`) and AlphaFold3 (`create_offset=False` on the
        # pair_input_layer_norm).
        self.ln_pair = LayerNorm(d_pair, bias=False)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="zero")
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")

        self.to_scale = Linear(d_cond, d_single, bias=True, init="default")
        self.to_scale.bias.data.fill_(-2.0)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        """Drop `ln_pair.bias` from checkpoints written before it was removed.

        The parameter had an identically zero gradient, so a trained checkpoint carries it at its
        zero init and discarding it changes nothing numerically. Without this a strict load of any
        existing checkpoint fails on an unexpected key.
        """
        state_dict.pop(f"{prefix}ln_pair.bias", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _kernel_attention_pair_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Attention core. ``compute_dtype`` picks the precision the core RUNS in, independent of
        the dtype the surrounding module carries; ``None`` keeps the caller's.

        The cast is real, not a flag: the autotune cache keys on the dtype of the tensors the
        kernel actually received (``tensor_dtype_of("Q")``), so casting here is what makes a bf16
        run and an fp32 run land in different cache buckets instead of overwriting each other.
        """
        if compute_dtype is not None:
            query, key, value, bias = (t.to(compute_dtype) for t in (query, key, value, bias))

        if self._backend == KernelBackend.PYTORCH:
            query = query * query.shape[-1] ** -0.5
            attention = torch.einsum("abihd,abjhd->abhij", query, key)
            bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, L, L)
            attention = attention + bias[None]
            if mask is not None:
                # finfo.min (not -inf): a fully-masked row of -inf softmaxes to NaN, so
                # the reference must use the largest-finite-negative fill to stay NaN-safe
                # (matches the triton kernel, which is finite on fully-masked rows).
                attention = attention.masked_fill(
                    ~mask[:, :, None, None, :],
                    torch.finfo(attention.dtype).min,
                )
            attention = F.softmax(attention, dim=-1)
            return torch.einsum("abhij,abjhd->abihd", attention, value)

        if self._backend == KernelBackend.TRITON:
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
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Forward pass.

        ``compute_dtype`` sets the precision the ATTENTION CORE runs in, per call. It is taken
        here rather than at construction because the surrounding projections and the core are not
        the same numerical decision: a caller can hold this module in fp32 and still want the
        quadratic softmax(qk^T + bias)v in bf16, which is the shape the whole-op wrapper already
        hardcoded (``whole_op.augmented_attention_pair_bias`` casts to bf16 unconditionally).
        Making it a parameter turns that into a choice the caller states.

        ``None`` keeps the caller's dtype -- the projections, the gate, and the output stay in
        whatever the module carries either way. Only the core is cast, and the result comes back
        in the input dtype, so this changes precision and not the module's interface.
        """
        in_dtype = single.dtype
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

        if mask is not None and mask.ndim == 2:
            mask = repeat(mask, "B L -> A B L", A=single.shape[0])

        if self.use_qk_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        out = self._kernel_attention_pair_bias(
            query, key, value, bias, mask, compute_dtype=compute_dtype
        )
        # back to the caller's precision before the gate/out projections, which are the module's
        # dtype -- the cast above is scoped to the core, not a change of the module's dtype.
        out = out.to(in_dtype)
        out = rearrange(out, "A B L H D -> A B L (H D)")

        out = sigmoid_gate(gate, out)
        out = self.to_out(out)
        return sigmoid_gate(self.to_scale(cond), out)
