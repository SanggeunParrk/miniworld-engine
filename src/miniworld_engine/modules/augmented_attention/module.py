# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/augmented_attention.py
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float

from miniworld_engine import kernels
from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules import dispatch as _dispatch
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
from miniworld_engine.modules.primitives import LayerNorm, Linear, RMSNorm


def _bf16_sm90_enabled() -> bool:
    """Whether a bf16 attention core on sm_90 may use the hand-CUDA kernels (settings.augmented_attention_bf16_sm90)."""
    from miniworld_engine import settings

    return settings.current().augmented_attention_bf16_sm90 and (not settings.current().force_portable_kernels)


def _bf16_sm100_enabled() -> bool:
    """Whether a bf16 attention core on sm_100 may use the hand-CUDA kernels (settings.augmented_attention_bf16_sm100)."""
    from miniworld_engine import settings

    return settings.current().augmented_attention_bf16_sm100 and (not settings.current().force_portable_kernels)


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
    kernel_type : {"compute_efficient", "memory_efficient"}
        Split-buffer or atomic gradient accumulation for the Triton backend.
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
        kernel_type: Literal["compute_efficient", "memory_efficient"] = "compute_efficient",
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        if kernel_type not in ("compute_efficient", "memory_efficient"):
            raise ValueError(f"unknown kernel_type {kernel_type!r}")
        self.kernel_type = kernel_type
        self.n_head = n_head
        self.use_qk_norm = use_qk_norm
        self.implementation = ImplementationType(implementation)
        # 'miniworld' (auto) -> the PORTABLE attention kernel (the only fused one).
        self._backend = resolve_augmented_attention(self.implementation)
        self._train_gate = _dispatch.train_gate("augmented_attention", self.implementation, self._backend)

        d_hidden = d_single // n_head

        self.ada_ln_in = AdaptiveLayerNorm(
            d_single, d_cond, implementation=self.implementation,
        )
        self.to_query = Linear(d_single, d_hidden * n_head, bias=True)
        self.to_key = Linear(d_single, d_hidden * n_head, bias=False)
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False)

        if use_qk_norm:
            self.norm_query = RMSNorm(d_hidden, implementation=self.implementation)
            self.norm_key = RMSNorm(d_hidden, implementation=self.implementation)

        # No offset: this LayerNorm's bias reaches the loss only through `to_bias`, as a per-head
        # constant added to every attention logit, and softmax(z + c) == softmax(z) exactly. Its
        # gradient is therefore identically zero -- measured, not assumed: perturbing it in fp64
        # moves the loss by ~1e-9 regardless of step size, while the same probe on the weight
        # gives a directional derivative that converges to 4 digits. Matches the upstream model
        # (`nn.LayerNorm(d_pair, bias=False)`) and AlphaFold3 (`create_offset=False` on the
        # pair_input_layer_norm).
        self.ln_pair = LayerNorm(d_pair, bias=False, implementation=self.implementation)
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

    def _fused_pair_core(self, query, pair, mask, compute_dtype, extra_pair_bias):
        """The H100 pair-fused core that serves this call (``cuda.fused_pair`` for bf16, ``cuda.fused_pair_tf32`` for fp32: the pair
        LayerNorm + head projection + attention in hand-written CUDA), as an entry function, or None."""
        if (
            extra_pair_bias is not None
            or self._backend != KernelBackend.PORTABLE
            or not _bf16_sm90_enabled()
            or not query.is_cuda
            or (mask is not None and mask.ndim != 2)
            or self.ln_pair.eps != 1e-5
        ):
            return None
        m1 = None if mask is None else mask[0]
        dt = compute_dtype or query.dtype
        if dt is torch.bfloat16:
            from miniworld_engine.kernels.augmented_attention.cuda import (
                fused_pair as mod,
            )

            fn = mod.attention_pair_fused
        elif dt is torch.float32 and compute_dtype is None and not torch.is_autocast_enabled("cuda"):
            from miniworld_engine.kernels.augmented_attention.cuda import (
                fused_pair_tf32 as mod,
            )

            fn = mod.attention_pair_tf32
        else:
            return None
        return fn if mod.available(query, pair, self.ln_pair.weight, self.to_bias.weight, m1) else None

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
        if ((compute_dtype or query.dtype) is torch.bfloat16 and self._backend == KernelBackend.PORTABLE
                and _bf16_sm90_enabled()):
            from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

            if cuda_sm90.available(query, bias):
                # Takes the fp32 (or bf16) tensors as they are: the bf16 cast happens in the kernels' one prep pass.
                return cuda_sm90.augmented_attention_bf16_sm90(query, key, value, bias, mask)
        if (compute_dtype is torch.bfloat16 and self._backend == KernelBackend.PORTABLE
                and _bf16_sm100_enabled()):
            from miniworld_engine.kernels.augmented_attention.cuda import (
                sm100 as cuda_sm100,
            )

            if cuda_sm100.available(query, bias, mask):
                return cuda_sm100.augmented_attention_bf16_sm100(query, key, value, bias)
        if compute_dtype is not None:
            query, key, value, bias = (t.to(compute_dtype) for t in (query, key, value, bias))

        backend = _dispatch.call_backend(self, self._backend, query, key, value, bias)   # card recorded for inference only: PyTorch for training
        if (
            backend == KernelBackend.PORTABLE
            and self.implementation == ImplementationType.MINIWORLD
            and _dispatch.restricted_card(query.device)
        ):
            backend = KernelBackend.PYTORCH  # H100 / B200: no Triton fallback; the CUDA cores were tried above
        if backend == KernelBackend.PYTORCH:
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

        if backend == KernelBackend.PORTABLE:
            return kernels.triton_augmented_attention_pair_bias(
                query,
                key,
                value,
                bias,
                mask,
                compute_efficient=self.kernel_type == "compute_efficient",
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
        extra_pair_bias: Float[torch.Tensor, "L L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Forward pass. ALWAYS returns the residual output ``single + delta(single, cond, pair, mask)``
        (the residual is this module's own input), so a block just calls ``single = module(...)``.
        A caller that composes the update some other way (a magnitude-preserving sum, a released
        checkpoint's own residual) takes :meth:`delta`."""
        return single + self.delta(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

    @typecheck
    def delta(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
        extra_pair_bias: Float[torch.Tensor, "L L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """The update alone (no residual), through the same dispatch as :meth:`forward`.

        ``extra_pair_bias`` [L, L]: an additive logit term for every head, sample and batch row, on top of the pair's own bias
        (``to_bias(ln_pair(pair))``) and before the key mask, in the dtype of the bias (the structural-token family's relation
        matrix). Differentiable if it requires a gradient. It rides the attention's ordinary bias input, so every core takes it
        (the portable attention kernels, the PyTorch einsums).

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
        query = self.to_query(single)
        key = self.to_key(single)
        value = self.to_value(single)
        gate = self.to_gate(single)

        num_aug, batch, len_res = query.shape[:3]
        n_head, hidden_dim = self.n_head, query.shape[-1] // self.n_head

        query, key, value = [
            x.view(num_aug, batch, len_res, n_head, hidden_dim)
            for x in (query, key, value)
        ]
        if self.use_qk_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        fused = self._fused_pair_core(query, pair, mask, compute_dtype, extra_pair_bias)
        if fused is not None:
            # H100: the pair LayerNorm + head projection + attention as hand-written CUDA (one read of the pair tensor)
            out = fused(
                query, key, value, pair, self.ln_pair.weight, self.to_bias.weight, None if mask is None else mask[0]
            )
        else:
            pair = self.ln_pair(pair)
            bias = self.to_bias(pair)
            if extra_pair_bias is not None:
                if extra_pair_bias.ndim != 2 or tuple(extra_pair_bias.shape) != tuple(bias.shape[1:3]):
                    raise ValueError(f"extra_pair_bias must be [L, L] = {tuple(bias.shape[1:3])}, got {tuple(extra_pair_bias.shape)}")
                bias = bias + extra_pair_bias.to(bias.dtype)[None, :, :, None]   # [B, L, L, H]: the same term for every head
            if mask is not None and mask.ndim == 2:
                mask = repeat(mask, "B L -> A B L", A=single.shape[0])
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
