import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine import kernels
from miniworld_engine._typecheck import typecheck
from miniworld_engine.integrations import attention_pair_bias_b200 as _b200
from miniworld_engine.integrations import h100_master as _h100_master
from miniworld_engine.modules import dispatch as _dispatch
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
        gate_bias: float = 0.0,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        #: A constant added to the gate logits (Chai-1's pairformer single attention: 1.0); the B200 kernels take none.
        self.gate_bias = float(gate_bias)
        self.use_qk_norm = use_qk_norm
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_augmented_attention(self.implementation)
        self._train_gate = _dispatch.train_gate("augmented_attention", self.implementation, self._backend)
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

        # No offset: ln_pair feeds only the logits (to_bias has no bias); an offset adds one constant per head to every logit, which the softmax cancels.
        self.ln_pair = LayerNorm(d_pair, bias=False, implementation=self.implementation)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints written before the offset was removed carry ln_pair.bias; it had no effect on the output, so dropping it is exact
        state_dict.pop(prefix + "ln_pair.bias", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
        # Hand-CUDA B200 (integrations.attention_pair_bias_b200 states its contract; the residual is folded there).
        if self.gate_bias == 0.0 and _b200.serves_inference(self, single, pair, mask):
            return _b200.update_inference(self, single, pair, mask)
        if self.gate_bias == 0.0 and _b200.serves_train(self, single, pair, mask):
            return _b200.update_train(self, single, pair, mask)
        # H100 / B200 have no Triton fallback: a MINIWORLD call the native CUDA path does not serve runs the PyTorch attention.
        no_triton = (
            self.implementation == ImplementationType.MINIWORLD
            and single.is_cuda
            and _dispatch.restricted_card(single.device)
        )
        backend = KernelBackend.PYTORCH if no_triton else _dispatch.call_backend(self, self._backend, single, pair)
        master = (
            self._backend == KernelBackend.PORTABLE
            and single.dtype == torch.bfloat16
            and self.to_query.weight.dtype == torch.float32
            and _h100_master.is_h100(single.device)
        )
        if master:
            wq, wk, wv, wb, wg, wo, bq = _h100_master.pack(
                [
                    self.to_query.weight,
                    self.to_key.weight,
                    self.to_value.weight,
                    self.to_bias.weight,
                    self.to_gate.weight,
                    self.to_out.weight,
                    self.to_query.bias,
                ]
            )
        single_res = (
            single  # residual == the ORIGINAL input (before ln_single rebinds `single`)
        )
        single = self.ln_single(single)
        query = (
            _h100_master.linear(single, self.to_query, wq, bq)
            if master
            else self.to_query(single)
        )
        key = (
            _h100_master.linear(single, self.to_key, wk)
            if master
            else self.to_key(single)
        )
        value = (
            _h100_master.linear(single, self.to_value, wv)
            if master
            else self.to_value(single)
        )

        fused_out = None
        if (
            not self.use_qk_norm
            and self._backend == KernelBackend.PORTABLE
            and query.dtype == torch.bfloat16
            and pair.dtype == torch.bfloat16
            and pair.shape[-1] != 128
            and self.ln_pair.eps == 1e-5
            and (mask is None or mask.ndim == 2)
        ):
            # H100: the pair LayerNorm + head projection + attention as hand-written CUDA (kernels/augmented_attention/cuda/fused_pair);
            # d_pair 128 is served by the native APB path above, other widths (256) land here
            from miniworld_engine.kernels.augmented_attention.cuda import fused_pair
            from miniworld_engine.modules.augmented_attention.module import (
                _bf16_sm90_enabled,
            )

            L_ = query.shape[1]
            q5, k5, v5 = (t.reshape(1, 1, L_, self.n_head, -1) for t in (query, key, value))
            if _bf16_sm90_enabled() and fused_pair.available(
                q5, pair, self.ln_pair.weight, self.to_bias.weight, None if mask is None else mask[0]
            ):
                fused_out = fused_pair.attention_pair_fused(
                    q5, k5, v5, pair, self.ln_pair.weight, self.to_bias.weight, None if mask is None else mask[0]
                ).reshape(1, L_, -1)
        if fused_out is not None:
            gate = _h100_master.linear(single, self.to_gate, wg) if master else self.to_gate(single)
            if self.gate_bias:
                gate = gate + self.gate_bias
            out = sigmoid_gate(gate, fused_out.to(single.dtype))
            out = _h100_master.linear(out, self.to_out, wo) if master else self.to_out(out)
            return single_res + out

        query = rearrange(query, "B L (H D) -> B H L D", H=self.n_head)
        key = rearrange(key, "B L (H D) -> B H L D", H=self.n_head)
        value = rearrange(value, "B L (H D) -> B H L D", H=self.n_head)

        if self.use_qk_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        if (
            backend == KernelBackend.PORTABLE
            and pair.dtype == torch.bfloat16
            and pair.shape[-1] <= 1024
        ):
            # portable inference: LayerNorm + the head projection in ONE pass over the pair, stored head-major (the unfused pair LayerNorm
            # and GEMM read and write it three times); the key mask is the attention kernel's own (it takes it below)
            ln_w = self.ln_pair.weight
            B_, L_ = pair.shape[0], pair.shape[1]
            if not torch.is_grad_enabled():
                from miniworld_engine.kernels.layernorm_linear.triton.fused import (
                    layernorm_linear_triton_fwd_t,
                )

                bias = layernorm_linear_triton_fwd_t(
                    pair, ln_w, torch.zeros_like(ln_w), self.to_bias.weight, self.ln_pair.eps
                ).view(self.n_head, B_, L_, L_).permute(1, 0, 2, 3)
            else:
                from miniworld_engine.kernels.layernorm_linear.autograd import layernorm_linear_triton_fn

                bias = layernorm_linear_triton_fn(
                    pair, ln_w, torch.zeros_like(ln_w), self.to_bias.weight.to(pair.dtype), None, self.ln_pair.eps, L_
                )
                bias = rearrange(bias, "B L L2 H -> B H L L2")
        else:
            pair = self.ln_pair(pair)
            bias = (
                _h100_master.linear(pair, self.to_bias, wb)
                if master
                else self.to_bias(pair)
            )
            bias = rearrange(bias, "B L L2 H -> B H L L2")
            if mask is not None:
                bias = bias.masked_fill(
                    ~mask[:, None, None, :], torch.finfo(bias.dtype).min
                )
        if backend == KernelBackend.PYTORCH:
            out = F.scaled_dot_product_attention(query, key, value, bias)
        elif backend == KernelBackend.PORTABLE:
            # Reuse the existing QK + pair-bias kernel with one augmentation.
            # Bias-only attention is a different operation and cannot replace QK.
            q, k, v = (t.transpose(1, 2).unsqueeze(0) for t in (query, key, value))
            out = (
                kernels.triton_augmented_attention_pair_bias(
                    q,
                    k,
                    v,
                    bias.permute(0, 2, 3, 1),
                    None if mask is None else mask.unsqueeze(0),
                )
                .squeeze(0)
                .transpose(1, 2)
            )
        else:
            raise InvalidImplementationError(self.implementation)

        gate = (
            _h100_master.linear(single, self.to_gate, wg)
            if master
            else self.to_gate(single)
        )
        if self.gate_bias:
            gate = gate + self.gate_bias
        out = rearrange(out, "B H L D -> B L (H D)")
        out = sigmoid_gate(gate, out)
        out = _h100_master.linear(out, self.to_out, wo) if master else self.to_out(out)
        return single_res + out
