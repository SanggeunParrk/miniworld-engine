"""The bias-only token DiT block: AF3 Alg. 23 with the attention's query-key half removed.

    a = BiasOnlyAttention(a, s, z, mask)           a + gated softmax(pair bias) v -- no query, no key
    a = ConditionedTransition(a, s)                a + transition(a, s)

The attention logits ARE the pair bias. Everything around the core is ``AugmentedAttentionPairBias``'s (AdaLN, the value and
gate projections, both sigmoid gates, the conditioning scale), minus ``to_query``, ``to_key`` and the QK-norm; it is the
``bias_only_v`` ablation of ``benchmarks/runners/bench.py`` made a module of its own.

Why a module and not a flag on ``DiTBlock``: with no query and no key the attention weights depend on neither the sample
nor the single representation, only on the pair -- ONE softmax serves every augmented sample, and during sampling every
solver step, since the pair carries no noise level. The fast path is therefore a different algorithm (hoisted weights and
a GEMM per head), not the flash-attention core with an argument switched off.

``implementation=PYTORCH`` is the reference. ``MINIWORLD`` / ``TRITON`` (the engine's kernels) take, on B200 in bf16 (or in
fp32 on TF32 tensor cores: an fp32 block with fp32 inputs, no autocast),
``integrations.bias_only_dit`` without autograd and ``integrations.bias_only_dit_train`` with it (CUDA and cuBLAS only), and
the reference composition everywhere else (the parts are built on the PyTorch reference, so no other backend runs).
Each part owns its residual (it returns ``x + f(x)``, like every such module in the engine); the block only chains them.

A block may take its pair bias precomputed (``bias=`` [B, H, L, L], head-major): ``pair_bias_all`` (``hoist.py``) makes every block's
from one LayerNorm and one GEMM. The block then touches neither ``ln_pair`` nor ``to_bias`` (nor ``pair``); its backward hands the
bias gradient to the hoist, which owns the gradients of the pair and of those weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine.integrations import bias_only_dit as _fused
from miniworld_engine.integrations import bias_only_dit_train as _train
from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear

_REF = ImplementationType.PYTORCH


class BiasOnlyAttention(nn.Module):
    """Pair-bias attention without queries and keys, with adaptive conditioning.

    ``out = single + sigmoid(to_scale(cond)) * to_out(sigmoid(g) * softmax(bias) v)`` (the residual is the module's own
    input), with ``bias = to_bias(LN(pair))`` [B, H, L, L]
    shared by every augmented sample, ``v`` and ``g`` projections of ``AdaLN(single, cond)``. A key mask [B, L] enters as the
    largest negative finite logit (a fully masked row is uniform, not NaN). ``bias=`` [B, H, L, L] given (a hoisted pair bias,
    ``pair_bias_all``): those are the logits, and ``ln_pair`` / ``to_bias`` / ``pair`` are not read.
    """

    def __init__(self, d_single: int, d_cond: int, d_pair: int, n_head: int, d_head: int | None = None) -> None:
        super().__init__()
        self.n_head = n_head
        d_hidden = d_head or d_single // n_head          # the head width; n_head x d_head may differ from d_single
        self.ada_ln_in = AdaptiveLayerNorm(d_single, d_cond, implementation=_REF)
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False)
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        # no offset, as in AugmentedAttentionPairBias: a per-head constant on every logit cancels in the softmax
        self.ln_pair = LayerNorm(d_pair, bias=False, implementation=_REF)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="zero")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")
        self.to_scale = Linear(d_cond, d_single, bias=True, init="default")
        self.to_scale.bias.data.fill_(-2.0)

    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        *,
        bias: Float[torch.Tensor, "B H L L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        single_res = single  # residual == the ORIGINAL input (before ada_ln_in rebinds `single`)
        single = self.ada_ln_in(single, cond)
        value, gate = self.to_value(single), self.to_gate(single)
        if bias is None:
            logits = self.to_bias(self.ln_pair(pair)).permute(0, 3, 1, 2)   # (B, H, L, L)
        else:
            logits = bias.to(value.dtype)                                # hoisted (pair_bias_all), head-major already
        value = rearrange(value, "A B L (H D) -> A B L H D", H=self.n_head)
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        # one softmax for every sample: the logits carry no augmentation axis
        attention = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhij,abjhd->abihd", attention, value).flatten(-2)
        out = self.to_out(sigmoid_gate(gate, out))
        return single_res + sigmoid_gate(self.to_scale(cond), out)


class BiasOnlyDiTBlock(nn.Module):
    """Token DiT block with bias-only attention: ``BiasOnlyAttention`` then ``ConditionedTransition``.

    ``forward(single, cond, pair, mask)`` -> ``single``'s shape. ``single`` and ``cond`` carry the augmentation axis
    (``A, B, L, d``); ``pair`` does not (``B, L, L, d_pair``). The token widths are the defaults (768 / cond 384 / pair 128 /
    16 heads x 48 / transition n = 2), the same as ``modules.dit.DiTBlock``; ``n_head=24`` (24 x 32), ``n_head=12`` (12 x 64) and ``d_head=64`` (16 x 64: 1024 attention channels) have the fused paths
    too. ``forward(..., bias=b)``: the block's pair bias precomputed ([B, H, L, L], from ``pair_bias_all``); ``pair`` may be None.
    """

    def __init__(
        self,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 128,
        n_head: int = 16,
        n: int = 2,
        *,
        d_head: int | None = None,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.implementation = ImplementationType(implementation)
        self.attention = BiasOnlyAttention(d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head, d_head=d_head)
        self.transition = ConditionedTransition(d_hidden=d_single, d_cond=d_cond, n=n, implementation=_REF)

    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        *,
        bias: Float[torch.Tensor, "B H L L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Each part returns its residual output (stream in, stream out); the fused paths fold both residuals in. ``bias``: this
        block's hoisted pair bias (``pair_bias_all``) in place of ``to_bias(ln_pair(pair))``."""
        if bias is None and pair is None:
            raise ValueError("BiasOnlyDiTBlock: pass the pair or the block's precomputed bias")
        if _fused.serves(self, single, cond, pair, mask, bias=bias):
            return _fused.update(self, single, cond, pair, mask, bias=bias)
        if _train.serves(self, single, cond, pair, mask, bias=bias):
            return _train.block(self, single, cond, pair, mask, bias=bias)
        if bias is None:
            single = self.attention(single, cond, pair, mask)
        else:
            single = self.attention(single, cond, pair, mask, bias=bias)
        return self.transition(single, cond)
