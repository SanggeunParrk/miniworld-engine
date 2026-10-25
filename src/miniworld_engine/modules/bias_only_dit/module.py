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

``implementation=PYTORCH`` is the reference. ``MINIWORLD`` / ``PORTABLE`` (the engine's kernels) take, on H100 / B200 in bf16 (or in
fp32 on TF32 tensor cores: an fp32 block with fp32 inputs, no autocast),
``integrations.bias_only_dit`` without autograd and ``integrations.bias_only_dit_train`` with it (CUDA and cuBLAS only), and
the composition of the parts everywhere else. The parts (AdaptiveLayerNorm, ConditionedTransition, the pair LayerNorm) are built with
the block's implementation, so they route per card as any module does: H100 / B200 run their PyTorch leg for a call no CUDA block path
serves; another card runs the portable kernels of the parts (forward and backward, bf16 and fp32) when the engine holds a record for
it, else the PyTorch statements with one ``UnrecordedCardWarning``. The attention core (softmax of the bias times the value) is the
batched einsum on every card: one softmax shared by every sample, a cuBLAS GEMM per head, which the portable core kernel does not beat.
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

    def __init__(self, d_single: int, d_cond: int, d_pair: int, n_head: int, d_head: int | None = None, *,
                 implementation: ImplementationType = _REF) -> None:
        super().__init__()
        self.n_head = n_head
        d_hidden = d_head or d_single // n_head          # the head width; n_head x d_head may differ from d_single
        # The parts route per card like every other module (``modules.dispatch``): the CUDA block paths of ``BiasOnlyDiTBlock`` take the
        # calls they serve (H100 / B200), and on any other card the portable kernels of these parts run where the card is recorded,
        # the PyTorch statements otherwise (one UnrecordedCardWarning). H100 / B200 resolve the parts to their PyTorch leg.
        self.ada_ln_in = AdaptiveLayerNorm(d_single, d_cond, implementation=implementation)
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False)
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        # no offset, as in AugmentedAttentionPairBias: a per-head constant on every logit cancels in the softmax
        self.ln_pair = LayerNorm(d_pair, bias=False, implementation=implementation)
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
        # only ``miniworld`` reaches the portable kernels; every other value keeps the parts on the PyTorch statements
        parts = self.implementation if self.implementation == ImplementationType.MINIWORLD else _REF
        self.attention = BiasOnlyAttention(d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head, d_head=d_head,
                                           implementation=parts)
        self.transition = ConditionedTransition(d_hidden=d_single, d_cond=d_cond, n=n, implementation=parts)

    def _psb_candidate(self, single, cond, pair, bias) -> bool:
        """A pair (or hoisted bias) of its own per sample: ``single`` [1, B, L, d] with ``pair`` [B, L, L, d_pair] or ``bias``
        [B, H, L, L], B > 1 (the samples folded into the batch axis, each attending with its own bias). Training on Hopper only
        (``integrations.bias_only_dit_train.block_psb``)."""
        if (pair is None and bias is None) or not torch.is_grad_enabled() or self.implementation is _REF:
            return False
        if not (single.ndim == 4 and single.shape[0] == 1 and single.shape[1] > 1):
            return False
        src = pair if bias is None else bias
        if src.ndim != 4 or src.shape[0] != single.shape[1]:
            return False
        from miniworld_engine.kernels.bias_only_dit.cuda import h100_enabled

        return bool(single.is_cuda and h100_enabled(single.device) and src.dtype is single.dtype)

    def _forward_psb(self, single, cond, bias, mask):
        """The fused per-sample-bias block (``bias_only_dit_train.block_psb``) on ``bias`` [B, H, L, L], or None when it does not serve
        the call. The bias is read as head-major [H, B, L, L]: a hoist's (``pair_bias_all``) is already that buffer, any other layout
        is copied (differentiably)."""
        B, L = single.shape[1], single.shape[2]
        if tuple(bias.shape) != (B, self.attention.n_head, L, L):
            return None
        s, c = single.reshape(B, 1, L, -1), cond.reshape(B, 1, L, -1)
        from miniworld_engine.integrations.bias_only_dit_hoist import SINK_ATTR

        hb = bias.permute(1, 0, 2, 3)
        hb = hb if hb.is_contiguous() else hb.contiguous()
        if not _train.serves_psb(self, s, c, hb, mask):
            return None
        return _train.block_psb(self, s, c, hb, mask, getattr(bias, SINK_ATTR, None)).reshape(single.shape)

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
        if self._psb_candidate(single, cond, pair, bias):  # a bias of its own per sample (training, Hopper)
            if bias is None:
                from miniworld_engine.modules.bias_only_dit.hoist import pair_bias_all

                bias = pair_bias_all([self], pair)[0]  # [B, H, L, L], made once: the reference below reads it too if not served
            out = self._forward_psb(single, cond, bias, mask)
            if out is not None:
                return out
        if _fused.serves(self, single, cond, pair, mask, bias=bias):
            return _fused.update(self, single, cond, pair, mask, bias=bias)
        if _train.serves(self, single, cond, pair, mask, bias=bias):
            return _train.block(self, single, cond, pair, mask, bias=bias)
        if bias is None:
            single = self.attention(single, cond, pair, mask)
        else:
            single = self.attention(single, cond, pair, mask, bias=bias)
        return self.transition(single, cond)
