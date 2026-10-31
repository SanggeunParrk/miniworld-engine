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
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels.bias_only_attention.triton.core_t import bias_only_core_t as _core_t
from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import pair_bias_logits as _pair_bias_logits
from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import supported as _pair_bias_supported
from miniworld_engine.integrations import bias_only_dit as _fused
from miniworld_engine.integrations import bias_only_dit_train as _train
from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.dispatch import portable_blocked
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear

_REF = ImplementationType.PYTORCH

#: the softmax of the pair bias per (pair, mask, weights), see ``BiasOnlyAttention._attention_weights``
_ATTENTION: dict = {}


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

    def _fused_pair_bias(self, pair) -> bool:
        """The portable family's one-pass pair LayerNorm + bias projection (``pair_bias``): a gradient is recorded and the pair is a contiguous
        bf16 / fp32 CUDA tensor whose channel count the kernel serves."""
        return bool(
            self.ada_ln_in.implementation == ImplementationType.MINIWORLD and pair is not None and torch.is_grad_enabled()
            and self.ln_pair.weight is not None and getattr(self.ln_pair, "bias", None) is None and self.to_bias.bias is None
            and (pair.requires_grad or self.ln_pair.weight.requires_grad or self.to_bias.weight.requires_grad)
            and _pair_bias_supported(pair, self.n_head)
            # its two Triton kernels (op bias_only_pair_bias) need the card's record: without it, off H100 / B200, the PyTorch statements
            and not portable_blocked("bias_only_pair_bias", pair.device, pair, self.ln_pair.weight, self.to_bias.weight))

    def _feature_major_psb(self, single) -> bool:
        """The same core for a pair of its own per sample (``single`` [1, B, L, d], B > 1; ``core_t`` with ``p`` [B, H, L, L])."""
        return bool(
            self.ada_ln_in.implementation == ImplementationType.MINIWORLD
            and single.is_cuda and single.ndim == 4 and single.shape[0] == 1 and single.shape[1] > 1
            and single.dtype in (torch.bfloat16, torch.float32)
            and torch.is_grad_enabled()
            and (single.requires_grad or any(p.requires_grad for p in self.parameters()))
            # the core's sigmoid-gate kernels need the card's record (op "bias_only_core"); H100 / B200 are not asked
            and not portable_blocked("bias_only_core", single.device, single, self.to_value.weight, self.to_out.weight))

    def _feature_major(self, single) -> bool:
        """The portable family's training core (``core_t``): a gradient is recorded, ONE pair serves every sample (``single`` [A, 1, L, d]) and the
        activations are bf16 or fp32 on a CUDA device."""
        return bool(
            self.ada_ln_in.implementation == ImplementationType.MINIWORLD
            and single.is_cuda and single.ndim == 4 and single.shape[1] == 1
            and single.dtype in (torch.bfloat16, torch.float32)
            and torch.is_grad_enabled()
            and (single.requires_grad or any(p.requires_grad for p in self.parameters()))
            # the core's sigmoid-gate kernels need the card's record (op "bias_only_core"); H100 / B200 are not asked
            and not portable_blocked("bias_only_core", single.device, single, self.to_value.weight, self.to_out.weight))

    def _attention_weights(self, pair, mask, bias, dtype):
        """softmax over the keys of the pair bias (masked keys at the largest negative finite logit): [B, H, L, L], shared by every sample.

        It depends on the pair, the mask and two weights only, so an inference call (no gradient) keeps it while those are unchanged -- the pair LayerNorm,
        the bias GEMM and the softmax are then not run again for the next block call of a sampling loop, or in a CUDA-graph replay under
        ``static_inputs()`` (``kernels._capture.lookup_inputs``: the pair is the sampler's per-item input, the weights are fixed). The entry holds the
        pair and the mask, so their addresses cannot be reused while it lives; an in-place update bumps ``_version`` and misses.
        """
        def build():
            if bias is None and self._fused_pair_bias(pair):
                # training: LayerNorm + projection in one pass over the pair, head-major logits (kernels.bias_only_attention.triton.pair_bias)
                logits = _pair_bias_logits(pair, self.ln_pair.weight, self.to_bias.weight, self.ln_pair.eps)
            elif bias is None:
                logits = self.to_bias(self.ln_pair(pair)).permute(0, 3, 1, 2)   # (B, H, L, L)
            else:
                logits = bias.to(dtype)                                         # hoisted (pair_bias_all), head-major already
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
            return torch.softmax(logits, dim=-1)

        if (bias is not None or pair is None or torch.is_grad_enabled() or not pair.is_cuda
                or self.ada_ln_in.implementation != ImplementationType.MINIWORLD):   # the PyTorch statements stay the reference
            return build()
        try:
            key = (pair.data_ptr(), pair._version, tuple(pair.shape), pair.dtype, dtype,
                   None if mask is None else (mask.data_ptr(), mask._version),
                   torch.is_autocast_enabled("cuda"),
                   self.ln_pair.weight.data_ptr(), self.ln_pair.weight._version,
                   self.to_bias.weight.data_ptr(), self.to_bias.weight._version)
        except RuntimeError:   # an inference tensor carries no version counter: not cached
            return build()
        entry = _capture.lookup_inputs(
            _ATTENTION, key, lambda: (build(), pair, mask), limit=64,
            valid=lambda e: e[1] is pair and e[2] is mask)
        return entry[0]

    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        *,
        bias: Float[torch.Tensor, "B H L L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        # residual == the ORIGINAL input; the AdaLN hands it back as a stream output of its own node (see forward_with_stream)
        single, single_res = self.ada_ln_in.forward_with_stream(single, cond)
        if self._feature_major(single):
            # training, one pair: the whole core (value / gate GEMM, softmax(bias) v, gate, to_out) in feature-major layout, no permute copies
            # (``kernels.bias_only_attention.triton.core_t``); the pair-bias softmax stays in autograd and receives dP
            attention = self._attention_weights(pair, mask, bias, single.dtype)
            with torch.autocast(device_type="cuda", enabled=False):
                out = _core_t(single.reshape(-1, single.shape[-1]), self.to_value.weight, self.to_gate.weight, self.to_out.weight,
                              attention[0].to(single.dtype))
            return single_res + sigmoid_gate(self.to_scale(cond), out.reshape(single.shape))
        if self._feature_major_psb(single) and (pair if bias is None else bias) is not None \
                and (pair if bias is None else bias).shape[0] == single.shape[1]:
            attention = self._attention_weights(pair, mask, bias, single.dtype)          # [B, H, L, L], a softmax per sample
            with torch.autocast(device_type="cuda", enabled=False):
                out = _core_t(single.reshape(-1, single.shape[-1]), self.to_value.weight, self.to_gate.weight, self.to_out.weight, attention.to(single.dtype))
            return single_res + sigmoid_gate(self.to_scale(cond), out.reshape(single.shape))
        if self.ada_ln_in.implementation == ImplementationType.MINIWORLD:
            # the portable family: the value and the gate projection are one GEMM over the concatenated weights (one read of `single`, one launch,
            # one dgrad and one wgrad in the backward); the PyTorch statements keep the two Linears
            vg = F.linear(single, torch.cat([self.to_value.weight, self.to_gate.weight], 0))
            value, gate = vg.split(self.to_value.out_features, dim=-1)
        else:
            value, gate = self.to_value(single), self.to_gate(single)
        value = rearrange(value, "A B L (H D) -> A B L H D", H=self.n_head)
        # one softmax for every sample: the logits carry no augmentation axis
        attention = self._attention_weights(pair, mask, bias, value.dtype)
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
        if single.ndim == 4 and single.shape[0] == 1 and single.shape[1] > 1 and _fused.serves_psb(self, single, cond, pair, mask, bias=bias):
            return _fused.update_psb(self, single, cond, pair, mask, bias=bias)  # a bias of its own per sample, inference (H100)
        if _fused.serves(self, single, cond, pair, mask, bias=bias):
            return _fused.update(self, single, cond, pair, mask, bias=bias)
        if _train.serves(self, single, cond, pair, mask, bias=bias):
            return _train.block(self, single, cond, pair, mask, bias=bias)
        if bias is None:
            single = self.attention(single, cond, pair, mask)
        else:
            single = self.attention(single, cond, pair, mask, bias=bias)
        return self.transition(single, cond)
