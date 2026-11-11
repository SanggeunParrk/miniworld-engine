"""The A100 ``"cuda"`` paths of the ATOM DiT block variants (:class:`~miniworld_engine.modules.atom_dit.AtomDiTBoltz2` /
``AtomDiTRoseTTAFold3`` / ``AtomDiTChai1``, :class:`VariantAtomCuda`) and of the token variants
(:class:`~miniworld_engine.modules.token_dit.TokenDiTBoltz2` / ``TokenDiTRoseTTAFold3`` / ``TokenDiTChai1``, :class:`VariantTokenCuda`). On
H100 the token variants run the one token DiT algorithm (TokenDiTProtenix's ``"cuda"`` route with the model's options) and the atom
variants the one atom DiT implementation (``_LocalAtomDiT``'s ``"cuda"``: kernels/integrations/local_dit_h100 with the model's options).

The A100 kernels (``kernels/dit_variants``: ``train_rows.cu`` / ``edge.cu`` and the cuBLAS products, the local DiT attention
cores of ``kernels/augmented_attention``) were written against a stack of variant blocks (``reference.py``'s parameter names). A block of
these modules is such a stack of one block: :class:`_Stack` is its read-only view onto the module's own submodules (nothing is copied; a
weight change is seen on the next call). Three attribute differences are bridged here:

  * ``expand`` is the concatenation of ``transition.expand_a`` / ``expand_b`` (the kernels' ``[g | v]`` rows; ``up`` follows it);
  * a bias-free query (RoseTTAFold3) is a zero bias;
  * AdaLN: ``to_bias`` is the kernels' ``to_shift``; the sigmoid form's ``ln_cond.weight`` is ``cond_norm_weight``; one AdaLN for q and k / v
    (every atom class here) is the kernels' ``q_adaln`` AND ``k_adaln``.

A100 atom calls, inference included, run ``cuda/train.py`` (autograd Functions, forward and backward) on the sm_80 builds, bf16 or fp32
(TF32), under the caller's autocast (the master layout: fp32 parameters, bf16 activations under bf16 autocast; the conditioning tables are
the module's own layers on the fp32 masters, the kernels cast the weights). Outside the range :meth:`_supports_cuda` states why; there is
no other backend behind it.
"""

from __future__ import annotations

import contextlib
from typing import Any, cast

import torch
import torch.nn as nn

from miniworld_engine.modules.norm import IdentityAdaptiveLayerNorm
from miniworld_engine.modules.transition import ConditionedTransition

_BF, _F32 = torch.bfloat16, torch.float32


class _AdaLN:
    """An AdaLN in the kernels' layout (``VariantAdaLN``): sigmoid form (``cond_norm_weight``, ``to_scale`` with bias) or identity form."""

    def __init__(self, ada) -> None:
        self._ada = ada
        self.identity_scale = isinstance(ada, IdentityAdaptiveLayerNorm)

    @property
    def cond_norm_weight(self):
        return self._ada.ln_cond.weight

    @property
    def to_scale(self):
        return self._ada.to_scale

    @property
    def to_shift(self):
        return self._ada.to_bias


class _Expand:
    """``expand.weight`` = ``[expand_a.weight; expand_b.weight]``: made at access while a gradient is recorded (so it reaches both), else
    kept per (pointer, version) of the two weights (every range check reads its shape: no concatenation per call, none inside a graph)."""

    def __init__(self, transition) -> None:
        self._tr, self._key, self._cat = transition, None, None

    @property
    def weight(self):
        wa, wb = self._tr.expand_a.weight, self._tr.expand_b.weight
        if torch.is_grad_enabled() and (wa.requires_grad or wb.requires_grad):
            return torch.cat([wa, wb])
        key = (wa.data_ptr(), wa._version, wb.data_ptr(), wb._version, wa.dtype)
        if key != self._key:
            self._cat, self._key = torch.cat([wa, wb]), key
        return self._cat


class _ZeroBiasQuery:
    """A query projection without a bias (RoseTTAFold3), as the kernels' ``to_q`` with a zero bias."""

    def __init__(self, linear) -> None:
        self._linear, self._zeros = linear, None

    @property
    def weight(self):
        return self._linear.weight

    @property
    def bias(self):
        w = self._linear.weight
        if self._zeros is None or self._zeros.device != w.device:
            self._zeros = torch.zeros(w.shape[0], dtype=_F32, device=w.device)
        return self._zeros


class _Block:
    def __init__(self, attention, transition, *, atom: bool) -> None:
        self._att, self._tr = attention, transition
        self.n_head = attention.n_head
        self.gating = hasattr(attention, "to_gate")
        self.kq_norm = hasattr(attention, "norm_query")
        self.up_gate = hasattr(transition, "up")
        self.project_output = not isinstance(attention.to_out, nn.Identity)
        self.to_q = attention.to_query if attention.to_query.bias is not None else _ZeroBiasQuery(attention.to_query)
        self.expand = _Expand(transition)
        # ``atom``: the atom blocks are the only callers (the token blocks run the one token DiT algorithm); one AdaLN for q and k / v, or
        # a K / V AdaLN of its own (the AF3 blocks on A100: OpenFold3, Protenix)
        assert atom, "the token DiT blocks run kernels/integrations/token_dit (TokenDiTProtenix's cuda route)"
        adaln = _AdaLN(attention.ada_ln_in)
        kv = getattr(attention, "ada_ln_kv", None)
        self.q_adaln, self.k_adaln = adaln, (adaln if kv is None else _AdaLN(kv))
        self.trans_adaln = _AdaLN(transition.ada_ln_in)

    to_k = property(lambda self: self._att.to_key)
    to_v = property(lambda self: self._att.to_value)
    to_g = property(lambda self: self._att.to_gate)
    to_out = property(lambda self: self._att.to_out)
    q_norm = property(lambda self: self._att.norm_query)
    k_norm = property(lambda self: self._att.norm_key)
    attn_zero = property(lambda self: self._att.to_scale)
    pair_norm = property(lambda self: self._att.ln_pair)
    to_pair_bias = property(lambda self: self._att.to_bias)
    up = property(lambda self: self._tr.up)
    squeeze = property(lambda self: self._tr.squeeze)
    trans_zero = property(lambda self: self._tr.to_scale)
    adaln_eps = property(lambda self: self._att.ada_ln_in.ln_in.eps)


class _Stack:
    """The module as a one-block variant stack (the interface of ``DiTVariantStack`` / ``AtomDiTVariantStack`` the kernels read)."""

    implementation = "miniworld"
    per_block_pair_norm = True

    def __init__(self, module, *, atom: bool) -> None:
        self._module = module
        self.parallel = module.parallel
        #: the K / V AdaLN normalises the query AdaLN's output (Protenix) instead of the raw atoms
        self.chain = getattr(module, "kv_norm", "shared") == "chained"
        self.key_window = getattr(module, "key_window", "slide")
        self.blocks = [_Block(module.attention, module.transition, atom=atom)]

    def parameters(self):
        return self._module.parameters()


def _stack(module, *, atom: bool) -> _Stack:
    view = module.__dict__.get("_variant_stack")
    if view is None:
        view = module.__dict__["_variant_stack"] = _Stack(module, atom=atom)
    return view


def _eps_reason(module) -> str | None:
    """The kernels take one AdaLN epsilon for every LayerNorm of the block and 1e-5 for the conditioning LayerNorm."""
    att, tr = module.attention, module.transition
    adalns = [att.ada_ln_in, tr.ada_ln_in] + ([att.ada_ln_kv] if hasattr(att, "ada_ln_kv") else [])
    eps = {a.ln_in.eps for a in adalns}
    if len(eps) != 1:
        return f"AdaLN epsilons differ across the block's parts ({sorted(eps)}; the kernels take one)"
    if any(not isinstance(a, IdentityAdaptiveLayerNorm) and a.ln_cond.eps != 1e-5 for a in adalns):
        return "conditioning LayerNorm epsilon != 1e-5"
    return None


def _a100(single) -> bool:
    return single.is_cuda and torch.cuda.get_device_capability(single.device) == (8, 0)


def _call_reason(module, single, cond, pair, mask, *, a100: bool = False) -> str | None:
    if not single.is_cuda:
        return "CUDA tensors only"
    if _a100(single):
        if not a100:
            return "A100: no sm_80 port of the DiT-variant kernels (kernels/dit_variants are sm_90a only)"
    elif torch.cuda.get_device_capability(single.device) != (9, 0):
        return "the H100 (sm_90) kernels only"
    if torch.is_autocast_enabled("cuda") and (torch.get_autocast_dtype("cuda") != _BF or single.dtype != _BF):
        # the master-weight layout (fp32 parameters, bf16 activations under bf16 autocast) is served, as the pre-redesign engine did
        # (git 596d568b8 modules/dit_variant: "a call with a gradient (bf16 under autocast ...)"); fp32 activations run without autocast
        return "autocast: bfloat16 activations under bfloat16 autocast only (float32 activations run without autocast, on TF32)"
    if single.ndim != 4 or single.shape[1] != 1:
        return f"single must be [A, 1, L, d] (one pair per call), got {tuple(single.shape)}"
    if single.dtype not in (_BF, _F32) or cond.dtype != single.dtype or pair.dtype != single.dtype:
        return f"dtypes {single.dtype} / {cond.dtype} / {pair.dtype} (single, cond, pair all bfloat16 or all float32)"
    if cond.ndim != 4 or tuple(cond.shape[1:3]) != tuple(single.shape[1:3]) or cond.shape[0] not in (1, single.shape[0]):
        return f"cond must be [A, 1, L, d_cond], got {tuple(cond.shape)}"
    return _eps_reason(module)


def _wants_grad(module, *tensors) -> bool:
    return torch.is_grad_enabled() and (any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)
                                        or any(p.requires_grad for p in module.parameters()))


def _shared_cond(cond: torch.Tensor) -> bool:
    """One conditioning row set for all samples (a single sample, or an expanded stride-0 cond): metadata only, no device read."""
    return cond.shape[0] == 1 or cond.stride(0) == 0


def _run(fn, stack, single, cond, call, *, wants_grad: bool, keep_autocast: bool = False, per_sample: bool = False):
    """``fn(stack, single [A, N, d], cond [N, c], *call)`` once for a shared ``cond``, else once per sample; -> [A, 1, N, d]. ``per_sample``: a
    ``cond`` that differs per sample goes to ``fn`` whole, [A, N, c], in one call (the A100 atom path's per-row tables)."""
    s = single[:, 0]
    # inference packs its own bf16 operands (an enclosing autocast must not recast its torch glue, e.g. the fp32 slid edge windows); the
    # training path keeps the caller's autocast: its conditioning tables run the module's own layers on the fp32 master weights
    off = torch.autocast("cuda", enabled=False) if not (wants_grad or keep_autocast) else contextlib.nullcontext()
    with torch.set_grad_enabled(wants_grad), off:
        if _shared_cond(cond):
            out = fn(stack, s, cond[0, 0], *call)
        elif per_sample:
            out = fn(stack, s, cond[:, 0], *call)
        else:
            out = torch.cat([fn(stack, s[i:i + 1], cond[i, 0], *call) for i in range(s.shape[0])])
    return out.unsqueeze(1)


class VariantTokenCuda:
    """Mixin of the Boltz-2 / RoseTTAFold3 / Chai-1 token DiT blocks: their A100 (sm_80) ``"cuda"`` paths. On H100 the block runs THE token
    DiT algorithm, TokenDiTProtenix's ``"cuda"`` route (kernels/integrations/token_dit, token_dit_train) with its options
    (``token_dit.options``: parallel block, identity AdaLN, no gating query, up-gate transition, flat q / k LayerNorm, bias-free query)."""

    # the block's own submodules (the token DiT class this is mixed into): declared for the type checker only
    attention: Any
    transition: Any
    parallel: bool

    # A100 (sm_80): the attention runs on kernels/integrations/augattn_sm80 (the module's AdaLN, then cuBLAS projections, the pair bias
    # kernels, the mma.sync core and the gates, forward and backward; bf16 or the TF32 core), including its variants: the flat q / k
    # LayerNorm and the bias-free query (RoseTTAFold3), no gating query (Chai-1). The transition: the sm_80 ConditionedTransition
    # (kernels/integrations/conditioned_transition_sm80) where it serves the module (RoseTTAFold3), else the module's own statements
    # (Boltz-2's up-gate transition, Chai-1's identity AdaLN). The parallel blocks add the transition's update of the block input.
    def _a100_token_reason(self, single, cond, pair, mask, compute_dtype, extra_pair_bias) -> str | None:
        from miniworld_engine.kernels.integrations import augattn_sm80
        from miniworld_engine.modules.attention import AugmentedAttentionPairBias

        if not isinstance(self.attention, AugmentedAttentionPairBias):
            return f"A100: attention {type(self.attention).__name__} (no sm_80 path)"
        if extra_pair_bias is not None:
            return "A100: extra_pair_bias (not an input of the sm_80 attention kernels)"
        dt = single.dtype
        if dt not in (_BF, _F32) or cond.dtype != dt or pair.dtype != dt:
            return f"A100: dtypes {single.dtype} / {cond.dtype} / {pair.dtype} (all bfloat16 or all float32)"
        if torch.is_autocast_enabled("cuda") and (torch.get_autocast_dtype("cuda") != _BF or dt != _BF):
            return "A100: autocast: bfloat16 activations under bfloat16 autocast only (float32 activations run without autocast, on TF32)"
        if compute_dtype not in (None, dt):
            return f"A100: compute_dtype {compute_dtype} (the sm_80 kernels run in the activations' dtype)"
        if not augattn_sm80.serves(self.attention, single, pair, mask, compute_dtype, cond=cond):
            return ("A100: outside the sm_80 AugmentedAttentionPairBias kernels' range (bf16 / fp32 with weights in that dtype or fp32, "
                    "head dim 32, 48 or 64, pair LayerNorm eps 1e-5, a bool key mask [B, L] / [A, B, L] or none)")
        return None

    def _a100_transition_delta(self, single, cond):
        """The transition's update of the block input (no residual): the sm_80 ConditionedTransition where it serves the module, else its own
        statements."""
        from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct_sm80

        grad = _wants_grad(self, single, cond)
        if (type(self.transition) is ConditionedTransition and not isinstance(self.transition.ada_ln_in, IdentityAdaptiveLayerNorm)
                and ct_sm80.serves(self.transition, single, cond, single.dtype, grad)):
            with torch.autocast("cuda", enabled=False):
                return ct_sm80.update(self.transition, single, cond, False, single.dtype, grad)
        return self.transition.delta(single, cond)

    def _supports_cuda(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        if _a100(single):
            return self._a100_token_reason(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        # H100: the one token DiT algorithm (TokenDiTProtenix's "cuda" route with this block's options)
        return super()._supports_cuda(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)  # pyright: ignore[reportAttributeAccessIssue]  -- cooperative mixin: the token DiT class after it in the MRO defines it

    def _forward_cuda(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        if _a100(single):
            from miniworld_engine.kernels.integrations import augattn_sm80

            single_a = augattn_sm80.forward(self.attention, single, cond, pair, mask)
            if not self.parallel:
                return self.transition(single_a, cond)
            return single_a + self._a100_transition_delta(single, cond)
        # H100: the one token DiT algorithm (TokenDiTProtenix's "cuda" route with this block's options), inference and training
        return super()._forward_cuda(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)  # pyright: ignore[reportAttributeAccessIssue]  -- cooperative mixin: the token DiT class after it in the MRO defines it


class VariantAtomCuda:
    """Mixin of the Boltz-2 / RoseTTAFold3 / Chai-1 atom DiT blocks: on H100 THE atom DiT implementation (``_LocalAtomDiT``'s ``"cuda"``:
    kernels/integrations/local_dit_h100 with the model's options, every key window), on A100 ``kernels/dit_variants`` (``cuda/train.py``
    on the sm_80 builds)."""

    # the block's own submodules and options (the atom DiT class this is mixed into): declared for the type checker only
    attention: Any
    transition: Any
    key_window: str

    def _atom_serving(self, single, cond, pair, mask, uid) -> tuple[Any, Any]:
        """A100: (the kernel function that serves the call, its trailing arguments) or (None, why not)."""
        if self.key_window not in ("pad", "slide", "slide_qblock"):
            return None, f"key_window {self.key_window!r} (the kernels know the clipped 'pad' window and the slid one)"
        why = _call_reason(self, single, cond, pair, mask, a100=True)
        if why is not None:
            return None, why
        if not _a100(single):
            return None, "kernels/dit_variants serve A100 only (H100: the one atom DiT implementation)"
        n = single.shape[2]
        if (single.shape[-1], self.attention.n_head, pair.shape[-1]) != (128, 4, 16) or self.transition.expand_a.weight.shape[0] != 2 * 128:
            return None, ("geometry: the atom DiT kernels are built for 128 wide x 4 heads of 32, transition n = 2, pair 16 wide, got "
                          f"{single.shape[-1]} wide x {self.attention.n_head} heads, pair {pair.shape[-1]}")
        if tuple(pair.shape) != (1, n // 32, 32, 128, 16):
            return None, f"pair must be the trunked atom pair [1, N / 32, 32, 128, 16] (N a multiple of 32), got {tuple(pair.shape)}"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (1, n)):
            return None, "mask must be a bool [1, N] key mask"
        if uid is not None and tuple(uid.shape) != (1, n):
            return None, f"uid must be [1, N], got {tuple(uid.shape)}"
        from miniworld_engine.kernels.dit_variants.cuda import train

        m, u = (None if mask is None else mask[0]), (None if uid is None else uid[0])
        # A100 (sm_80): every call -- inference included -- on the training path (train.py: the row kernels, the mma.sync window attention
        # and the edge kernels, built for sm_80 by cuda/sm80.py), bf16 or fp32 (TF32), the caller's autocast kept
        if not train.atom_serves(_stack(self, atom=True), single[:, 0], cond[0, 0] if _shared_cond(cond) else cond[:, 0], pair[0], m, u,
                                 need_grad=False):
            return None, ("A100: outside the sm_80 atom training path's range (bfloat16 / float32, [A, 1, N, 128] with N a multiple of 64, "
                          "pair 16 wide, a conditioning shared by the samples or one per sample)")
        return train.atom_stack_train, (pair[0], m, u)

    @staticmethod
    def _h100(single) -> bool:
        return single.is_cuda and torch.cuda.get_device_capability(single.device) == (9, 0)

    def _supports_cuda(self, single, cond, pair, mask=None, uid=None) -> str | None:
        if self._h100(single):
            from miniworld_engine.modules.atom_dit import _LocalAtomDiT

            return _LocalAtomDiT._supports_cuda(cast(_LocalAtomDiT, self), single, cond, pair, mask, uid=uid)
        fn, why = self._atom_serving(single, cond, pair, mask, uid)
        return why if fn is None else None

    def _forward_cuda(self, single, cond, pair, mask=None, uid=None) -> torch.Tensor:
        if self._h100(single):
            from miniworld_engine.modules.atom_dit import _LocalAtomDiT

            return _LocalAtomDiT._forward_cuda(cast(_LocalAtomDiT, self), single, cond, pair, mask, uid)
        fn, call = self._atom_serving(single, cond, pair, mask, uid)
        return _run(fn, _stack(self, atom=True), single, cond, call, wants_grad=_wants_grad(self, single, cond, pair), keep_autocast=True,
                    per_sample=True)
