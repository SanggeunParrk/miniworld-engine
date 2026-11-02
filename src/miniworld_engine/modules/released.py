"""Adapters from the *released checkpoints' parameter layout* to the engine modules' own.

The AlphaFold 3 family's released weights (AF3, Protenix, Boltz, Chai, RoseTTAFold3 ... as FoldForge's dense modules name them) differ
from the engine modules in names and in two layouts:

* the triangle multiplication's ``projection`` / ``gate`` are ONE matrix of ``2 * hidden`` rows whose rows interleave the two branches
  per hidden channel (row ``2 k`` = branch a of channel k, row ``2 k + 1`` = branch b); the engine has ``to_left`` / ``to_right`` (+ gates);
* the transition's ``transition1`` is one ``2 * n * c`` matrix (``chunk(2)`` -> a, b); the engine has ``expand_a`` / ``expand_b``;
* the outer-product mean's ``output_w`` is ``[K, K, out]`` (``[G, K, K, out]`` grouped), the engine's ``to_out`` a ``Linear``.

Every ``*_from_released`` function maps one released module's ``state_dict`` (optionally below ``prefix``) to the engine module's
``state_dict`` (pure tensor reshuffling, no copy of values, CPU-safe); the matching ``from_released`` classmethod of the engine module
builds the module with the options the released equations need and loads it strictly. Nothing is guessed: what the weights do not
say (head counts, the ending node's bias orientation, the OPM divisor) is an argument.

Released names (the keys the converters read), per module:

    TriangleMultiplication  left_norm_input.{weight,bias} projection.weight gate.weight center_norm.{weight,bias}
                            output_projection.weight gating_linear.weight
    TriangleAttention       act_norm.{weight,bias} pair_bias_projection.weight q_projection.weight k_projection.weight v_projection.weight
                            gating_query.{weight[,bias]} output_projection.{weight[,bias]} [output_projection_transposed.{weight[,bias]}]
    Transition              input_layer_norm.{weight,bias} transition1.weight transition2.weight
    OuterProductMean        layer_norm_input.{weight,bias} left_projection.{weight[,bias]} right_projection.{weight[,bias]} output_w output_b
                            [product_norm.{weight,bias}]
    MSAPairWeightedAveraging act_norm.{weight,bias} pair_norm.weight pair_logits.weight v_projection.weight gating_query.weight
                            output_projection.weight
    AttentionPairBias       (the Pairformer's single attention) adaptive_layernorm.layer_norm.{weight,bias} q_projection.{weight,bias}
                            k_projection.weight v_projection.weight gating_query.weight adaptive_zero_init.transition2.weight, and the
                            pair logits ``norm`` / ``projection`` (``single_pair_logits_norm`` / ``single_pair_logits_projection``)
    PairformerBlock         triangle_multiplication_outgoing / _incoming, pair_attention1 / pair_attention2, pair_transition, and with the
                            single track single_pair_logits_norm, single_pair_logits_projection, single_attention_, single_transition
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

Tensor = torch.Tensor
State = Mapping[str, Tensor]


def _sub(state: State, prefix: str) -> dict[str, Tensor]:
    """The entries of ``state`` below ``prefix`` (a dotted path, with or without the trailing dot), prefix removed."""
    if prefix and not prefix.endswith("."):
        prefix += "."
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _need(state: State, *keys: str) -> None:
    missing = [k for k in keys if k not in state]
    if missing:
        raise KeyError(f"released state_dict lacks {missing} (has {sorted(state)[:12]}...)")


def _put(out: dict[str, Tensor], name: str, state: State, key: str) -> None:
    if key in state:
        out[name] = state[key]


def _norm(out: dict[str, Tensor], name: str, state: State, key: str, bias: bool = True) -> None:
    _need(state, f"{key}.weight")
    out[f"{name}.weight"] = state[f"{key}.weight"]
    if bias:
        _need(state, f"{key}.bias")
        out[f"{name}.bias"] = state[f"{key}.bias"]


def triangle_multiplication_from_released(state: State, *, outgoing: bool, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``TriangleMultiplication`` weights from the released interleaved ``projection`` / ``gate``.

    The released contraction is ``einsum("cik,cjk->cij", a, b)`` (outgoing) / ``einsum("ckj,cki->cij", a, b)`` (incoming); the engine's
    incoming contracts ``left[k, i] * right[k, j]``, so an incoming module takes branch b as ``left`` and branch a as ``right``.
    """
    s = _sub(state, prefix)
    _need(s, "projection.weight", "gate.weight", "output_projection.weight", "gating_linear.weight")
    projection, gate = s["projection.weight"], s["gate.weight"]
    hidden = projection.shape[0] // 2
    if projection.shape[0] != 2 * hidden or gate.shape != projection.shape:
        raise ValueError(f"projection {tuple(projection.shape)} / gate {tuple(gate.shape)} must both be [2 * hidden, c_pair]")
    p = projection.reshape(hidden, 2, -1).transpose(0, 1)
    g = gate.reshape(hidden, 2, -1).transpose(0, 1)
    left, right = (0, 1) if outgoing else (1, 0)
    out: dict[str, Tensor] = {}
    _norm(out, "ln_pair", s, "left_norm_input")
    _norm(out, "ln_out", s, "center_norm")
    out["to_left.weight"], out["to_right.weight"] = p[left], p[right]
    out["to_left_gate.weight"], out["to_right_gate.weight"] = g[left], g[right]
    out["to_out.weight"] = s["output_projection.weight"]
    out["to_gate.weight"] = s["gating_linear.weight"]
    return {k: v.contiguous() for k, v in out.items()}


def triangle_attention_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``TriangleAttention`` weights. Biases are carried when present (``gate_bias`` / ``out_bias``); the dual output's second
    projection is ``to_out_t``."""
    s = _sub(state, prefix)
    _need(s, "pair_bias_projection.weight", "q_projection.weight", "k_projection.weight", "v_projection.weight",
          "gating_query.weight", "output_projection.weight")
    out: dict[str, Tensor] = {}
    _norm(out, "ln_pair", s, "act_norm")
    for name, key in (("to_bias", "pair_bias_projection"), ("to_query", "q_projection"), ("to_key", "k_projection"),
                      ("to_value", "v_projection"), ("to_gate", "gating_query"), ("to_out", "output_projection"),
                      ("to_out_t", "output_projection_transposed")):
        _put(out, f"{name}.weight", s, f"{key}.weight")
        _put(out, f"{name}.bias", s, f"{key}.bias")
    return out


def transition_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``Transition`` weights: ``transition1`` is ``[2 n c, c]``, ``chunk(2)`` -> ``expand_a`` / ``expand_b``."""
    s = _sub(state, prefix)
    _need(s, "transition1.weight", "transition2.weight")
    a, b = s["transition1.weight"].chunk(2, dim=0)
    out: dict[str, Tensor] = {}
    _norm(out, "ln_in", s, "input_layer_norm")
    out["expand_a.weight"], out["expand_b.weight"] = a.contiguous(), b.contiguous()
    out["squeeze.weight"] = s["transition2.weight"]
    return out


def outer_product_mean_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``OuterProductMean`` weights. ``output_w`` ``[K, K, out]`` (``[G, K, K, out]``) becomes ``to_out.weight`` ``[out, G K K]``."""
    s = _sub(state, prefix)
    _need(s, "left_projection.weight", "right_projection.weight", "output_w", "output_b")
    out: dict[str, Tensor] = {}
    _norm(out, "ln_msa", s, "layer_norm_input")
    for name, key in (("to_left", "left_projection"), ("to_right", "right_projection")):
        _put(out, f"{name}.weight", s, f"{key}.weight")
        _put(out, f"{name}.bias", s, f"{key}.bias")
    w = s["output_w"]
    out["to_out.weight"] = w.reshape(-1, w.shape[-1]).T.contiguous()
    out["to_out.bias"] = s["output_b"]
    if "product_norm.weight" in s:
        _norm(out, "product_norm", s, "product_norm")
    return out


def msa_pair_weighted_averaging_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``MSAPairWeightedAveraging`` weights (the pair LayerNorm offset is dropped: a per-head constant, softmax-invariant)."""
    s = _sub(state, prefix)
    _need(s, "pair_norm.weight", "pair_logits.weight", "v_projection.weight", "gating_query.weight", "output_projection.weight")
    out: dict[str, Tensor] = {}
    _norm(out, "ln_msa", s, "act_norm")
    out["ln_pair.weight"] = s["pair_norm.weight"]
    out["to_bias.weight"] = s["pair_logits.weight"]
    out["to_value.weight"] = s["v_projection.weight"]
    out["to_gate.weight"] = s["gating_query.weight"]
    out["to_out.weight"] = s["output_projection.weight"]
    return out


def attention_pair_bias_from_released(attention: State, norm: State, projection: State) -> dict[str, Tensor]:
    """Engine ``AttentionPairBias`` weights from the released single attention (``attention``, AF3 form) and the pair-logit
    ``norm`` / ``projection`` states (each already stripped of its prefix: ``{"weight", "bias"}`` / ``{"weight"}``). The pair
    LayerNorm offset is dropped (softmax-invariant)."""
    a = attention
    _need(a, "adaptive_layernorm.layer_norm.weight", "adaptive_layernorm.layer_norm.bias", "q_projection.weight", "q_projection.bias",
          "k_projection.weight", "v_projection.weight", "gating_query.weight", "adaptive_zero_init.transition2.weight")
    _need(norm, "weight")
    _need(projection, "weight")
    return {
        "ln_single.weight": a["adaptive_layernorm.layer_norm.weight"],
        "ln_single.bias": a["adaptive_layernorm.layer_norm.bias"],
        "to_query.weight": a["q_projection.weight"],
        "to_query.bias": a["q_projection.bias"],
        "to_key.weight": a["k_projection.weight"],
        "to_value.weight": a["v_projection.weight"],
        "ln_pair.weight": norm["weight"],
        "to_bias.weight": projection["weight"],
        "to_gate.weight": a["gating_query.weight"],
        "to_out.weight": a["adaptive_zero_init.transition2.weight"],
    }


def pairformer_block_from_released(state: State, *, outgoing_first: bool = True, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``PairformerBlock`` weights from a released ``PairformerBlock`` (pair track; the single track when its keys are there)."""
    s = _sub(state, prefix)
    out: dict[str, Tensor] = {}
    for src, dst, outgoing in (("triangle_multiplication_outgoing", "tri_multi_outgoing", True),
                               ("triangle_multiplication_incoming", "tri_multi_incoming", False)):
        out |= {f"{dst}.{k}": v for k, v in triangle_multiplication_from_released(s, outgoing=outgoing, prefix=src).items()}
    if any(k.startswith("pair_attention1.") for k in s):
        for src, dst in (("pair_attention1", "tri_atten_starting"), ("pair_attention2", "tri_atten_ending")):
            out |= {f"{dst}.{k}": v for k, v in triangle_attention_from_released(s, prefix=src).items()}
    out |= {f"transition_pair.{k}": v for k, v in transition_from_released(s, prefix="pair_transition").items()}
    if any(k.startswith("single_attention_.") for k in s):
        out |= {f"single_attention.{k}": v for k, v in attention_pair_bias_from_released(
            _sub(s, "single_attention_"), _sub(s, "single_pair_logits_norm"), _sub(s, "single_pair_logits_projection")).items()}
        out |= {f"single_transition.{k}": v for k, v in transition_from_released(s, prefix="single_transition").items()}
    return out


def _adaln_from_released(s: State, key: str, name: str, out: dict[str, Tensor]) -> bool:
    """Engine ``VariantAdaLN`` tensors of a released ``AdaptiveLayerNorm`` at ``key``; returns its ``identity_scale``."""
    identity = f"{key}.single_cond_layer_norm.weight" not in s
    _need(s, f"{key}.single_cond_scale.weight", f"{key}.single_cond_bias.weight")
    out[f"{name}.to_scale.weight"] = s[f"{key}.single_cond_scale.weight"]
    out[f"{name}.to_shift.weight"] = s[f"{key}.single_cond_bias.weight"]
    if not identity:
        out[f"{name}.cond_norm_weight"] = s[f"{key}.single_cond_layer_norm.weight"]
        _need(s, f"{key}.single_cond_scale.bias")
        out[f"{name}.to_scale.bias"] = s[f"{key}.single_cond_scale.bias"]
    return identity


def _variant_transition_from_released(s: State, key: str, prefix: str, out: dict[str, Tensor]) -> None:
    """``DiffusionTransition`` / ``GatedDiffusionTransition`` at ``key`` -> the variant block's ``trans_*`` / ``expand`` / ``up`` / ``squeeze``."""
    _need(s, f"{key}.transition1.weight", f"{key}.adaptive_zero_init.transition2.weight",
          f"{key}.adaptive_zero_init.adaptive_zero_cond.weight", f"{key}.adaptive_zero_init.adaptive_zero_cond.bias")
    out[f"{prefix}.expand.weight"] = s[f"{key}.transition1.weight"]
    out[f"{prefix}.squeeze.weight"] = s[f"{key}.adaptive_zero_init.transition2.weight"]
    out[f"{prefix}.trans_zero.weight"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.weight"]
    out[f"{prefix}.trans_zero.bias"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.bias"]
    _adaln_from_released(s, f"{key}.adaptive_layernorm", f"{prefix}.trans_adaln", out)
    _put(out, f"{prefix}.up.weight", s, f"{key}.a_to_b.weight")


def _variant_attention_common(s: State, key: str, prefix: str, out: dict[str, Tensor]) -> None:
    for name, src in (("to_q", "q_projection"), ("to_k", "k_projection"), ("to_v", "v_projection"), ("to_g", "gating_query")):
        _put(out, f"{prefix}.{name}.weight", s, f"{key}.{src}.weight")
    _put(out, f"{prefix}.to_q.bias", s, f"{key}.q_projection.bias")
    _need(s, f"{key}.adaptive_zero_init.adaptive_zero_cond.weight", f"{key}.adaptive_zero_init.adaptive_zero_cond.bias")
    out[f"{prefix}.attn_zero.weight"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.weight"]
    out[f"{prefix}.attn_zero.bias"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.bias"]
    _put(out, f"{prefix}.to_out.weight", s, f"{key}.adaptive_zero_init.transition2.weight")
    for name, src in (("q_norm", "query_layer_norm"), ("k_norm", "key_layer_norm")):
        _put(out, f"{prefix}.{name}.weight", s, f"{key}.{src}.weight")
        _put(out, f"{prefix}.{name}.bias", s, f"{key}.{src}.bias")


def dit_variant_stack_from_released(state: State, *, n_head: int, super_block_size: int = 4, prefix: str = "") -> tuple[dict[str, Tensor], dict]:
    """``(engine state_dict, constructor arguments)`` of ``DiTVariantStack`` from a released token ``DiffusionTransformer`` (Boltz-2,
    RoseTTAFold3, Chai-1 layouts; AlphaFold 3's own DiT is the per-block ``DiTBlock``). The options that the weights show
    (``identity_scale``, ``gating``, ``kq_norm``, ``up_gate``, per-block pair projection, ``pair_norm_bias``, sizes) are returned in the
    arguments; ``parallel`` and the eps's are the caller's. The shared pair projection (``super_block_size`` blocks per ``pair_logits_projection``
    matrix) is split per block."""
    s = _sub(state, prefix)
    n_blocks = 1 + max(int(k.split(".")[1]) for k in s if k.startswith("self_attention."))
    per_block = "pair_input_layer_norm.0.weight" in s
    out: dict[str, Tensor] = {}
    identity = None
    for i in range(n_blocks):
        base = f"blocks.{i}"
        attn = f"self_attention.{i}"
        identity = _adaln_from_released(s, f"{attn}.adaptive_layernorm", f"{base}.attn_adaln", out)
        _variant_attention_common(s, attn, base, out)
        _variant_transition_from_released(s, f"transition_block.{i}", base, out)
        norm = f"pair_input_layer_norm.{i}" if per_block else "pair_input_layer_norm"
        _norm(out, f"{base}.pair_norm", s, norm, bias=f"{norm}.bias" in s)
        if per_block:
            proj = s[f"pair_logits_projection.{i}.weight"]
        else:
            sb, off = divmod(i, super_block_size)
            proj = s[f"pair_logits_projection.{sb}.weight"][off * n_head:(off + 1) * n_head]
        out[f"{base}.to_pair_bias.weight"] = proj
    q = s["self_attention.0.q_projection.weight"]
    args = {
        "n_blocks": n_blocks, "d_single": q.shape[0], "d_cond": s["self_attention.0.adaptive_zero_init.adaptive_zero_cond.weight"].shape[1],
        "d_pair": (s["pair_input_layer_norm.0.weight"] if per_block else s["pair_input_layer_norm.weight"]).shape[0],
        "n_head": n_head, "n": s["transition_block.0.adaptive_zero_init.transition2.weight"].shape[1] // q.shape[0],
        "identity_scale": identity, "gating": "self_attention.0.gating_query.weight" in s,
        "kq_norm": "self_attention.0.query_layer_norm.weight" in s, "up_gate": "transition_block.0.a_to_b.weight" in s,
        "pair_norm_bias": ("pair_input_layer_norm.0.bias" if per_block else "pair_input_layer_norm.bias") in s,
    }
    return out, args


def atom_dit_variant_stack_from_released(state: State, *, prefix: str = "") -> tuple[dict[str, Tensor], dict]:
    """``(engine state_dict, constructor arguments)`` of ``AtomDiTVariantStack`` from a released ``DiffusionCrossAttTransformer``
    (``cross_attention.i`` / ``transition_block.i`` / ``pair_input_layer_norm`` / ``pair_logits_projection``). ``parallel``,
    ``key_window`` and the eps's are the caller's."""
    s = _sub(state, prefix)
    n_blocks = 1 + max(int(k.split(".")[1]) for k in s if k.startswith("cross_attention."))
    per_block = "pair_input_layer_norm.0.weight" in s
    out: dict[str, Tensor] = {}
    identity = None
    for i in range(n_blocks):
        base, attn = f"blocks.{i}", f"cross_attention.{i}"
        identity = _adaln_from_released(s, f"{attn}.q_adaptive_layernorm", f"{base}.q_adaln", out)
        _adaln_from_released(s, f"{attn}.k_adaptive_layernorm", f"{base}.k_adaln", out)
        _variant_attention_common(s, attn, base, out)
        _variant_transition_from_released(s, f"transition_block.{i}", base, out)
        if per_block:
            norm = f"pair_input_layer_norm.{i}"
            _norm(out, f"{base}.pair_norm", s, norm, bias=f"{norm}.bias" in s)
            out[f"{base}.to_pair_bias.weight"] = s[f"pair_logits_projection.{i}.weight"]
    if not per_block:
        _norm(out, "pair_norm", s, "pair_input_layer_norm", bias="pair_input_layer_norm.bias" in s)
        out["to_pair_bias.weight"] = s["pair_logits_projection.weight"]
    d = s["cross_attention.0.q_projection.weight"].shape[0]
    pair_norm = "pair_input_layer_norm.0.weight" if per_block else "pair_input_layer_norm.weight"
    args = {
        "n_blocks": n_blocks, "d": d, "d_cond": s["cross_attention.0.adaptive_zero_init.adaptive_zero_cond.weight"].shape[1],
        "d_pair": s[pair_norm].shape[0], "n": s["transition_block.0.adaptive_zero_init.transition2.weight"].shape[1] // d,
        "identity_scale": identity, "gating": "cross_attention.0.gating_query.weight" in s,
        "project_output": "cross_attention.0.adaptive_zero_init.transition2.weight" in s,
        "kq_norm": "cross_attention.0.query_layer_norm.weight" in s, "up_gate": "transition_block.0.a_to_b.weight" in s,
        "per_block_pair_norm": per_block,
        "pair_norm_bias": ("pair_input_layer_norm.0.bias" if per_block else "pair_input_layer_norm.bias") in s,
    }
    return out, args


def set_norm_eps(module: nn.Module, eps: float) -> None:
    """Set ``eps`` on every LayerNorm of ``module`` that still has the default (the released LayerNorms are ``1e-5`` unless a family says
    otherwise; a norm with its own non-default eps, e.g. the grouped OPM ``product_norm`` 0.1, is left alone)."""
    for sub in module.modules():
        if isinstance(sub, nn.LayerNorm) and sub.eps == 1e-5:
            sub.eps = eps


def load(module: nn.Module, converted: Mapping[str, Tensor], *, dtype: torch.dtype | None = None,
         norm_eps: float = 1e-5) -> nn.Module:
    """Strict load of converted tensors into ``module``. The matrices take ``dtype`` (default: the dtype of the released weights, so an
    fp32 checkpoint is not rounded to the module's construction dtype); the fp32-pinned norms stay fp32. ``norm_eps`` is set on the
    LayerNorms that have the default eps."""
    tensors = dict(converted)
    if dtype is None:
        dtype = next(iter(tensors.values())).dtype
    module.to(dtype)
    module.load_state_dict(tensors, strict=True)
    set_norm_eps(module, norm_eps)
    return module
