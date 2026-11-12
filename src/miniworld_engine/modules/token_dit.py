"""The token-track diffusion transformer (AlphaFold 3 Algorithm 23 and the models' variants of it): one class per model, picked by
:class:`TokenDiT`.

    TokenDiT(model="boltz2")                    # -> TokenDiTBoltz2()

Each class is ONE block in the engine layout and hard-codes its model's math (no option flags); :data:`TOKEN_DIT_MODELS` maps the model
names to them. ``attention`` is an :class:`~miniworld_engine.modules.attention.AugmentedAttentionPairBias` (or a model variant of it, or
``BiasOnlyAttention``), ``transition`` a :class:`~miniworld_engine.modules.transition.ConditionedTransition` (or a variant); each part owns
its AdaLN, its pair LayerNorm + projection and its residual. A model's stack is a list of blocks; ``single`` / ``cond`` [A, B, L, d] carry
the augmentation axis, the pair [B, L, L, d_pair] does not::

    sequential    a = attention(a, s, z); a = transition(a, s)                    a' = a + dA(a),  out = a' + dT(a')
    parallel      out = a + attention.delta(a, s, z) + transition.delta(a, s)      (RoseTTAFold3, Chai-1)

    TokenDiTProtenix          sequential, sigmoid AdaLN, gating query, SwiGLU transition; d_cond 384, pair 128
                              (AlphaFold 3, OpenFold3 v0.5 and preview-2, Protenix, Protenix-v2, OpenDDE, IntelliFold2, ESMFold2)
    TokenDiTMiniWorld         the same with the per-head q / k RMSNorm (MiniWorld's QK-norm token DiT)
    TokenDiTMiniWorldBiasOnly the attention without query and key: the logits are the pair bias (MiniWorld's bias-only token DiT)
    TokenDiTBoltz2            TokenDiTProtenix with the up-gate transition (``UpGatedConditionedTransition``); d_cond 768
    TokenDiTRoseTTAFold3      parallel; the affine LayerNorm of the flat q / k and a bias-free query (``FlatQKNormAugmentedAttention``)
    TokenDiTChai1             parallel; identity AdaLN (``IdentityAdaptiveLayerNorm``, epsilon 0.1) in both parts, no
                              gating query (``UngatedAugmentedAttention``); pair 256

A released checkpoint (AlphaFold 3's one shared pair LayerNorm with a projection per super block of four blocks, Chai-1's affine pair
LayerNorm) is converted to these blocks: the shared LayerNorm's weight is replicated per block, the super-block projection sliced into each
block's head rows, and the pair LayerNorm's offset dropped (it adds one constant per head to every key's logit, which the softmax cancels).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F

from miniworld_engine.kernels import _capture
from miniworld_engine.modules import _anthropic
from miniworld_engine.modules.attention import (
    AugmentedAttentionPairBias,
    BiasOnlyAttention,
    FlatQKNormAugmentedAttention,
    UngatedAugmentedAttention,
)
from miniworld_engine.modules.dit_variant_cuda import MasterEntryCast, VariantTokenCuda, fp32_conditioning
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.norm import IdentityAdaptiveLayerNorm
from miniworld_engine.modules.primitives import BackendModule, key_mask, random_input, shared_cond_stream, wide
from miniworld_engine.modules.transition import ConditionedTransition, UpGatedConditionedTransition

_capture.instance_cache_attr("_anthropic_pack")  # the anthropic backend's packed W_qkvg (clear_weight_caches)


def _token_dit_inputs(block, *, length, samples, dtype, device, generator, depth=None, stream=None) -> dict:
    """A token DiT block's measurement inputs: ``single`` / ``cond`` [S, 1, L, d] (S samples on the augmentation axis), ``pair`` [1, L, L, d_pair],
    the key ``mask`` [1, L]. Stream ``token_folds`` (the multi-fold contract, docs/modules.md): ``depth`` = F folds (trunk_seeds of one input),
    ``single`` [S, F, L, d], ``cond`` one per fold [1, F, L, d] expanded over the S samples, ``pair`` [F, L, L, d_pair], one key mask [1, L]
    expanded over the folds (FoldForge's grouped sampler step). A ``*_shared_cond`` stream (``token_single_shared_cond``,
    ``token_folds_shared_cond``): ``cond`` [1, F, L, d] expanded (stride 0) over the S samples, as FoldForge's ``TokenDiTStack`` calls the
    block; the measurers keep it expanded (requirements/README.md)."""
    d_single, d_cond, d_pair = block.transition.d_hidden, block.transition.d_cond, block.attention.ln_pair.normalized_shape[0]
    stream, shared = shared_cond_stream(stream)
    if shared and stream == "token_single":
        return {
            "single": random_input((samples, 1, length, d_single), dtype=dtype, device=device, generator=generator),
            "cond": random_input((1, 1, length, d_cond), dtype=dtype, device=device, generator=generator).expand(samples, -1, -1, -1),
            "pair": random_input((1, length, length, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(1, length, device=device),
        }
    if stream == "token_folds":
        f = depth or 1
        return {
            "single": random_input((samples, f, length, d_single), dtype=dtype, device=device, generator=generator),
            "cond": random_input((1, f, length, d_cond), dtype=dtype, device=device, generator=generator).expand(samples, -1, -1, -1),
            "pair": random_input((f, length, length, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(1, length, device=device).expand(f, length),
        }
    return {
        "single": random_input((samples, 1, length, d_single), dtype=dtype, device=device, generator=generator),
        "cond": random_input((samples, 1, length, d_cond), dtype=dtype, device=device, generator=generator),
        "pair": random_input((1, length, length, d_pair), dtype=dtype, device=device, generator=generator),
        "mask": key_mask(1, length, device=device),
    }


# ================================================================================================= the "anthropic" backend (the release)
# One adapter per release op; the block's statements around the attention core are the ones of ``AugmentedAttentionPairBias.delta``
# (``_release_delta``), the core is the model's release kernel. Every release op here is forward only (no backward).
#
# The glue is laid out the way the kits' own sampler steps lay it out (af3_torch FusedDiT / dtk_modules, Boltz-2 bz_sampler_dit, openfold3
# dit_glue): ONE GEMM against the packed [W_q | W_k | W_v | W_g] (packed once per weight version, not per call) gives the q / k / v / gate
# rows the cores read in place (column views, no cat), and the pair bias comes out of its GEMM head-major [B, H, L, L] (no permute copy;
# a core that takes the key mask inside the bias folds it in place). fp32 products follow the engine's TF32 rule
# (``torch.backends.cuda.matmul.allow_tf32``) through each kit's own precision word (dtk ``in_prec`` tf32 / ieee, Boltz-2 tf32 / ieee).


def _grad_reason(module: torch.nn.Module, *tensors) -> str | None:
    if torch.is_grad_enabled() and (
        any(p.requires_grad for p in module.parameters()) or any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)
    ):
        return "the release op has no backward: call it under torch.no_grad() / inference_mode() (eval only)"
    return None


def _core_dtype(single: torch.Tensor) -> torch.dtype:
    """The dtype the attention core sees: the activation's, or the autocast dtype (the master-weight layout)."""
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else single.dtype


def _tf32() -> bool:
    """The engine's fp32 rule for this call: TF32 products allowed (the measurers allow them for every backend)."""
    return bool(torch.backends.cuda.matmul.allow_tf32)


def _token_mask_reason(mask: torch.Tensor | None, single: torch.Tensor) -> str | None:
    if mask is None:
        return None
    if mask.ndim not in (2, 3) or tuple(mask.shape[-2:]) != tuple(single.shape[1:3]):
        return f"mask must be [B, L] or [A, B, L], got {tuple(mask.shape)}"
    if not bool(mask.any(-1).all()):
        return "a (sample, batch) row with no valid key (the release kernels average such a row, the reference gives 0)"
    return None


def _packed_linear(att, names: tuple[str, ...], x: torch.Tensor) -> torch.Tensor:
    """``cat([att.<name>(x) for name in names], -1)`` as ONE GEMM against the packed weight (and bias, zeros where a part has none), packed
    once per parameter version and cached on ``att`` (the kits pack W_qkvg once: af3_torch ``dtk_modules`` FusedDiT, Boltz-2
    ``bz_sampler_dit._State``, openfold3 ``dit_glue._weights``)."""
    mods = [getattr(att, n) for n in names]
    params = [p for m in mods for p in (m.weight, m.bias) if p is not None]
    if torch.is_grad_enabled() and any(p.requires_grad for p in params):  # a gradient: the pack is part of the graph, never cached
        w = torch.cat([m.weight for m in mods], 0)
        b = None
        if any(m.bias is not None for m in mods):
            b = torch.cat([m.bias if m.bias is not None else m.weight.new_zeros(m.weight.shape[0]) for m in mods], 0)
        if not torch.is_autocast_enabled(x.device.type) and w.dtype != x.dtype:
            w, b = w.to(x.dtype), None if b is None else b.to(x.dtype)
        return F.linear(x, w, b)
    key = (names, tuple((p.data_ptr(), p._version, p.dtype) for p in params))
    hit = att.__dict__.get("_anthropic_pack")
    if hit is None or hit[0] != key:
        with torch.no_grad():
            w = torch.cat([m.weight for m in mods], 0).contiguous()
            b = None
            if any(m.bias is not None for m in mods):
                b = torch.cat([m.bias if m.bias is not None else m.weight.new_zeros(m.weight.shape[0]) for m in mods], 0).contiguous()
        hit = (key, w, b)
        att.__dict__["_anthropic_pack"] = hit
    _, w, b = hit
    if not torch.is_autocast_enabled(x.device.type) and w.dtype != x.dtype:
        w, b = w.to(x.dtype), None if b is None else b.to(x.dtype)
    return F.linear(x, w, b)


def _bias_planes(att, pair: torch.Tensor, extra_pair_bias: torch.Tensor | None) -> torch.Tensor:
    """``to_bias(ln_pair(pair))`` (+ ``extra_pair_bias``) head-major [B, H, L, L], contiguous: the bias GEMM written as ``W z^T`` so its
    output is the planes the cores read (the kits' head-major bias hoist), no [B, L, L, H] -> [B, H, L, L] copy."""
    z = att.ln_pair(pair)
    bsz, n = z.shape[0], z.shape[1]
    w = att.to_bias.weight
    if not torch.is_autocast_enabled(z.device.type) and w.dtype != z.dtype:
        w = w.to(z.dtype)
    planes = torch.matmul(w, z.reshape(bsz, n * n, z.shape[-1]).transpose(1, 2)).view(bsz, -1, n, n)  # [B, H, L, L]
    if att.to_bias.bias is not None:
        planes = planes + att.to_bias.bias.to(planes.dtype)[None, :, None, None]
    if extra_pair_bias is not None:
        planes = planes + extra_pair_bias.to(planes.dtype)[None, None]
    return planes


def _cueq_bias_planes(att, pair: torch.Tensor, extra_pair_bias: torch.Tensor | None) -> torch.Tensor:
    """:func:`_bias_planes` on cuEquivariance's ``pair_repr_to_attention_bias`` (LayerNorm -> Linear, forward and backward; head-major
    [B, H, L, L]): the LayerNorm affine handed as a pair (``ln_pair`` has no offset: zeros) and every operand in the pair's dtype -- the
    op's homogeneous form; a partial affine or mixed dtypes would send it to its own PyTorch formula. No mask here: the core masks."""
    import importlib

    op = importlib.import_module("cuequivariance_ops_torch").pair_repr_to_attention_bias
    dt = pair.dtype
    w_ln = att.ln_pair.weight.to(dt)
    b_ln = att.ln_pair.bias.to(dt) if getattr(att.ln_pair, "bias", None) is not None else torch.zeros_like(w_ln)
    b_proj = att.to_bias.bias.to(dt) if att.to_bias.bias is not None else None
    with torch.autocast(pair.device.type, enabled=False):
        planes, _ = op(pair, None, att.to_bias.weight.to(dt), b_proj, w_ln, b_ln, att.to_bias.weight.shape[0], 1, att.ln_pair.eps)
    if extra_pair_bias is not None:
        planes = planes + extra_pair_bias.to(planes.dtype)[None, None]
    return planes


def _core_torch(q, k, v, bias, mask, gate):
    """The reference's core statements (scores and softmax in at least fp32) on head-major bias planes; the gate fused as the caller asks."""
    a, b, n, h, d = q.shape
    w = wide(q)
    with torch.autocast(q.device.type, enabled=False):
        s = torch.einsum("abqhd,abkhd->abhqk", q.to(w), k.to(w)) * d**-0.5 + bias.to(w)[None]
        if mask is not None:
            s = s.masked_fill(~mask.bool()[:, :, None, None, :], float("-inf"))
        o = torch.einsum("abhqk,abkhd->abqhd", torch.softmax(s, -1), v.to(w)).reshape(a, b, n, h * d)
        if gate is not None:
            o = sigmoid_gate(gate.to(w), o)
    return o.to(q.dtype)


def _release_delta(att, single, cond, pair, mask, extra_pair_bias, core, *, fused_gate: bool = False, bias_fn=None) -> torch.Tensor:
    """``AugmentedAttentionPairBias.delta`` (no residual) with ``core(q, k, v, bias, mask, gate)`` -> [A, B, L, H * D] as its attention core:
    q / k / v [A, B, L, H, D] (column views of one packed GEMM), bias [B, H, L, L] head-major (+ ``extra_pair_bias``; the core may fold the
    mask into it in place), mask [A, B, L] or None, gate the gating query's logits ``to_gate(a)`` [A, B, L, H * D] when the release op fuses
    ``sigmoid(gate) *`` (``fused_gate``; then only ``to_out`` follows), else None."""
    in_dtype = single.dtype
    x = att.ada_ln_in(single, cond)
    names = ("to_query", "to_key", "to_value") + (("to_gate",) if fused_gate else ())
    rows = _packed_linear(att, names, x)  # [A, B, L, (3 | 4) C]
    c = att.to_query.weight.shape[0]
    query, key = att.flat_qk_norm(rows[..., :c], rows[..., c : 2 * c])
    value = rows[..., 2 * c : 3 * c]
    a = query.shape[0]
    h = att.n_head
    d = c // h
    query, key, value = (t.unflatten(-1, (h, d)) for t in (query, key, value))
    bias = (bias_fn or _bias_planes)(att, pair, extra_pair_bias)
    if mask is not None and mask.ndim == 2:
        mask = mask[None].expand(a, -1, -1)
    gate = rows[..., 3 * c :] if fused_gate else None
    out = core(query, key, value, bias, mask, gate).to(in_dtype)
    out = att.to_out(out) if fused_gate else att.gate_project(x, out)
    return sigmoid_gate(att.to_scale(cond), out)


def _shared_mask_rows(mask: torch.Tensor | None) -> torch.Tensor | None:
    """The [B, L] key mask an [A, B, L] mask holds for every sample (metadata only: A == 1 or stride 0), else None."""
    if mask is None:
        return None
    return mask[0] if (mask.shape[0] == 1 or mask.stride(0) == 0) else None


def _fold_mask(bias: torch.Tensor, mask_rows: torch.Tensor | None, value: float) -> torch.Tensor:
    """``bias`` [B, H, L, L] with ``value`` on every masked key, in place (the planes are this call's own)."""
    if mask_rows is not None:
        bias.masked_fill_(~mask_rows.bool()[:, None, None, :], value)
    return bias


def _core_af3_apb(q, k, v, bias, mask, gate):
    """af3_torch's token core in 16 bit: ``opt_core.attn.apb_core.pair_bias_attention`` (the apb_attn Triton kernel the kit's DTK step and
    its ``apb`` lever call, openfold3's ``dit_glue`` core), one launch per batch row for its A samples sharing the row's head-major bias,
    the key mask by name and the gate fused (as ``dit_glue`` passes them)."""
    apb = _anthropic.load("opt_core.attn.apb_core")
    outs = []
    for j in range(q.shape[1]):
        km = None if mask is None else mask[:, j]
        outs.append(
            apb.pair_bias_attention(
                q[:, j].transpose(1, 2),
                k[:, j].transpose(1, 2),
                v[:, j].transpose(1, 2),
                bias[j],
                km,
                gate=None if gate is None else gate[:, j],
                layout="shnd",
            )
        )  # [A, L, H * D]
    return torch.stack(outs, 1)


def _core_dtk_flash(q, k, v, bias, mask, gate, *, dtype=None):
    """``opt_core.kernels.dtk_kernels.flash_bias_attn``, one [H, L, D] problem per (sample, batch row): af3_torch's DTK per-sample core
    (``dtk_modules`` FusedDiT's last arm), and the core RoseTTAFold3's kit calls (``rosettafold3_opt.dtk.attn``: one launch per sample, the
    bias in the activation dtype, no mask). fp32 products: ``in_prec`` tf32 under the engine's TF32 rule, else ieee (the kit's default)."""
    dk = _anthropic.load("opt_core.kernels.dtk_kernels")
    a, b, n, h, d = q.shape
    dt = dtype or q.dtype
    prec = "tf32" if dt == torch.float32 and _tf32() else "ieee"
    out = torch.empty(a, b, n, h * d, dtype=dt, device=q.device)
    for j in range(b):
        bh = bias[j].to(dt)  # [H, L, L], head-major already
        for i in range(a):
            km = None if mask is None else mask[i, j].to(torch.int32)
            dk.flash_bias_attn(
                q[i, j].to(dt).transpose(0, 1),
                k[i, j].to(dt).transpose(0, 1),
                v[i, j].to(dt).transpose(0, 1),
                bias=bh,
                key_mask=km,
                gate=None if gate is None else gate[i, j],
                out=out[i, j],
                out_dtype=dt,
                scale=d**-0.5,
                in_prec=prec,
            )
    return out


#: The tier word the token cores ask opt_core's pair-bias provider with: the kits' speed mode (protenix_v2 / opendde / af3_torch / Chai-1
#: ``fast``); the selection is the graphed column (the measurer and the kits' samplers replay CUDA graphs).
_PROVIDER_WORD = "fast"


def _provider_selection(n: int, samples: int, d: int, h: int, dtype: torch.dtype, device, cell: str = "dit_h16d48"):
    """``(Selection, None)`` or ``(None, reason)``: the row opt_core's pair-bias provider (``opt_core.kernels.apb``) selects for the call."""
    provider = _anthropic.load("opt_core.kernels.apb")
    cc = tuple(torch.cuda.get_device_capability(device))
    return _anthropic.kernel_selection(
        provider,
        cc,
        dtype,
        cell,
        int(n),
        word=_PROVIDER_WORD,
        samples=int(samples),
        timing="graphed",
        capture=True,
        head_dim=int(d),
        heads=int(h),
    )


def _core_provider(q, k, v, bias, mask_rows, gate, sel, *, cell: str = "dit_h16d48"):
    """``opt_core.kernels.apb.pair_bias_attention`` with the selection ``sel``: the A samples of a batch row in one call (the kits' DiT
    site: protenix_v2 ``apb_core._make_dit_forward``, opendde ``odde_apb_bind.dit_views``, af3_torch ``FusedDiT.forward_batched``), q / k /
    v / gate the [A, L, H, D] column views, the row's [1, H, L, L] bias with the key mask folded in (-1e9, af3_torch's fold)."""
    provider = _anthropic.load("opt_core.kernels.apb")
    a, b, n, h, d = q.shape
    _fold_mask(bias, mask_rows, -1e9)
    out = torch.empty(a, b, n, h * d, dtype=q.dtype, device=q.device)
    for j in range(b):
        g = None if gate is None else gate[:, j].unflatten(-1, (h, d))
        o, _ = provider.pair_bias_attention(
            q[:, j], k[:, j], v[:, j], bias[j][None], None, g, selection=sel, layout="snhd", cell=cell, capture=True
        )
        out[:, j] = o.reshape(a, n, h * d)
    return out


# ================================================================================================= the "cuda" backend (the engine's H100 kernels)
# The fused H100 block paths of kernels/integrations (token_dit: inference; token_dit_train: the autograd Function; bias_only_dit /
# bias_only_dit_train likewise for the bias-only block). Each returns the whole block (both residuals) and reads the block's own
# parameters by name. Their gates (``serves``) are the kernels' ranges; they still test the attributes the removed dispatch set
# (``implementation`` / ``attention._backend``), which :func:`_kernel_view` supplies with the engine-kernel values.


class _View:
    """``module`` with some attributes overridden (read-only forwarding of everything else, parameters included)."""

    # The overrides are the view's own attributes, so only the other names reach ``__getattr__``; no ``object.__getattribute__`` (Dynamo
    # cannot trace it: a graph break per block call inside FoldForge's compiled sampler step).
    def __init__(self, module, **over):
        self._module = module
        for name, value in over.items():
            setattr(self, name, value)

    def __getattr__(self, name):
        if name == "_module":
            raise AttributeError(name)
        return getattr(self._module, name)


def _kernel_view(block):
    return _View(block, implementation="miniworld", attention=_View(block.attention, _backend="portable"))


# ------------------------------------------------------------------------------------------------------------ per-block call cache
# A sampler calls every block with the same shapes tens of times (FoldForge: 2 eager warm-up steps + a capture per forward), and the
# block's "cuda" decision (``_cuda_route``) and the fused step's per-call constants (``token_dit.infer_args``: the leaves, the options)
# are the same at every such call. They were re-derived at each one -- ~25 parameter lookups and two range checks through the kernel
# view, about as much host time as the step's own launches. Kept per block (``block.__dict__``), keyed by the call's metadata (shapes,
# strides, dtypes, devices: no device read), for grad-free calls only (a training call's route reads ``requires_grad``). Stale when:
#   * any module registers a parameter or a submodule (a replaced weight / part: the global registration hooks below bump _STRUCTURE);
#   * the block is converted (``.to`` / ``.float`` / ``set_master_weights``: ``TokenDiTProtenix._apply`` drops it);
#   * ``kernels._capture.clear_weight_caches(block)`` (a registered instance cache attribute).
# A plain attribute edit after the first call (a LayerNorm's eps, ``use_qk_norm``) is not seen: call ``clear_weight_caches(block)``.
# Under Dynamo nothing is cached: the trace itself is the cache (its guards are the key).
_CALL_CACHE = _capture.instance_cache_attr("_cuda_call_cache")
_STRUCTURE = [0]


def _structure_changed(*_args) -> None:
    _STRUCTURE[0] += 1


torch.nn.modules.module.register_module_parameter_registration_hook(_structure_changed)
torch.nn.modules.module.register_module_module_registration_hook(_structure_changed)


def _call_cache(block) -> dict | None:
    """The block's call cache (see above), or None when this call must not use one (grad mode, Dynamo tracing)."""
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return None
    entry = block.__dict__.get(_CALL_CACHE)
    if entry is None or entry[0] != _STRUCTURE[0]:
        entry = block.__dict__[_CALL_CACHE] = (_STRUCTURE[0], {})
    return entry[1]


def _meta(t: torch.Tensor | None):
    return None if t is None else (t.shape, t.stride(), t.dtype, t.device)


def _h100_reason(single: torch.Tensor) -> str | None:
    if not single.is_cuda:
        return "CUDA tensors only"
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return "the H100 (sm_90) kernels only"
    return None


def _a100(single: torch.Tensor) -> bool:
    """A CUDA tensor on an A100 (sm_80): the card of the ``"cuda"`` backend's sm_80 paths (``kernels/integrations/*_sm80.py``)."""
    return single.is_cuda and torch.cuda.get_device_capability(single.device) == (8, 0)


def _a100_dtype(
    single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor | None, compute_dtype
) -> tuple[torch.dtype | None, str | None]:
    """(the dtype the A100 kernels run the block in, None) or (None, why not): bf16 activations (bf16 or fp32 master parameters, with or
    without bf16 autocast) or fp32 activations without autocast (TF32 tensor cores)."""
    dt = single.dtype
    if dt not in (torch.bfloat16, torch.float32) or cond.dtype != dt or (pair is not None and pair.dtype != dt):
        return None, f"A100: dtypes {single.dtype} / {cond.dtype} / {None if pair is None else pair.dtype} (all bfloat16 or all float32)"
    if torch.is_autocast_enabled("cuda") and (torch.get_autocast_dtype("cuda") != torch.bfloat16 or dt != torch.bfloat16):
        return None, "A100: autocast: bfloat16 activations under bfloat16 autocast only (float32 activations run without autocast, on TF32)"
    if compute_dtype not in (None, dt):
        return None, f"A100: compute_dtype {compute_dtype} (the sm_80 kernels run in the activations' dtype)"
    return dt, None


def _wants_grad(module: torch.nn.Module, *tensors) -> bool:
    return torch.is_grad_enabled() and (
        any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors) or any(p.requires_grad for p in module.parameters())
    )


def _shared_key_mask(mask: torch.Tensor | None) -> tuple[torch.Tensor | None, bool]:
    """(the [B, L] key mask the samples share, True), or (``mask``, False) for an [A, B, L] mask that may differ per sample. An [A, B, L]
    mask is shared when A == 1 or it is a [B, L] mask expanded over A (stride 0): metadata only, no device read."""
    if mask is None or mask.ndim == 2:
        return mask, True
    if mask.ndim == 3 and (mask.shape[0] == 1 or mask.stride(0) == 0):
        return mask[0], True
    return mask, False


class TokenDiTProtenix(BackendModule):
    """AF3 Alg. 23 in the engine layout: pair-bias attention, then the conditioned transition::

        a = AugmentedAttentionPairBias(a, s, z, mask)     # a + attention(a, s, z)
        a = ConditionedTransition(a, s)                  # a + transition(a, s)

    ``forward(single, cond, pair, mask)`` -> ``single``'s shape. ``single`` and ``cond`` carry the augmentation axis (``A, B, L, d``);
    ``pair`` does not (``B, L, L, d_pair``), because the pair bias is shared across augmentations. ``B`` is the multi-fold axis: B folds
    (trunk_seeds of one input) each with its own pair and conditioning, one key mask expanded over B (docs/modules.md, multi-fold contract). Both parts build their own
    ``AdaptiveLayerNorm`` from their ``(d_hidden, d_cond)``. At atom widths (128 / 128 / pair 16 / 4 heads) it is also the dense atom DiT.

    AlphaFold 3's own token DiT is this block (registered as ``"af3"``).

    ``backend="anthropic"``: the release's attention core (bf16: ``opt_core.attn.apb_core``; fp32: opt_core's pair-bias provider
    ``opt_core.kernels.apb`` by the tier word "fast", else ``dtk_kernels.flash_bias_attn`` per sample),
    the rest of the block as ``"pytorch"``; eval only, CUDA H100, head dim <= 64 (:meth:`_supports_anthropic`).
    """

    version = 3  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cueq", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,
        "cuda:a100_sm80": 3,
        "triton:a100_sm80": 2,
        "triton:h100_sm90": 2,
        "cuda:h100_sm90": 2,
    }  # cuda:h100_sm90 2 (2026-11-12): bf16 inference folds the conditioning into one grouped GEMM (CondFold), 128 x 128 SwiGLU tiles at small M, PDL row passes (outputs change: fp32 tables; every subclass); 2026-11-10: kits; sm_80 paths; 3: transition weight casts read by captures; triton:a100_sm80 / triton:h100_sm90 2 (2026-11-11): the Triton attention core returns its bias gradient head-major on the A100 / H100 (every subclass)

    #: The q / k RMSNorm of the attention (:class:`TokenDiTMiniWorld`).
    qk_norm: bool = False
    #: The parallel schedule ``a + dA(a) + dT(a)`` (:class:`TokenDiTRoseTTAFold3`, :class:`TokenDiTChai1`).
    parallel: bool = False
    #: The parts (a model's variant of each).
    attention_part: type[AugmentedAttentionPairBias] = AugmentedAttentionPairBias
    transition_part: type[ConditionedTransition] = ConditionedTransition

    def __init__(
        self,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 128,
        n_head: int = 16,
        n: int = 2,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        qk: dict[str, Any] = {"use_qk_norm": True} if self.qk_norm else {}
        self.attention = self.attention_part(d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head, **qk)
        self.transition = self.transition_part(d_hidden=d_single, d_cond=d_cond, n=n)
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """AF3 Alg. 23; each part returns its residual output (stream in, stream out).

        ``mask`` is a key mask, ``[B, L]`` or ``[A, B, L]``. ``extra_pair_bias`` ``[L, L]`` is added to every head's attention logits, on
        top of the pair's own bias (``AugmentedAttentionPairBias``'s ``extra_pair_bias``). ``compute_dtype``: the precision of the attention
        core."""
        kw: dict[str, Any] = {"compute_dtype": compute_dtype} if compute_dtype is not None else {}
        if extra_pair_bias is not None:
            kw["extra_pair_bias"] = extra_pair_bias
        if self.parallel:
            return single + self.attention.delta(single, cond, pair, mask, **kw) + self.transition.delta(single, cond)
        single = self.attention(single, cond, pair, mask, **kw)
        return self.transition(single, cond)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return _token_dit_inputs(
            self, length=length, samples=samples, dtype=dtype, device=device, generator=generator, depth=depth, stream=stream
        )

    # ------------------------------------------------------------------------------------------------------------ "triton"
    # The block's own statements with the attention core on the Triton pair-bias attention kernel (forward and backward): the block's
    # backend is set on its attention part (``set_backend``); the transition and every projection are the module's statements.
    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        self.attention.set_backend("triton" if backend == "triton" else "pytorch")
        return self

    def _supports_triton(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        return self.attention._supports_triton(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

    def _forward_triton(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        return self._forward_pytorch(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

    # ------------------------------------------------------------------------------------------------------------ "cuda"
    # H100: no gradient -> kernels/integrations/token_dit (the fused inference block: hoisted pair bias, CUDA rows, cuBLAS, the sm_90a
    # gated attention core); a gradient -> kernels/integrations/token_dit_train (one autograd Function, forward + backward on CUDA rows,
    # cuBLAS and the sm_90a cores). Range = the two gates: bf16 / fp32, [A, 1, L, 768] (B > 1 trunk_seeds: bf16 inference only, one
    # pair per trunk_seed, the key mask expanded over B), 16 heads x 48, cond 384 or 768, pair 128 / 256 / 512 (the fused blocks' row
    # kernels and weight packs are built per cond / pair width), transition n = 2, LayerNorm eps 1e-5,
    # a key mask shared by the samples; inference L a multiple of 128, training any L (padded to 128).
    def _apply(self, fn, *args, **kwargs):
        self.__dict__.pop(_CALL_CACHE, None)  # a conversion can change the parameters' dtype / device (the call cache's constants)
        return super()._apply(fn, *args, **kwargs)

    def _cuda_route(self, single, cond, pair, mask, compute_dtype, extra_pair_bias) -> tuple[str | None, Any]:
        """("infer" | "train", the [B, L] key mask), ("a100", None) or (None, why not). A grad-free H100 decision is kept in the block's
        call cache (``_call_cache``), keyed by the call's metadata."""
        cache = _call_cache(self)
        if cache is None:
            return self._cuda_route_of(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        key = ("route", _meta(single), _meta(cond), _meta(pair), _meta(mask), compute_dtype, _meta(extra_pair_bias))
        hit = cache.get(key)
        if hit is not None:
            return ("infer", _shared_key_mask(mask)[0]) if hit == "infer" else (None, hit[1])
        route, out = self._cuda_route_of(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        if _h100_reason(single) is None:  # the H100 route only (the A100 route also reads the autocast state)
            if route == "infer":
                cache[key] = "infer"
            elif route is None:
                cache[key] = (None, out)
        return route, out

    def _cuda_route_of(self, single, cond, pair, mask, compute_dtype, extra_pair_bias) -> tuple[str | None, Any]:
        if _a100(single):
            return self._a100_route(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        why = _h100_reason(single)
        if why is not None:
            return None, why
        km, shared = _shared_key_mask(mask)
        if not shared:
            return None, "a key mask that differs across the samples (the fused blocks take one [B, L] mask)"
        view = _kernel_view(self)
        if _wants_grad(self, single, cond, pair, extra_pair_bias):
            from miniworld_engine.kernels.integrations import token_dit_train

            if token_dit_train.serves(view, single, cond, pair, km, compute_dtype, extra_pair_bias):
                return "train", km
            return None, (
                "training: outside the fused H100 training block's range (bf16 / fp32, [A, 1, L, d] with heads x head width "
                "16 x 48 / 24 x 32 / 12 x 64 (d 768) or 16 x 64 (d 1024), cond 384 or 768, pair 128 / 256 / 512, n = 2, key mask [1, L], eps 1e-5)"
            )
        if single.shape[1] > 1 and km is not None and km.stride(0) != 0:
            return None, "B > 1 trunk_seeds take one key mask expanded over B (stride 0)"
        from miniworld_engine.kernels.integrations import token_dit

        if token_dit.serves(view, single, cond, pair, compute_dtype, extra_pair_bias):
            return "infer", km
        return None, (
            "inference: outside the fused H100 inference block's range (bf16 / fp32, [A, 1, L, d] with L a multiple of 128 and "
            "heads x head width 16 x 48 / 24 x 32 / 12 x 64 (d 768) or 16 x 64 (d 1024), cond 384 or 768, pair 128 / 256 / 512, n = 2, eps 1e-5)"
        )

    def _supports_cuda(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        route, why = self._cuda_route(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        return why if route is None else None

    def _forward_cuda(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        route, km = self._cuda_route(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        if route == "a100":
            return self._forward_a100(single, cond, pair, mask)
        view = _kernel_view(self)
        if route == "train":
            from miniworld_engine.kernels.integrations import token_dit_train

            return token_dit_train.block(view, single, cond, pair, km, compute_dtype, extra_pair_bias)
        from miniworld_engine.kernels.integrations import token_dit

        cache = _call_cache(self)
        if cache is None:
            return token_dit.update(view, single, cond, pair, km, extra_pair_bias)
        args = cache.get(("args", single.dtype))
        if args is None:
            args = cache["args", single.dtype] = token_dit.infer_args(view, single.dtype)
        return token_dit.update(view, single, cond, pair, km, extra_pair_bias, args)

    # A100 (sm_80): the block's two parts on their sm_80 paths, forward and backward. The attention: kernels/integrations/augattn_sm80 (the
    # module's AdaLN, then one autograd Function from the AdaLN output to the residual: cuBLAS projections, the pair bias kernels, the
    # mma.sync core, the gates; bf16 or the TF32 core for fp32; the per-head q / k RMSNorm of TokenDiTMiniWorld on the q / k columns of the
    # projection between the GEMM and the core, forward and backward). The transition:
    # kernels/integrations/conditioned_transition_sm80 (AdaLN rows + the SwiGLU tail, bf16 or TF32). Range = their gates: bf16 / master /
    # fp32, head dim 32, 48 or 64, LayerNorm eps 1e-5, a bool key mask [B, L] / [A, B, L] or none, no extra_pair_bias.
    def _a100_route(self, single, cond, pair, mask, compute_dtype, extra_pair_bias):
        from miniworld_engine.kernels.integrations import augattn_sm80
        from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct_sm80

        if extra_pair_bias is not None:
            return None, "A100: extra_pair_bias (not an input of the sm_80 attention kernels)"
        dt, why = _a100_dtype(single, cond, pair, compute_dtype)
        if why is not None:
            return None, why
        if single.ndim != 4 or cond.ndim != 4:
            return None, f"A100: single / cond must be [A, B, L, d], got {tuple(single.shape)} / {tuple(cond.shape)}"
        if not ct_sm80.serves(self.transition, single, cond, cast(torch.dtype, dt), _wants_grad(self, single, cond, pair)):
            return None, (
                "A100: outside the sm_80 ConditionedTransition kernels' range (bf16 / fp32, widths of their table, sigmoid AdaLN "
                "without a shift bias, bias-free SwiGLU)"
            )
        if not augattn_sm80.serves(self.attention, single, pair, mask, compute_dtype, cond=cond):
            return None, (
                "A100: outside the sm_80 AugmentedAttentionPairBias kernels' range (bf16 / fp32 with weights in that dtype or "
                "fp32, head dim 32, 48 or 64, pair LayerNorm eps 1e-5, a bool key mask [B, L] / [A, B, L] or none)"
            )
        return "a100", None

    def _forward_a100(self, single, cond, pair, mask):
        from miniworld_engine.kernels.integrations import augattn_sm80
        from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct_sm80

        grad = _wants_grad(self, single, cond, pair)
        single = augattn_sm80.forward(self.attention, single, cond, pair, mask)
        with torch.autocast("cuda", enabled=False):
            return ct_sm80.update(self.transition, single, cond, True, single.dtype, grad)

    # ------------------------------------------------------------------------------------------------------------ "anthropic"
    #: the release op this class's attention core runs on, and whether that op fuses the gating query's sigmoid
    #: AF3 / OpenFold3 / Protenix / OpenDDE / IntelliFold2 / ESMFold2 (one block for all of them):
    #:   16 bit  opt_core.attn.apb_core.pair_bias_attention (af3_torch's DTK apb lever, openfold3's dit_glue core): the A samples of a batch
    #:           row in one launch, key mask by name, gate fused;
    #:   fp32    opt_core.kernels.apb.pair_bias_attention by the tier word "fast" (the fp32 DiT site of protenix_v2 apb_core / opendde
    #:           odde_apb_bind / af3_torch FusedDiT): the row the provider selects for the cell, the key mask folded into the bias; a call the
    #:           provider refuses (a key mask that differs per sample, a head layout without a cell) takes af3_torch FusedDiT's last arm, the
    #:           per-sample dtk_kernels.flash_bias_attn.
    _release_module = "opt_core.attn.apb_core"
    _release_path = "common/opt_core"
    _release_fused_gate = True

    def _release_core(self, q, k, v, bias, mask, gate):
        if q.dtype in (torch.bfloat16, torch.float16):
            return _core_af3_apb(q, k, v, bias, mask, gate)
        a, _, n, h, d = q.shape
        rows = _shared_mask_rows(mask)
        sel = _provider_selection(n, a, d, h, q.dtype, q.device)[0] if mask is None or rows is not None else None
        if sel is not None:
            return _core_provider(q, k, v, bias, rows, gate, sel)
        return _core_dtk_flash(q, k, v, bias, mask, gate)

    def _release_range(self, single, cond, pair, mask) -> str | None:
        """The class's own limits beyond the shared ones (dtype, mask, head width)."""
        if _core_dtype(single) not in (torch.bfloat16, torch.float32):
            return f"core dtype {_core_dtype(single)} (bfloat16 / float32 only)"
        if single.shape[-1] // self.attention.n_head > 64:
            return "head dim > 64 (the release kernels' limit)"
        return _anthropic.missing("opt_core.kernels.dtk_kernels") or _anthropic.missing("opt_core.kernels.apb")

    def _supports_anthropic(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        why = _anthropic.unavailable(self._release_module, single.device, path=self._release_path)
        if why is not None:
            return why
        if compute_dtype is not None:
            return "compute_dtype (the release core runs in the activations' dtype)"
        if getattr(self.attention, "use_qk_norm", False):
            return "per-head q / k RMSNorm (no release op)"
        return (
            _grad_reason(self, single, cond, pair, extra_pair_bias)
            or _token_mask_reason(mask, single)
            or self._release_range(single, cond, pair, mask)
        )

    def _forward_anthropic(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        """``_forward_pytorch`` with the attention core on the release op (``_release_core``)."""
        delta = _release_delta(
            self.attention, single, cond, pair, mask, extra_pair_bias, self._release_core, fused_gate=self._release_fused_gate
        )
        if self.parallel:
            return single + delta + self.transition.delta(single, cond)
        return self.transition(single + delta, cond)

    # ------------------------------------------------------------------------------------------------------------ "cueq"
    # The pair-bias producer ``to_bias(ln_pair(pair))`` on cuEquivariance's ``pair_repr_to_attention_bias`` (LayerNorm -> Linear, Triton,
    # forward and backward), head-major [B, H, L, L]; everything else is the module's statements (the attention core the reference's,
    # scores and softmax in fp32). cuEquivariance's ``attention_pair_bias`` cannot take the block: it applies its own LayerNorm to the
    # single, not the AdaLN. Eval and train.
    def _supports_cueq(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        import importlib

        from miniworld_engine.modules import _cueq

        why = _cueq.unavailable(single.device)
        if why is not None:
            return why
        if compute_dtype is not None:
            return "compute_dtype (the module's core runs in the activations' dtype here)"
        if getattr(self.attention, "use_qk_norm", False):
            return "per-head q / k RMSNorm (not this backend's form)"
        if mask is not None and (mask.ndim not in (2, 3) or tuple(mask.shape[-2:]) != tuple(single.shape[1:3])):
            return f"mask must be [B, L] or [A, B, L], got {tuple(mask.shape)}"
        if pair.dtype not in (torch.bfloat16, torch.float16, torch.float32) or pair.ndim != 4:
            return f"pair {pair.dtype} {tuple(pair.shape)} (a [B, L, L, d_pair] bf16 / fp16 / fp32 pair)"
        threshold = int(
            importlib.import_module("cuequivariance_ops_torch.pair_repr_to_attention_bias").CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD
        )
        if pair.shape[1] * pair.shape[2] <= threshold**2:
            return f"L <= {threshold} (the op runs its own PyTorch formula there)"
        return None

    def _forward_cueq(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        gated = hasattr(self.attention, "to_gate") and self._release_fused_gate
        delta = _release_delta(
            self.attention, single, cond, pair, mask, extra_pair_bias, _core_torch, fused_gate=gated, bias_fn=_cueq_bias_planes
        )
        if self.parallel:
            return single + delta + self.transition.delta(single, cond)
        return self.transition(single + delta, cond)


class TokenDiTMiniWorld(TokenDiTProtenix):
    """MiniWorld's token DiT block: :class:`TokenDiTProtenix` with the per-head q / k RMSNorm (``attention.norm_query`` /
    ``attention.norm_key``). No ``"anthropic"`` backend: no release folder has the QK-RMSNorm token DiT."""

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    #: 4 (2026-11-11): A100 on the token DiT's sm_80 attention (augattn_sm80, the per-head q / k RMSNorm inside it) and the sm_80 transition;
    #: was the general composition with a materialised-score attention (a100_families.token)
    backend_versions: ClassVar[dict[str, int]] = {**TokenDiTProtenix.backend_versions, "cuda:a100_sm80": 4}

    backends = ("pytorch", "cuda", "triton")
    qk_norm = True


class TokenDiTBoltz2(VariantTokenCuda, TokenDiTProtenix):
    """Boltz-2's token DiT block: :class:`TokenDiTProtenix` with the up-gate transition ``silu(W_a h) (W_b h) (W_u h)``
    (``transition.up``); the conditioning is 768 wide."""

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results
    #: 2026-11-10: H100 bf16 / fp32 inference with a conditioning per sample in one call (was one kernel call per sample)
    #: 3 (2026-11-10): H100 fp32 training / inference bias made key-permuted by the pair LayerNorm (no gathers), the pair passed as a view
    #: 4 (2026-11-10): H100 on the one token DiT algorithm (TokenDiTProtenix's cuda route with this block's options), eval and train
    #: 5 (2026-11-12): bf16 inference on the folded, grouped conditioning GEMM (TokenDiTProtenix cuda:h100_sm90 2); Boltz-2: the up-gate
    #: transition in the SwiGLU GEMM's epilogue
    backend_versions: ClassVar[dict[str, int]] = {**TokenDiTProtenix.backend_versions, "cuda:h100_sm90": 5}

    backends = (
        "pytorch",
        "anthropic",
        "cueq",
        "cuda",
        "triton",
    )  # "cuda": H100 the one token DiT algorithm (its options), A100 VariantTokenCuda

    transition_part = UpGatedConditionedTransition

    def __init__(
        self, d_single: int = 768, d_cond: int = 768, d_pair: int = 128, n_head: int = 16, n: int = 2, *, backend: str = "pytorch"
    ) -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, backend=backend)

    # "anthropic": Boltz-2's own flash pair-bias kernel (boltz2/opt/forward/sampler/src/bz_sampler_dit.py ``_flash_bias_kernel``, the core of
    # its ``dit_fused`` sampler lever): rows [q | k | v | g], sigmoid(g) fused, one bias per batch row shared by the samples with the key mask
    # folded in as Boltz's ``(1 - mask) * -1e6``; bf16 operands (bf16) or fp32 dots (fp32: Boltz-2's ``tf32`` word under the engine's TF32
    # rule, else ``ieee``); head dim 48 only. The rows are the packed q / k / v / g GEMM's output read in place (Boltz-2's per-layer qkvg GEMM).
    _release_module = "bz_sampler_dit"
    _release_path = "boltz2/opt/forward/sampler/src"
    _release_fused_gate = True

    def _release_core(self, q, k, v, bias, mask, gate):
        import triton

        bz = _anthropic.load(self._release_module, path=self._release_path)
        a, b, n, h, d = q.shape
        c, dt = h * d, q.dtype
        prec = "bf16" if dt == torch.bfloat16 else ("tf32" if _tf32() else "ieee")
        bq, bkv, warps, stages = bz._flash_cfg(n, prec)
        qb = torch.zeros(c, dtype=dt, device=q.device)  # the q bias is in q already (to_query's bias)
        out = torch.empty(a, b, n, c, dtype=dt, device=q.device)
        packed = q.flatten(-2)  # [A, B, L, C] column view of the [A, B, L, 4 C] q / k / v / g rows
        for j in range(b):
            t = packed[:, j]
            if (
                t.stride(1) == 4 * c
                and gate is not None
                and gate.data_ptr() == q.data_ptr() + 3 * c * q.element_size()
                and t.stride(0) == n * t.stride(1)
            ):
                rows = torch.as_strided(t, (a * n, 4 * c), (4 * c, 1))  # [A * L, 4 C] in place
            else:
                rows = torch.cat([u[:, j].reshape(a * n, c) for u in (q, k, v, cast(torch.Tensor, gate))], -1).contiguous()
            pb = bias[j]  # [H, L, L] head-major
            if mask is not None:
                pb = pb.float().add_((1 - mask[0, j].float())[None, None, :] * -1e6)
            pb = pb.to(dt).contiguous()
            o = out[:, j]
            ctx = o.view(a * n, c) if o.is_contiguous() else torch.empty(a * n, c, dtype=dt, device=q.device)
            bz._flash_bias_kernel[(triton.cdiv(n, bq), h, a)](
                rows,
                qb,
                pb,
                ctx,
                n,
                rows.stride(0),
                pb.stride(0),
                pb.stride(1),
                ctx.stride(0),
                float(d**-0.5),
                H=h,
                HD=d,
                D=c,
                PREC={"bf16": 3, "tf32": 1, "ieee": 0}[prec],
                OUT_BF16=dt == torch.bfloat16,
                BQ=bq,
                BKV=bkv,
                num_warps=warps,
                num_stages=stages,
            )
            if not o.is_contiguous():
                o.copy_(ctx.view(a, n, c))
        return out

    def _release_range(self, single, cond, pair, mask) -> str | None:
        if _core_dtype(single) not in (torch.bfloat16, torch.float32):
            return f"core dtype {_core_dtype(single)} (bfloat16 / float32 only)"
        if single.shape[-1] // self.attention.n_head != 48:
            return "head dim != 48 (Boltz-2's kernel is written for 32 + 16 columns per head)"
        if mask is not None and mask.ndim == 3 and not bool((mask == mask[:1]).all()):
            return "a key mask that differs across the samples (Boltz-2 folds one mask per batch row into the shared bias)"
        return None


class TokenDiTRoseTTAFold3(VariantTokenCuda, TokenDiTProtenix):
    """RoseTTAFold3's token DiT block (``af3_diffusion_transformer.py``): the parallel schedule ``a + dA(a) + dT(a)``
    (``no_residual_connection_between_attention_and_transition: true``), a bias-free query (``to_q``: ``bias=False``) and an affine LayerNorm
    of the flat q / k (``kq_norm``: ``attention.norm_query`` / ``attention.norm_key``, over all heads)."""

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results
    #: 2026-11-10: H100 bf16 / fp32 inference with a conditioning per sample in one call (was one kernel call per sample)
    #: 3 (2026-11-10): H100 fp32 training / inference bias made key-permuted by the pair LayerNorm (no gathers), the pair passed as a view
    #: 4 (2026-11-10): H100 on the one token DiT algorithm (TokenDiTProtenix's cuda route with this block's options), eval and train
    #: 5 (2026-11-12): bf16 inference on the folded, grouped conditioning GEMM (TokenDiTProtenix cuda:h100_sm90 2); Boltz-2: the up-gate
    #: transition in the SwiGLU GEMM's epilogue
    backend_versions: ClassVar[dict[str, int]] = {**TokenDiTProtenix.backend_versions, "cuda:h100_sm90": 5}

    backends = (
        "pytorch",
        "anthropic",
        "cueq",
        "cuda",
        "triton",
    )  # "cuda": H100 the one token DiT algorithm (its options), A100 VariantTokenCuda

    parallel = True
    attention_part = FlatQKNormAugmentedAttention

    def make_inputs(self, **kwargs) -> dict:
        return {**super().make_inputs(**kwargs), "mask": None}  # the model calls the block without a key mask

    # "anthropic": the core RoseTTAFold3's kit runs (rosettafold3/opt/rosettafold3_opt/dtk.py ``attn``: opt_core's
    # ``dtk_kernels.flash_bias_attn`` per sample, the bias in the activation dtype, no key mask; the gate stays outside, as the kit applies
    # it in torch; fp32 products tf32 under the engine's TF32 rule).
    _release_module = "opt_core.kernels.dtk_kernels"
    _release_fused_gate = False

    def _release_core(self, q, k, v, bias, mask, gate):
        return _core_dtk_flash(q, k, v, bias, None, None)

    def _release_range(self, single, cond, pair, mask) -> str | None:
        if _core_dtype(single) not in (torch.bfloat16, torch.float32):
            return f"core dtype {_core_dtype(single)} (bfloat16 / float32 only)"
        if single.shape[-1] // self.attention.n_head > 64:
            return "head dim > 64 (the kernel's limit)"
        if mask is not None and not bool(mask.all()):
            return "a key mask (RoseTTAFold3's release core takes none)"
        return None


class TokenDiTChai1(MasterEntryCast, VariantTokenCuda, TokenDiTProtenix):
    """Chai-1's token DiT block: the parallel schedule, the identity AdaLN ``(1 + W_s s) LN(a) + W_b s`` in both parts
    (``attention.ada_ln_in``, ``transition.ada_ln_in``), no gating query; the pair is 256 wide. ``adaln_eps``: the epsilon of both AdaLNs'
    LayerNorm of ``a`` (Chai-1's release, its traced ``layer_norm(x, None, None, 0.1)``: 0.1, the default)."""

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results
    #: 2026-11-10: H100 bf16 / fp32 inference with a conditioning per sample in one call (was one kernel call per sample)
    #: 3 (2026-11-10): H100 fp32 training / inference bias made key-permuted by the pair LayerNorm (no gathers), the pair passed as a view
    #: 4 (2026-11-10): H100 on the one token DiT algorithm (TokenDiTProtenix's cuda route with this block's options), eval and train
    #: 5 (2026-11-12): bf16 inference on the folded, grouped conditioning GEMM (TokenDiTProtenix cuda:h100_sm90 2); Boltz-2: the up-gate
    #: transition in the SwiGLU GEMM's epilogue
    backend_versions: ClassVar[dict[str, int]] = {**TokenDiTProtenix.backend_versions, "cuda:h100_sm90": 5}

    backends = (
        "pytorch",
        "anthropic",
        "cueq",
        "cuda",
        "triton",
    )  # "cuda": H100 the one token DiT algorithm (its options), A100 VariantTokenCuda

    parallel = True
    attention_part = UngatedAugmentedAttention

    def make_inputs(self, **kwargs) -> dict:
        inputs = {**super().make_inputs(**kwargs), "mask": None}  # the model calls the block without a key mask
        return fp32_conditioning(inputs, kwargs.get("stream"), kwargs["generator"])

    def __init__(
        self,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 256,
        n_head: int = 16,
        n: int = 2,
        *,
        adaln_eps: float = 0.1,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, backend=backend)
        self.attention.ada_ln_in = IdentityAdaptiveLayerNorm(d_single, d_cond, eps=adaln_eps)
        self.transition.ada_ln_in = IdentityAdaptiveLayerNorm(d_single, d_cond, eps=adaln_eps)

    # "anthropic": the core Chai-1's kit serves (chai1/opt/forward/dstep_megakernel/chai1_fastln/dit_attn.py ``_serve4``): opt_core's
    # pair-bias provider with the row it selects for fp32 cell dit_h16d48 under the tier word "fast" (graphed, capture), [S, H, N, D] with one
    # [1, H, N, N] bias, no gate, no key mask (Chai-1 folds its mask into the bias). A stock row the provider selects (SDPA ...) is served as the
    # release serves it. Chai-1's denoiser runs this core in fp32 only (its Router takes fp32 q / k / v): under the master layout (bf16
    # activations) the core's operands are widened to fp32 as Chai-1's fp32 DiT has them, the rest of the block stays in autocast.
    _release_module = "opt_core.kernels.apb"
    _release_fused_gate = False  # no gating query
    _CELL = "dit_h16d48"

    def _chai_selection(self, n: int, samples: int, d: int, h: int, device):
        return _provider_selection(n, samples, d, h, torch.float32, device, self._CELL)

    def _release_core(self, q, k, v, bias, mask, gate):
        a, _, n, h, d = q.shape
        sel, why = self._chai_selection(n, a, d, h, q.device)
        if sel is None:
            raise RuntimeError(why)
        q, k, v, bias = (t.float() for t in (q, k, v, bias))
        return _core_provider(q, k, v, bias, None, None, sel, cell=self._CELL)

    def _release_range(self, single, cond, pair, mask) -> str | None:
        if _core_dtype(single) not in (torch.bfloat16, torch.float32):
            return f"core dtype {_core_dtype(single)} (bfloat16 / float32 only; Chai-1's release core runs float32)"
        if mask is not None and not bool(mask.all()):
            return "a key mask (Chai-1's release core takes the mask inside its bias, not as a mask)"
        h = self.attention.n_head
        return self._chai_selection(single.shape[2], single.shape[0], single.shape[-1] // h, h, single.device)[1]


class TokenDiTMiniWorldBiasOnly(BackendModule):
    """MiniWorld's bias-only token DiT block: AF3 Alg. 23 with the attention's query-key half removed::

        a = BiasOnlyAttention(a, s, z, mask)           a + gated softmax(pair bias) v -- no query, no key
        a = ConditionedTransition(a, s)                a + transition(a, s)

    ``forward(single, cond, pair, mask)`` -> ``single``'s shape. ``single`` and ``cond`` carry the augmentation axis (``A, B, L, d``);
    ``pair`` does not (``B, L, L, d_pair``). The token widths are the defaults (768 / cond 384 / pair 128 / 16 heads x 48 / transition
    n = 2), the same as :class:`TokenDiTProtenix`. ``forward(..., bias=b)``: the block's pair bias precomputed ([B, H, L, L], from
    :func:`pair_bias_all`); ``pair`` may be None. A pair of its own per sample is ``single`` [1, B, L, d] with ``pair`` [B, L, L, d_pair].
    No ``"anthropic"`` backend: no release folder has the bias-only attention.
    """

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda", "triton")
    # 2026-11-10: sm_80 paths; cuda:h100_sm90 2 (2026-11-12): the inference pair-bias hoist (runner.hoist) in one pass for a 256-wide
    # bf16 pair, and on the TF32 tensor-core kernel (3 terms) for a 128-wide fp32 pair
    # cuda:a100_sm80 3 (2026-11-12): the generic sm_80 pair-bias backward (pair_bias_gen_bwd: padded W' pitch, next row prefetched; same values) 2.1x faster
    # cuda:h100_sm90 3 (2026-11-12): pair_bias_all on H100 is the fused hoist (head-major contiguous biases, as engine 2.4.4): the
    # inference block no longer copies a permuted bias view per call (L 768 24-block stack 9.8 -> ~3.8 ms)
    backend_versions: ClassVar[dict[str, int]] = {"cuda:a100_sm80": 3, "cuda:h100_sm90": 3}

    def __init__(
        self,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 128,
        n_head: int = 16,
        n: int = 2,
        *,
        d_head: int | None = None,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.attention = BiasOnlyAttention(d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head, d_head=d_head)
        self.transition = ConditionedTransition(d_hidden=d_single, d_cond=d_cond, n=n)
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Each part returns its residual output (stream in, stream out). ``bias``: this block's hoisted pair bias (:func:`pair_bias_all`) in
        place of ``to_bias(ln_pair(pair))``."""
        if bias is None and pair is None:
            raise ValueError("TokenDiTMiniWorldBiasOnly: pass the pair or the block's precomputed bias")
        single = self.attention(single, cond, pair, mask) if bias is None else self.attention(single, cond, pair, mask, bias=bias)
        return self.transition(single, cond)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return _token_dit_inputs(
            self, length=length, samples=samples, dtype=dtype, device=device, generator=generator, depth=depth, stream=stream
        )

    # ------------------------------------------------------------------------------------------------------------ "triton"
    # Engine 2.4's portable block (BiasOnlyDiTBlock off H100 / B200), any card: both parts on their "triton" backends (set_backend) --
    # BiasOnlyAttention (Triton AdaLN, the one-pass pair LayerNorm + bias kernel, the feature-major training core) and ConditionedTransition
    # (Triton AdaLN, the SwiGLU / gate tail kernels); the block's statements chain them.
    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        parts = "triton" if backend == "triton" else "pytorch"
        self.attention.set_backend(parts)
        self.transition.set_backend(parts)
        return self

    def _supports_triton(self, single, cond, pair=None, mask=None, *, bias=None) -> str | None:
        return self.attention._supports_triton(single, cond, pair, mask, bias=bias) or self.transition._supports_triton(single, cond)

    def _forward_triton(self, single, cond, pair=None, mask=None, *, bias=None) -> torch.Tensor:
        return self._forward_pytorch(single, cond, pair, mask, bias=bias)

    # ------------------------------------------------------------------------------------------------------------ "cuda"
    # H100: no gradient -> kernels/integrations/bias_only_dit (the fused inference block: P = softmax(pair bias) once per pair, the
    # pv_gate core, CUDA rows, cuBLAS); a gradient -> kernels/integrations/bias_only_dit_train (one autograd Function on the CUDA rows,
    # cuBLAS and the bias-only cores). Range = the two gates: bf16 / fp32 (fp32 = fp32 weights, no autocast), [A, 1, L, 768] with L a
    # multiple of 128 (training L <= 768), 16 heads x 48, cond 384 or 768, pair 128 / 256, n = 2, eps 1e-5,
    # key mask [1, L] or none, the pair or the block's hoisted bias [1, H, L, L].
    # A100 (sm_80): kernels/integrations/bias_only_dit_sm80 (the AdaLN and ConditionedTransition sm_80 paths around one autograd Function
    # for the attention: cuBLAS, the generic sm_80 pair bias, the row softmax, the sm_80 bias-only cores PvGate / Dpb; bf16 / master / fp32).
    def _cuda_route(self, single, cond, pair, mask, bias):
        if _a100(single):
            from miniworld_engine.kernels.integrations import bias_only_dit_sm80

            why = bias_only_dit_sm80.refusal(self, single, cond, pair, mask, bias, _wants_grad(self, single, cond, pair, bias))
            return (None, why) if why is not None else ("a100", None)
        why = _h100_reason(single)
        if why is not None:
            return None, why
        view = _kernel_view(self)
        if _wants_grad(self, single, cond, pair, bias):
            from miniworld_engine.kernels.integrations import bias_only_dit_train

            if bias_only_dit_train.serves(view, single, cond, pair, mask, bias=bias):
                return "train", None
            return None, (
                "training: outside the fused H100 bias-only training block's range (bf16 / fp32 weights, [A, 1, L, 768] with L "
                "a multiple of 128 <= 768, heads x head width 16 x 48 / 24 x 32 / 12 x 64 / 16 x 64, "
                f"cond {' / '.join(map(str, bias_only_dit_train.KERNEL_D_CONDS))}, pair {' / '.join(map(str, bias_only_dit_train.KERNEL_D_PAIRS))}, "
                "n = 2, key mask [1, L], eps 1e-5)"
            )
        from miniworld_engine.kernels.integrations import bias_only_dit

        if bias_only_dit.serves(view, single, cond, pair, mask, bias=bias):
            return "infer", None
        return None, (
            "inference: outside the fused H100 bias-only inference block's range (bf16 / fp32 weights without autocast, "
            "[A, 1, L, 768] with L a multiple of 128, heads x head width 16 x 48 / 24 x 32 / 12 x 64 / 16 x 64, cond 384 or 768, "
            "pair 128 / 256, n = 2, key mask [1, L], eps 1e-5)"
        )

    def _supports_cuda(self, single, cond, pair=None, mask=None, *, bias=None) -> str | None:
        route, why = self._cuda_route(single, cond, pair, mask, bias)
        return why if route is None else None

    def _forward_cuda(self, single, cond, pair=None, mask=None, *, bias=None) -> torch.Tensor:
        route, _ = self._cuda_route(single, cond, pair, mask, bias)
        if route == "a100":
            from miniworld_engine.kernels.integrations import bias_only_dit_sm80

            return bias_only_dit_sm80.block(self, single, cond, pair, mask, _wants_grad(self, single, cond, pair, bias))
        view = _kernel_view(self)
        if route == "train":
            from miniworld_engine.kernels.integrations import bias_only_dit_train

            return bias_only_dit_train.block(view, single, cond, pair, mask, bias=bias)
        from miniworld_engine.kernels.integrations import bias_only_dit

        return bias_only_dit.update(view, single, cond, pair, mask, bias=bias)


def pair_bias_all(blocks, pair: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Every bias-only block's pair bias at once: ``tuple(to_bias(ln_pair(pair)).permute(0, 3, 1, 2) for each block)`` -- [B, H, L, L] each
    -- from one LayerNorm and one GEMM, differentiable in ``pair`` and in every block's ``ln_pair.weight`` / ``to_bias.weight``.

    ``ln_pair`` has no offset, so with LN0 the LayerNorm without affine and the fold W'_b = W_b diag(gamma_b), ``bias_b = LN0(pair) W'_b^T``:
    the normalization is the same for every block, only gamma differs. Pass block b its bias (``block(single, cond, pair, mask,
    bias=biases[b])``) and the block touches neither ``ln_pair`` nor ``to_bias``; the gradients reach them, and the pair, through this op.
    Same parameters, the same result up to rounding (gamma applied to the weight instead of the activation). Under activation checkpointing
    call it OUTSIDE the checkpointed blocks.

    ``blocks``: :class:`TokenDiTMiniWorldBiasOnly` blocks (or their ``attention`` modules); they must share the LayerNorm eps and the head
    count, and ``ln_pair`` must have no offset."""
    mods = list(blocks)
    if not mods:
        raise ValueError("pair_bias_all: no blocks")
    attns = [m.attention if hasattr(m, "attention") else m for m in mods]
    eps, H = attns[0].ln_pair.eps, attns[0].to_bias.weight.shape[0]
    for a in attns:
        if (
            a.ln_pair.eps != eps
            or a.to_bias.weight.shape[0] != H
            or getattr(a.ln_pair, "bias", None) is not None
            or getattr(a.to_bias, "bias", None) is not None
            or a.ln_pair.weight is None
        ):
            raise ValueError(
                "pair_bias_all: the blocks must share eps and head count, ln_pair must have a weight and no offset, to_bias no bias"
            )
    # the "cuda" backend on a B200 or an H100: the fused hoist (one LayerNorm and one GEMM for every block, the blocks' bias gradients
    # written into one buffer, each block's bias a contiguous head-major [1, H, L, L] slice); pair [1, L, L, 128] (or 256 on H100) bf16 or
    # fp32 -- any other pair (and the A100) takes the PyTorch fold below, whose per-block biases are permuted views
    if getattr(mods[0], "backend", "pytorch") == "cuda":
        from miniworld_engine.kernels.integrations import bias_only_dit_hoist

        if bias_only_dit_hoist.serves("miniworld", attns, pair):
            return bias_only_dit_hoist.pair_bias_all(attns, pair)
    # LN0 in at least fp32 (as the engine's LayerNorm computes), one matmul against W'_all, per-block head-major views [B, H, L, L] of the
    # [B, L, L, N] product
    dp = pair.shape[-1]
    ln0 = F.layer_norm(pair.to(wide(pair)), (dp,), None, None, eps).to(pair.dtype)
    wf = torch.cat([a.to_bias.weight * a.ln_pair.weight.to(a.to_bias.weight.dtype)[None, :] for a in attns], 0)  # [N, dp]
    b = torch.matmul(ln0, wf.t().to(ln0.dtype))  # [B, L, L, N]
    return tuple(b[..., i * H : (i + 1) * H].permute(0, 3, 1, 2) for i in range(len(attns)))


# ======================================================================================================================== selector
#: model name -> its token DiT class. Several models share a class where their blocks are the same math.
TOKEN_DIT_MODELS: dict[str, type[BackendModule]] = {
    "af3": TokenDiTProtenix,
    "boltz2": TokenDiTBoltz2,
    "rosettafold3": TokenDiTRoseTTAFold3,
    "chai1": TokenDiTChai1,
    "openfold3": TokenDiTProtenix,
    "openfold3_preview2": TokenDiTProtenix,
    "protenix": TokenDiTProtenix,
    "protenix_v2": TokenDiTProtenix,
    "opendde": TokenDiTProtenix,
    "intellifold2": TokenDiTProtenix,
    "esmfold2": TokenDiTProtenix,
    "miniworld": TokenDiTMiniWorld,
    "miniworld_bias_only": TokenDiTMiniWorldBiasOnly,
}


class TokenDiT:
    """``TokenDiT(model=..., *args, **kwargs)``: builds ``TOKEN_DIT_MODELS[model](*args, **kwargs)`` (the model's own class, not a
    wrapper: its state dict has the model class's keys)."""

    models = TOKEN_DIT_MODELS

    def __new__(cls, model: str, *args, **kwargs):
        try:
            klass = TOKEN_DIT_MODELS[model]
        except KeyError:
            raise ValueError(f"TokenDiT: unknown model {model!r}; known: {', '.join(sorted(TOKEN_DIT_MODELS))}") from None
        return klass(*args, **kwargs)


__all__ = [
    "TOKEN_DIT_MODELS",
    "TokenDiT",
    "TokenDiTBoltz2",
    "TokenDiTChai1",
    "TokenDiTMiniWorld",
    "TokenDiTMiniWorldBiasOnly",
    "TokenDiTProtenix",
    "TokenDiTRoseTTAFold3",
    "pair_bias_all",
]
