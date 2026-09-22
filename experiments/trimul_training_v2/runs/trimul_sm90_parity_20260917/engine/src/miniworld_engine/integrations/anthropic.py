"""Inference adapters for Anthropic's uplifting-biomolecular-modeling release.

Upstream algorithms live unchanged under third_party/anthropic. These adapters
provide engine entry points; they do not implement or claim upstream kernels.
Low-level release calls require no_grad(). TriMul modules also expose an explicit
native_rebuilt training baseline in anthropic_training.py.
"""
from importlib import import_module
import os
from pathlib import Path
import sys

REVISION = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_ROOT = None


def configure(root=None):
    """Locate the vendored release; reject an already-imported different copy."""
    global _ROOT
    root = Path(root or os.environ.get("MINIWORLD_ANTHROPIC_ROOT") or
                Path(__file__).resolve().parents[3] / "third_party/anthropic/upstream").resolve()
    core = root / "common/opt_core"
    if not (core / "opt_core/__init__.py").is_file():
        raise RuntimeError("Anthropic sources missing: run scripts/import_anthropic.py or set MINIWORLD_ANTHROPIC_ROOT")
    loaded = sys.modules.get("opt_core")
    if loaded is not None and Path(loaded.__file__).resolve().parent != core / "opt_core":
        raise RuntimeError(f"opt_core already loaded from a different source: {loaded.__file__}")
    if _ROOT is not None and _ROOT != root:
        raise RuntimeError("Changing the upstream source in a live process is not supported")
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    _ROOT = root
    return root


def provider(family):
    """Access a named upstream provider, retaining its selection/refusal contract."""
    if family not in {"triattn", "trimul", "transition", "ln", "apb", "triattn_xla", "trimul_xla", "pallas"}:
        raise ValueError(f"Unknown provider: {family}")
    if _ROOT is None:
        configure()
    return import_module(f"opt_core.kernels.{family}")


def carried_kernel(name):
    """Access every upstream META-registered kernel by its original name."""
    if _ROOT is None:
        configure()
    k = import_module("opt_core.kernels")
    k.route(name)
    return import_module(name)


def operation(name):
    """Access the release's MSA/OPM operations and fused pair blocks."""
    allowed = {"msa_fused.msa_triton", "msa_opm", "msa_pwa", "msa_pwa2"}
    if name not in allowed:
        raise ValueError(f"Unknown upstream operation: {name}")
    if _ROOT is None:
        configure()
    return import_module(f"opt_core.ops.{name}")


def module_triangle_attention(module, pair, mask):
    """Full upstream surround + selected core; preserve out-of-place residual."""
    _inference()
    if module.training and module.p_drop:
        raise RuntimeError("Anthropic TriangleAttention requires eval() when dropout is enabled")
    if module.use_qk_norm:
        raise ValueError("Anthropic block surround does not support Q/K RMSNorm; use a core-only row")
    if _ROOT is None:
        configure()
    pf = import_module("opt_core.attn.pair_fused")
    weights = dict(w_o=module.to_out.weight, w_q=module.to_query.weight,
                   w_k=module.to_key.weight, w_v=module.to_value.weight,
                   w_g=module.to_gate.weight, w_b=module.to_bias.weight,
                   ln_w=module.ln_pair.weight, ln_b=module.ln_pair.bias)
    key = (_signature(weights.values()), module.ln_pair.eps, module.n_head)
    if getattr(module, "_anthropic_pack_key", None) != key:
        module._anthropic_weights = pf.pack_triattn_weights(
            **weights, n_heads=module.n_head,
            head_dim=module.to_value.weight.shape[0] // module.n_head,
            eps=module.ln_pair.eps)
        module._anthropic_pack_key = key
    # Engine masks are key-only, in either starting or ending attention frame.
    # A pairwise AND would incorrectly mask entire query rows.
    m5 = None if mask is None else mask[:, None, None, None, :].expand(-1, pair.shape[1], -1, -1, -1)
    core = "tier:" + module.anthropic_row.removeprefix("block:")
    kwargs = dict(ending=not module.starting, residual=False, impl="fpf", core=core, ln="fused")
    plan = pf._plan_triattn(pair, module._anthropic_weights, m5, variant=None, engage_cells=False, **kwargs)
    module.anthropic_selection = {"row": module.anthropic_row, "surround": plan,
                                  "residual": "out-of-place engine add"}
    update = pf.tri_attn_block(pair, module._anthropic_weights, m5, **kwargs)
    return pair + update


def _inference():
    import torch
    if torch.is_grad_enabled():
        raise RuntimeError("Anthropic integration is inference-only; use torch.no_grad() or torch.inference_mode(). Training support is deferred.")


def triangle_attention(q, k, v, bias, mask=None, scale=None, *, row="k2b", **kwargs):
    _inference()
    return provider("triattn").triangle_attention(q, k, v, bias, mask, scale, word=row, **kwargs)


def triangle_multiplication(z, mask=None, *, weights, direction="outgoing", row="v4", residual=True, cache=None, **kwargs):
    _inference()
    if row == "native_rebuilt":
        # A separately rebuilt payload is explicitly named, never substituted for
        # the release's sealed CUDA-13 binaries or advertised as its exact tier.
        root = configure() if _ROOT is None else _ROOT
        if not os.environ.get("TRIMUL_NATIVE_BUILD_DIR"):
            raise RuntimeError("native_rebuilt requires TRIMUL_NATIVE_BUILD_DIR")
        py = Path(os.environ["TRIMUL_NATIVE_BUILD_DIR"]).resolve().parent / "python"
        if not (py / "trimul_native/face.py").is_file():
            raise RuntimeError("Rebuilt payload must retain its source python/ and testvectors/ next to build/")
        loaded = sys.modules.get("trimul_native")
        if loaded is not None and Path(loaded.__file__).resolve().parent != py / "trimul_native":
            raise RuntimeError("trimul_native already loaded from a different source")
        if str(py) not in sys.path:
            sys.path.insert(0, str(py))
        face = import_module("trimul_native.face")
        face.check(device=z.device.index, gate=False)
        return face.serve(z, mask, direction=direction, weights=weights, residual=residual, cache=cache, **kwargs)
    return provider("trimul").triangle_multiplication(z, mask, direction=direction, weights=weights,
                                                      word=row, residual=residual, cache=cache, **kwargs)


def pack_transition(**weights):
    return provider("transition").pack(**weights)


def transition(x, weights, *, row="v2", residual=True, **kwargs):
    _inference()
    return provider("transition").transition(x, weights, word=row, residual=residual, **kwargs)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, *, row="fastln", **kwargs):
    _inference()
    return provider("ln").layer_norm(x, normalized_shape, weight, bias, eps, word=row, **kwargs)


def pair_bias_attention(q, k, v, bias, key_mask=None, gate=None, *, row="apb_attn", **kwargs):
    _inference()
    return provider("apb").pair_bias_attention(q, k, v, bias, key_mask, gate, word=row, **kwargs)


def atom_attention(q, k, v, bias, gate=None, *, row="fpf_atom", **kwargs):
    _inference()
    return provider("apb").atom_attention(q, k, v, bias, gate, word=row, **kwargs)


def _signature(tensors):
    # Track replacement, load_state_dict/copy_, device and dtype changes. Packed
    # inference tensors have no version counter, so require ordinary Parameters.
    return tuple((id(t), t._version, t.device, t.dtype, tuple(t.shape)) for t in tensors)


def module_transition(module, x):
    _inference()
    ts = (module.expand_a.weight, module.expand_b.weight, module.squeeze.weight,
          module.ln_in.weight, module.ln_in.bias)
    key = (_signature(ts), module.ln_in.eps)
    if getattr(module, "_anthropic_pack_key", None) != key:
        module._anthropic_weights = pack_transition(w_a=ts[0], w_b=ts[1], w_o=ts[2], ln_w=ts[3], ln_b=ts[4], eps=module.ln_in.eps)
        module._anthropic_pack_key = key
    y, selection = transition(x, module._anthropic_weights, row=module.anthropic_row)
    module.anthropic_selection = selection
    return y


def module_trimul(module, pair, mask, dropout_p):
    import torch
    if torch.is_grad_enabled() or (module.training and dropout_p):
        from .anthropic_training import module_update
        update = module_update(module, pair, mask)
        if module.training and dropout_p:
            update = update * module._make_drop_row_scale(pair, dropout_p)
        return pair + update
    _inference()
    if module.training and dropout_p:
        raise RuntimeError("Anthropic TriMul module requires eval() when dropout is enabled")
    if module.ln_pair.eps != module.ln_out.eps:
        raise ValueError("Anthropic TriMul requires equal input/output LayerNorm epsilon")
    weights = dict(ln_in_w=module.ln_pair.weight, ln_in_b=module.ln_pair.bias,
                   w_ag=module.to_left_gate.weight, w_ap=module.to_left.weight,
                   w_bg=module.to_right_gate.weight, w_bp=module.to_right.weight,
                   ln_out_w=module.ln_out.weight, ln_out_b=module.ln_out.bias,
                   w_o=module.to_out.weight, w_og=module.to_gate.weight)
    key = (_signature(weights.values()), module.ln_pair.eps)
    if getattr(module, "_anthropic_pack_key", None) != key:
        module._anthropic_weights = {k: t.detach().contiguous() for k, t in weights.items()}
        module._anthropic_cache = {}
        module._anthropic_pack_key = key
    pairmask = None if mask is None else (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(pair.dtype)
    result = triangle_multiplication(pair, pairmask, weights=module._anthropic_weights,
                                     direction="outgoing" if module.outgoing else "incoming",
                                     row=module.anthropic_row, cache=module._anthropic_cache, eps=module.ln_pair.eps)
    module.anthropic_selection = {"row": module.anthropic_row, "local_rebuild": module.anthropic_row == "native_rebuilt"}
    for key, value in module._anthropic_cache.items():
        if isinstance(key, tuple) and key and key[0] == "_sel":
            module.anthropic_selection = value
    return result
