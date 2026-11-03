"""The engine's stacks and the remaining whole modules; dimensions live in registry_module.csv.

* ``dit_variant`` / ``atom_dit_variant``: ``modules.dit_variant.DiTVariantStack`` / ``AtomDiTVariantStack``, the Boltz-2, RoseTTAFold3,
  Chai-1 and AlphaFold 3 DiT stacks (a preset flag: ``boltz2=1``, ``rosettafold3=1``, ``chai1=1``, ``af3=1``). A stack of two token blocks:
  what a stack does that a block does not (one pair bias for its blocks) is part of the call.
* ``swa_atom_transformer``: ``modules.swa_atom_transformer.SWAAtomTransformer``, one block, ``esmfold2=1`` (the fused SWA block), ``af3=1``,
  ``af3_global=1`` (global attention).
* ``input_feature_embedder``: ``modules.input_feature_embedder.InputFeatureEmbedder`` at MiniWorld's widths (atom 128, token 384, 4 heads,
  3 blocks, window 128); ``esmfold2=1`` the ESMFold2 atom features with the linear token pool.
* ``rmsnorm_native``: ``modules.primitives.RMSNorm`` (``has_weight``: the affine weight).

Registry dims are integers: a preset is a flag (``name=1``), never a string.
"""

from __future__ import annotations

import torch

NAMES = ("dit_variant", "atom_dit_variant", "swa_atom_transformer", "input_feature_embedder", "rmsnorm_native")
TOKEN_PRESETS = ("boltz2", "rosettafold3", "chai1", "af3")
ATOM_PRESETS = {"boltz2": "BOLTZ2_ATOM", "rosettafold3": "ROSETTAFOLD3_ATOM", "chai1": "CHAI1_ATOM", "af3": "AF3_ATOM"}


def _preset(dims: dict) -> str:
    named = [p for p in TOKEN_PRESETS if dims.get(p)]
    if len(named) != 1:
        raise ValueError(f"a DiT variant row names exactly one preset of {TOKEN_PRESETS}, got {dims}")
    return named[0]


def _plain(dims: dict, flags) -> dict:
    return {k: v for k, v in dims.items() if k not in flags}


def _token_stack(dims, p, impl, dt):
    from miniworld_engine.modules import dit_variant
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    options = getattr(dit_variant, {"boltz2": "BOLTZ2", "rosettafold3": "ROSETTAFOLD3", "chai1": "CHAI1", "af3": "AF3"}[_preset(dims)])
    kw = {**_plain(dims, TOKEN_PRESETS), **options}
    return dit_variant.DiTVariantStack(implementation=IT(impl), **kw).cuda().to(dt)


def _atom_stack(dims, p, impl, dt):
    from miniworld_engine.modules import dit_variant
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    kw = {**_plain(dims, (*TOKEN_PRESETS, "pad_window")), **getattr(dit_variant, ATOM_PRESETS[_preset(dims)])}
    if dims.get("pad_window"):   # the clipped, OR-masked 128-key window (FoldForge's exact mode; Boltz-2 / RoseTTAFold3 encoders)
        kw["key_window"] = "pad"
    return dit_variant.AtomDiTVariantStack(implementation=IT(impl), **kw).cuda().to(dt)


def _token_inputs(b, length, dims, dt, s):
    """``single`` [A, L, D], ``cond`` [L, c_s] (shared by the samples), ``pair`` [L, L, C], ``mask`` [L]."""
    from miniworld_engine.modules import dit_variant

    options = getattr(dit_variant, {"boltz2": "BOLTZ2", "rosettafold3": "ROSETTAFOLD3", "chai1": "CHAI1", "af3": "AF3"}[_preset(dims)])
    return (torch.randn(b, length, dims.get("d_single", 768), device="cuda", dtype=dt),
            torch.randn(length, options["d_cond"], device="cuda", dtype=dt),
            torch.randn(length, length, options["d_pair"], device="cuda", dtype=dt),
            torch.ones(length, dtype=torch.bool, device="cuda"))


def _atom_inputs(b, n, dims, dt, s):
    """``single`` [A, N, 128], ``cond`` [N, 128], the trunked ``pair`` [N / 32, 32, 128, d_pair], ``mask`` [N]."""
    return (torch.randn(b, n, dims.get("d", 128), device="cuda", dtype=dt),
            torch.randn(n, dims.get("d_cond", 128), device="cuda", dtype=dt),
            torch.randn((n + 31) // 32, 32, 128, dims.get("d_pair", 16), device="cuda", dtype=dt),
            torch.ones(n, dtype=torch.bool, device="cuda"))


SWA_FLAGS = ("esmfold2", "af3", "af3_global")


def _swa_config(dims):
    from miniworld_engine.modules.swa_atom_transformer import SWAAtomTransformer

    style = next((f for f in SWA_FLAGS if dims.get(f)), None)
    if style is None:
        raise ValueError(f"a swa_atom_transformer row names one of {SWA_FLAGS}, got {dims}")
    return SWAAtomTransformer.Config(n_block=dims.get("n_block", 1), block_style="esmfold2" if style == "esmfold2" else "af3",
                                     global_attn=style == "af3_global", fused_triton=True if style == "esmfold2" else None,
                                     swa_window_size=dims.get("window", 128))


def _swa_stack(dims, p, impl, dt):
    from miniworld_engine.modules.swa_atom_transformer import SWAAtomTransformer

    # the stack takes its implementation from its config's dispatch (the fused SWA block on the esmfold2 style), as MiniWorld builds it
    return SWAAtomTransformer(_swa_config(dims)).cuda().to(dt)


def _swa_inputs(b, n, dims, dt, s):
    from miniworld_engine.modules.swa_atom_attention import build_attention_params
    from miniworld_engine.modules.swa_atom_transformer import SWAAtomTransformer

    config = _swa_config(dims)
    rope = SWAAtomTransformer(config).cuda()
    cos, sin = rope.build_rope(torch.randn(1, n, 3, device="cuda"), torch.arange(n, device="cuda")[None])
    params = build_attention_params(cos, sin, torch.ones(b, n, dtype=torch.bool, device="cuda"), b)
    return (torch.randn(b, n, config.d_atom, device="cuda", dtype=dt), torch.randn(b, n, config.d_cond, device="cuda", dtype=dt), params)


def _embedder(dims, p, impl, dt):
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.input_feature_embedder import InputFeatureEmbedder

    esm = bool(dims.get("esmfold2"))
    linear = bool(dims.get("linear_pool", esm))   # the ESMFold2 features pool linearly unless the row says ``linear_pool=0``
    return InputFeatureEmbedder(dims["d_atom"], dims["d_token"], dims["n_head"], dims["n_block"], swa_window_size=dims.get("window", 128),
                                features="esmfold2" if esm else "legacy", token_pool="linear" if linear else "relu",
                                implementation=IT(impl)).cuda().to(dt)


def _embedder_inputs(b, n, dims, dt, s):
    """One structure of ``n`` atoms in ``n // 8`` tokens: positions, reference mask, element, charge, space uid, atom mask, atom -> token."""
    tokens = max(1, n // 8)
    return (torch.randn(b, n, 3, device="cuda", dtype=dt), torch.ones(b, n, device="cuda", dtype=dt),
            torch.randint(1, 20, (b, n), device="cuda").to(dt), torch.zeros(b, n, device="cuda", dtype=dt),
            torch.arange(n, device="cuda").div(8, rounding_mode="floor")[None].expand(b, n).contiguous(),
            torch.ones(b, n, dtype=torch.bool, device="cuda"),
            torch.arange(n, device="cuda").div(8, rounding_mode="floor").clamp_max(tokens - 1)[None].expand(b, n).contiguous(), tokens)


def _rmsnorm(dims, p, impl, dt):
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.primitives import RMSNorm

    return RMSNorm(dims["d_norm"], elementwise_affine=bool(dims.get("has_weight")), implementation=IT(impl)).cuda().to(dt)


def _rmsnorm_inputs(b, n, dims, dt, s):
    return (torch.randn(b, n, dims["d_norm"], device="cuda", dtype=dt),)


def cases():
    from miniworld_engine.autotune.builder import Case, _shapes

    return [
        Case("dit_variant", _token_stack, _token_inputs, **_shapes("dit_variant")),
        Case("atom_dit_variant", _atom_stack, _atom_inputs, **_shapes("atom_dit_variant")),
        Case("swa_atom_transformer", _swa_stack, _swa_inputs, **_shapes("swa_atom_transformer")),
        Case("input_feature_embedder", _embedder, _embedder_inputs, **_shapes("input_feature_embedder")),
        Case("rmsnorm_native", _rmsnorm, _rmsnorm_inputs, **_shapes("rmsnorm_native")),
    ]
