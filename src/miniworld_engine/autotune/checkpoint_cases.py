"""Released-checkpoint dimensions of whole modules; dimensions live in registry_module.csv.

``triangle_attention_checkpoint`` (``modules.triangle_attention.TriangleAttention`` at the checkpoints' widths) and ``layernorm_native``
(``modules.primitives.LayerNorm``, every affine norm width the checkpoints hold). The leaf probes that used to stand beside them
(projected QKV core, LayerNorm+Linear, RMSNorm modulation, gated residual, SwiGLU, the atom output projection) are parts of
modules, not modules: their kernels are reached through the module rows that contain them (``dit``, ``swa_dit``, ``attention_pair_bias``,
``triangle_attention_*``; see ``dit_cases``). Removed 2026-11-03 (git history keeps them).
"""

from __future__ import annotations

import torch

NAMES = (
    "triangle_attention_checkpoint",
    "layernorm_native",
)


def _inputs(kind, b, length, dims, dt, stream):
    width = dims["d_norm"]
    if stream == "atom_pair":
        # Atom pair windows: Q=32, K=128. Never allocate a dense N_atom squared pair.
        return (torch.randn(b, (length + 31) // 32, 32, 128, width, device="cuda", dtype=dt),)
    if stream == "token_pair":
        shape = (b, length, length, width)
    elif stream == "msa_token":
        shape = (b, 8, length, width)
    else:
        shape = (b, length, width)
    return (torch.randn(*shape, device="cuda", dtype=dt),)


def cases():
    from miniworld_engine.autotune.builder import Case, _mask, _pair, _shapes
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.primitives import LayerNorm
    from miniworld_engine.modules.triangle_attention import TriangleAttention

    return [
        Case(
            "triangle_attention_checkpoint",
            lambda dims, p, impl, dt: (
                TriangleAttention(**dims, implementation=ImplementationType(impl))
                .cuda()
                .to(dt)
            ),
            lambda b, l, dims, dt, s: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
            **_shapes("triangle_attention_checkpoint"),
        ),
        Case(
            "layernorm_native",
            lambda dims, p, impl, dt: (
                LayerNorm(
                    dims["d_norm"],
                    bias=bool(dims["has_bias"]),
                    implementation=ImplementationType(impl),
                )
                .cuda()
                .to(dt)
            ),
            lambda b, l, dims, dt, s: _inputs("layernorm_native", b, l, dims, dt, s),
            **_shapes("layernorm_native"),
        ),
    ]
