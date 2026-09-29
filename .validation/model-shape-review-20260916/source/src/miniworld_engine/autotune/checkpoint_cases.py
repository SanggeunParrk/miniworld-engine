"""Exact checkpoint leaf operations; dimensions live in registry_module.csv.

Keep these probes separate from whole-layer constructors: projected QKV, rounded
SwiGLU expansion and FP32 norm affine tensors are real production contracts.
"""

from __future__ import annotations

import torch
from torch import nn

NAMES = (
    "layernorm_linear_atom_output",
    "rms_norm_modulation",
    "triangle_attention_checkpoint",
    "gated_linear",
    "swiglu_ffn",
    "projected_attention",
    "layernorm_native",
    "layernorm_linear_native",
)


class Leaf(nn.Module):
    def __init__(self, kind: str, dims: dict, dtype: torch.dtype):
        super().__init__()
        self.kind = kind
        self.dims = dims

        def weight(name, shape, dt=dtype):
            self.register_parameter(
                name, nn.Parameter(torch.randn(*shape, device="cuda", dtype=dt))
            )

        if kind == "rms_norm_modulation":
            for name in ("wscale", "wshift", "wgate"):
                weight(name, (dims["d_hidden"], dims["d_cond"]))
        elif kind == "gated_linear":
            weight("weight", (dims["d_out"], dims["d_hidden"]))
        elif kind == "swiglu_ffn":
            for name in ("wa", "wb"):
                weight(name, (dims["d_expanded"], dims["d_hidden"]))
            weight("ws", (dims["d_hidden"], dims["d_expanded"]))
        elif kind in ("layernorm_linear_native", "layernorm_linear_atom_output"):
            weight("norm", (dims["d_norm"],), torch.float32)
            weight("weight", (dims["n_head"], dims["d_norm"]))

    def forward(self, *args):
        from miniworld_engine import ops

        if self.kind == "rms_norm_modulation":
            y, gate = ops.rms_norm_modulation(
                *args,
                self.wscale,
                self.wshift,
                self.wgate,
                eps=torch.finfo(args[0].dtype).eps,
            )
            return y + gate
        if self.kind == "gated_linear":
            return ops.gated_linear(*args, self.weight)
        if self.kind == "swiglu_ffn":
            return ops.swiglu_ffn(*args, self.wa, self.wb, self.ws)
        if self.kind in ("layernorm_linear_native", "layernorm_linear_atom_output"):
            return ops.layer_norm_linear(*args, self.norm, self.weight)
        if self.kind == "projected_attention":
            return ops.augmented_attention_pair_bias(*args)
        raise ValueError(self.kind)


def _activation(b, length, width, dt, stream):
    if stream == "token_pair":
        shape = (b, length, length, width)
    elif stream == "msa_token":
        shape = (b, 8, length, width)
    else:
        shape = (b, length, width)
    return torch.randn(*shape, device="cuda", dtype=dt)


def _inputs(kind, b, length, dims, dt, stream):
    if kind == "rms_norm_modulation":
        return (
            _activation(b, length, dims["d_hidden"], dt, stream),
            _activation(b, length, dims["d_cond"], dt, stream),
        )
    if kind == "projected_attention":
        heads = dims["n_head"]
        depth = dims["d_hidden"] // heads
        # Triangle rows are independent augmentation streams sharing one pair bias.
        aug = length if stream == "token_pair" else b
        qkv = tuple(
            torch.randn(aug, 1, heads, length, depth, device="cuda", dtype=dt)
            for _ in range(3)
        )
        bias = torch.randn(1, heads, length, length, device="cuda", dtype=dt)
        mask = torch.ones(aug, 1, length, device="cuda", dtype=torch.bool)
        return (*qkv, bias, mask)
    width = dims.get("d_norm", dims.get("d_hidden"))
    if stream == "atom_pair":
        # Atom pair windows: Q=32, K=128. Never allocate a dense N_atom squared pair.
        return (
            torch.randn(
                b, (length + 31) // 32, 32, 128, width, device="cuda", dtype=dt
            ),
        )
    x = _activation(b, length, width, dt, stream)
    return (x, torch.randn_like(x)) if kind == "gated_linear" else (x,)


def cases():
    from miniworld_engine.autotune.builder import Case, _mask, _pair, _shapes
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.primitives import LayerNorm
    from miniworld_engine.modules.triangle_attention import TriangleAttention

    result = [
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

    def make(kind):
        return Case(
            kind,
            lambda dims, p, impl, dt: Leaf(kind, dims, dt),
            lambda b, l, dims, dt, s: _inputs(kind, b, l, dims, dt, s),
            **_shapes(kind),
        )

    result.extend(
        make(kind)
        for kind in NAMES
        if kind not in ("triangle_attention_checkpoint", "layernorm_native")
    )
    return result
