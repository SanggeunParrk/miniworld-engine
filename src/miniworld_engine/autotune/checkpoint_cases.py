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
    # _op_gated_residual: x + gate * branch, SWADiTBlock's per-op adaLN-Zero residual (atom stream, augmented rows)
    "gated_residual",
    "triangle_attention_checkpoint",
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
        elif kind == "swiglu_ffn":
            for name in ("wa", "wb"):
                weight(name, (dims["d_expanded"], dims["d_hidden"]))
            weight("ws", (dims["d_hidden"], dims["d_expanded"]))
        elif kind in ("layernorm_linear_native", "layernorm_linear_atom_output"):
            weight("norm", (dims["d_norm"],), torch.float32)
            weight("weight", (dims["n_head"], dims["d_norm"]))

    def forward(self, *args):
        from miniworld_engine.kernels.augmented_attention.whole_op import augmented_attention_pair_bias as _op_augmented_attention_pair_bias
        from miniworld_engine.kernels.gated_projection.interface import gated_residual as _op_gated_residual
        from miniworld_engine.kernels.layernorm_linear.interface import layer_norm_linear as _op_layer_norm_linear
        from miniworld_engine.kernels.rmsnorm.interface import rms_norm_modulation as _op_rms_norm_modulation
        from miniworld_engine.kernels.transition.interface import swiglu_ffn as _op_swiglu_ffn
        if self.kind == "rms_norm_modulation":
            y, gate = _op_rms_norm_modulation(
                *args,
                self.wscale,
                self.wshift,
                self.wgate,
                eps=torch.finfo(args[0].dtype).eps,
            )
            return y + gate
        if self.kind == "gated_residual":
            return _op_gated_residual(*args)
        if self.kind == "swiglu_ffn":
            return _op_swiglu_ffn(*args, self.wa, self.wb, self.ws)
        if self.kind in ("layernorm_linear_native", "layernorm_linear_atom_output"):
            return _op_layer_norm_linear(*args, self.norm, self.weight)
        if self.kind == "projected_attention":
            return _op_augmented_attention_pair_bias(*args)
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
    if kind == "gated_residual":
        return (x, torch.randn_like(x), torch.randn_like(x))
    return (x,)


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
