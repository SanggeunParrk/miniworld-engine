"""Compile boundaries for the unchanged upstream SWA primitives.

Only kernel launches (including upstream loading, validation and row selection)
are opaque. Projections, RoPE, window construction and module composition remain
visible to Inductor. Loading stays lazy and happens outside Dynamo tracing.
These inference-only ops allocate fresh outputs and never mutate their inputs.
"""

import torch
from torch import Tensor

from miniworld_engine.integrations import anthropic as upstream


@torch.library.custom_op("miniworld_anthropic::swa_rms", mutates_args=())
def rms(x: Tensor, scale: Tensor | None, shift: Tensor | None, eps: float) -> Tensor:
    return upstream.carried_kernel("dtk_kernels").ln_modulate(
        x, scale, shift, rms=True, eps=eps, sigmoid_scale=False)


@rms.register_fake
def _rms_fake(x, scale, shift, eps):
    return x.new_empty(x.shape)


@torch.library.custom_op("miniworld_anthropic::swa_gate", mutates_args=())
def gate(x: Tensor, gate: Tensor, res: Tensor | None, sigmoid: bool) -> Tensor:
    return upstream.carried_kernel("dtk_kernels").gate_residual(
        x, gate=gate, res=res, sigmoid_gate=sigmoid)


@gate.register_fake
def _gate_fake(x, gate, res, sigmoid):
    return x.new_empty(x.shape, dtype=x.dtype if res is None else res.dtype)


@torch.library.custom_op("miniworld_anthropic::swa_swiglu", mutates_args=())
def swiglu(ab: Tensor) -> Tensor:
    return upstream.carried_kernel("dtk_kernels").swiglu(ab)


@swiglu.register_fake
def _swiglu_fake(ab):
    return ab.new_empty((ab.shape[0], ab.shape[1] // 2))


@torch.library.custom_op("miniworld_anthropic::swa_gather", mutates_args=())
def gather(q: Tensor, k: Tensor, v: Tensor, bias: Tensor, indices: Tensor,
           heads: int, scale: float) -> Tensor:
    return upstream.carried_kernel("gather_attn").gather_attn(
        q, k, v, bias, indices, heads, scale=scale,
        ensure_sorted=False, allow_candidate=True)


@gather.register_fake
def _gather_fake(q, k, v, bias, indices, heads, scale):
    return q.new_empty(q.shape, dtype=v.dtype)
