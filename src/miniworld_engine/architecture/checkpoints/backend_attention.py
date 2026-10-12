"""Backend routing for projected checkpoint attention, without changing weights.

The engine public pair-biased core supports square BF16 attention with D <= 64.
Triangle rows map to its augmentation axis: the pair bias stays shared and the
full per-row key mask is retained. Rectangular atom windows and FP32 coordinate
streams use the shared PyTorch implementation until the engine supports them.
"""

from __future__ import annotations

import math

import torch

from miniworld_engine.architecture.exceptions import ImplementationType


def engine_enabled(module: torch.nn.Module, x: torch.Tensor) -> bool:
    return (
        getattr(module, "foldforge_implementation", None)
        == ImplementationType.MINIWORLD_ENGINE
        and not module.training
        and x.is_cuda
        and x.dtype == torch.bfloat16
    )


def engine_attention_supported(
    module: torch.nn.Module, x: torch.Tensor, head_dim: int
) -> bool:
    return engine_enabled(module, x) and 0 < head_dim <= 64


def gated_projection(
    module: torch.nn.Module,
    linear: torch.nn.Module,
    gate: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Use the public engine gate/output op without moving the residual or outer gate."""
    if (
        engine_enabled(module, value)
        and gate.dtype == value.dtype
        and linear.weight.dtype == value.dtype
        and gate.shape == value.shape
    ):
        from miniworld_engine import ops

        return ops.gated_linear(gate, value, linear.weight, linear.bias)
    return linear(torch.sigmoid(gate) * value)


def layernorm_projection(
    norm: torch.nn.Module, linear: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """Fuse narrow bias-free LN+projection in FP32; retain native output dtype.

    Large output projections use the existing norm and GEMM: the current fused
    kernel is slower for those checkpoint shapes (see the wiring benchmark).
    """
    nh = linear.weight.shape[0]
    if (
        engine_enabled(norm, x)
        and norm.weight is not None
        and norm.bias is None
        and linear.bias is None
        and 0 < nh <= 16
    ):
        from miniworld_engine import ops

        value = x.unsqueeze(0) if x.ndim == 2 else x
        out = ops.layer_norm_linear(value, norm.weight, linear.weight, norm.eps)
        return out.squeeze(0) if x.ndim == 2 else out
    return linear(norm(x))


# Compatibility name for existing attention callers.
pair_bias_projection = layernorm_projection


def _masked_result(
    out: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    # Released checkpoints use a finite replacement/additive mask (-1e9).
    # A completely masked row therefore averages V, whereas the engine returns
    # zero for that row. Preserve the checkpoint convention, including padding.
    if mask is None:
        return out
    valid = mask.bool().any(dim=-1)[..., None, None, None]
    mean = value.float().mean(dim=-2, keepdim=True).to(value.dtype)
    return torch.where(valid, out, mean)


def engine_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Unscaled Q/K/V [..., row, head, token, D], bias [..., head, token, token]."""
    from miniworld_engine import ops

    rows, heads, length, dim = q.shape[-4:]
    batch = math.prod(q.shape[:-4])
    shape = q.shape
    q, k, v = [
        t.reshape(batch, rows, heads, length, dim).transpose(0, 1) for t in (q, k, v)
    ]
    bias = bias.reshape(batch, heads, length, length)
    if mask is not None:
        mask = mask.reshape(batch, rows, length).transpose(0, 1).bool()
    out = ops.augmented_attention_pair_bias(q, k, v, bias, mask)
    out = _masked_result(out, v, mask)
    return out.transpose(0, 1).reshape(shape)


def engine_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    num_aug: int = 1,
) -> torch.Tensor:
    """Square attention with explicit augmentation axis and shared pair bias.

    Leading Q/K/V dimensions flatten as [num_aug, batch]. Pair bias has only
    [batch, heads, token, token]; it is shared across diffusion samples.
    """
    from miniworld_engine import ops

    shape = q.shape
    heads, length, dim = shape[-3:]
    flat_batch = math.prod(shape[:-3])
    if num_aug < 1 or flat_batch % num_aug:
        raise ValueError("Q/K/V leading dimensions must contain num_aug samples")
    batch = flat_batch // num_aug
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Engine pair attention requires equal query/key/value shapes")
    target = (batch, heads, length, length)
    if bias is None:
        bias = q.new_zeros(target)
    elif num_aug == 1:
        bias = bias.expand(*shape[:-1], length).reshape(target)
    else:
        bias = bias.expand(target)
    if mask is not None:
        mask = mask.expand(*shape[:-3], length).reshape(num_aug, batch, length).bool()
    q, k, v = [t.reshape(num_aug, batch, heads, length, dim) for t in (q, k, v)]
    out = ops.augmented_attention_pair_bias(q, k, v, bias, mask)
    return _masked_result(out, v, mask).reshape(shape)
