"""Augmented attention with split or atomic gradient accumulation.

Both backends recompute attention scores in backward. The split backend keeps
per-augmentation bias gradients and per-key-tile query gradients before reducing
these buffers. The atomic backend accumulates into shared FP32 output buffers.
Automatic selection bounds the quadratic temporary storage for large training
shapes; explicit backend requests remain available for tuning and comparisons.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.augmented_attention.triton.main import (
    triton_augmented_attention_pair_bias as _pair_bias_compute_efficient,
)
from miniworld_engine.kernels.augmented_attention.triton.memory_efficient import (
    triton_augmented_attention_pair_bias as _pair_bias_memory_efficient,
)

__all__ = ["triton_augmented_attention_pair_bias"]

# A storage budget, not a measured performance crossover. The split backend's
# query-gradient workspace is additional to this per-augmentation bias buffer.
_SPLIT_BIAS_BUDGET_BYTES = 1 << 30


def _split_bias_bytes(shape):
    a, b, length, heads, _ = shape
    return a * b * heads * length * length * 4


def triton_augmented_attention_pair_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    compute_efficient: bool | None = None,
) -> torch.Tensor:
    """Fused attention with optional automatic selection of backward storage.

    ``None`` uses atomic accumulation when training would require more than 1 GiB
    for the split backend's unreduced bias gradient. Shape-only selection supports
    torch.compile and CUDA graph capture without querying free device memory.
    ``True`` explicitly selects split accumulation; ``False`` selects atomic.
    Atomic FP32 accumulation can change the order of floating-point additions.
    """
    if compute_efficient is None:
        training = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query, key, value, bias)
        )
        compute_efficient = not training or _split_bias_bytes(query.shape) <= _SPLIT_BIAS_BUDGET_BYTES
    fn = _pair_bias_compute_efficient if compute_efficient else _pair_bias_memory_efficient
    return fn(query, key, value, bias, mask)
