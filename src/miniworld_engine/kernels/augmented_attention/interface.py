"""Augmented attention with split or atomic gradient accumulation.

Both backends recompute attention scores in backward. The split backend keeps
per-augmentation bias gradients and per-key-tile query gradients before reducing
these buffers. The atomic backend accumulates into shared FP32 output buffers.
The default compute-efficient backend bounds temporary storage by query chunking.
The separate atomic backend is used only when explicitly requested.
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

def triton_augmented_attention_pair_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    compute_efficient: bool = True,
) -> torch.Tensor:
    """Fused pair-bias attention; compute-efficient is the default at every shape.

    Its backward bounds temporary storage by processing query chunks. Explicit
    ``False`` retains the separate atomic backend for callers that request it.
    """
    fn = _pair_bias_compute_efficient if compute_efficient else _pair_bias_memory_efficient
    return fn(query, key, value, bias, mask)
