"""Public entry point for the augmented-attention (pair-bias) family.

Fused attention with an additive pair bias and an optional mask, in two backends with an
identical ``(q, k, v, bias, mask)`` signature:

  * **compute-efficient** (``triton/main.py``, the default): stores partial query
    gradients per KV tile and bias gradients per augmentation, then reduces them.
  * **memory-efficient** (``triton/memory_efficient.py``): atomically accumulates
    query and shared bias gradients without those expanded buffers.

Both backends recompute attention probabilities in backward. Atomic accumulation
uses less workspace, but floating-point accumulation order can vary between runs.

The choice is a per-call kwarg rather than two exported names, so this module is the family's
single public door and the backend split stays an implementation detail of the family.
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
    """Fused augmented attention with pair bias.

    ``compute_efficient`` (default ``True``) selects split-buffer gradient
    accumulation. Pass ``False`` for atomic accumulation with lower workspace
    memory. Both backends recompute attention probabilities in backward.
    """
    fn = (
        _pair_bias_compute_efficient
        if compute_efficient
        else _pair_bias_memory_efficient
    )
    return fn(query, key, value, bias, mask)
