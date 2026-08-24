"""Public entry point for the augmented-attention (pair-bias) family.

Fused attention with an additive pair bias and an optional mask, in two backends with an
identical ``(q, k, v, bias, mask)`` signature:

  * **compute-efficient** (``triton/main.py``, the default): stores the attention
    probabilities so the backward reuses them instead of recomputing. Faster across the
    benchmarked shapes (wins for L up to ~1024); costs more activation memory.
  * **memory-efficient** (``triton/memory_efficient.py``): flash-style, recomputes the
    attention in the backward. Lower memory, preferable on memory-tight or very long-L cases.

The choice is a per-call kwarg rather than two exported names, so this module is the family's
single public door and the backend split stays an implementation detail of the family.
"""

from __future__ import annotations

import torch

from .triton.main import (
    triton_augmented_attention_pair_bias as _pair_bias_compute_efficient,
)
from .triton.memory_efficient import (
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

    ``compute_efficient`` (default ``True``) selects the compute-efficient backend
    (stores attention probabilities; faster at the benchmarked shapes). Pass
    ``False`` for the memory-efficient flash-style backend on memory-tight or
    very long-L cases. Both backends carry a real backward and are numerically
    equivalent to the torch reference.
    """
    fn = (
        _pair_bias_compute_efficient
        if compute_efficient
        else _pair_bias_memory_efficient
    )
    return fn(query, key, value, bias, mask)
