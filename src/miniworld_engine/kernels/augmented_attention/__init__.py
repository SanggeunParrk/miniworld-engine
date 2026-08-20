"""Augmented-attention (pair-bias) fused triton kernels.

Two backends with an identical ``(q, k, v, bias, mask)`` signature, selected per
call via the ``compute_efficient`` kwarg:

  * **compute-efficient** (``compute_efficient=True``, the default): stores the
    attention probabilities, so the backward reuses them instead of recomputing.
    Faster across the benchmarked shapes (wins for L up to ~1024); costs more
    activation memory.
  * **memory-efficient** (``compute_efficient=False``): flash-style, recomputes
    the attention in the backward. Lower memory, preferable on memory-tight or
    very long-L cases.

The two live in ``triton/main.py`` and ``triton/memory_efficient.py``.
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
