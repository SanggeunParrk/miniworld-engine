"""Public entry point for the triangle-attention (pair-bias) family.

Fused triangular self-attention over a pair activation with an additive pair bias, carrying
its own backward. This module is the family's public door: importers name it rather than the
``triton/`` layout.

The door is also where the forward is chosen: the qualified H100 bf16 D32 hand-CUDA training forward
(``cuda/training_forward.py``, L in 384/768/1024) where it applies, the portable Triton forward otherwise and always under
``settings.force_portable_kernels``. ``triton/main.py`` holds the pure Triton kernels and never reaches for CUDA.
"""

from __future__ import annotations

from miniworld_engine import settings
from miniworld_engine.kernels.triangle_attention.triton.main import (
    TritonTriangleAttentionPairBiasFunction,
)

__all__ = ["triton_triangle_attention_pair_bias"]


def triton_triangle_attention_pair_bias(q, k, v, bias):
    """``(B, H, L, L2, D)`` triangle attention with a pair bias (autograd); see the module docstring for the forward routing."""
    if not settings.current().force_portable_kernels:
        bias = bias.contiguous()  # what the Triton entry does too; the CUDA forward takes only a contiguous bias
        from miniworld_engine.kernels.triangle_attention.cuda import training_forward

        if training_forward.can_use(q, k, v, bias):
            return training_forward.pair_bias_attention(q, k, v, bias)
    return TritonTriangleAttentionPairBiasFunction.apply(q, k, v, bias)
