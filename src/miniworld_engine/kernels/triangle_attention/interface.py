"""Public entry point for the triangle-attention (pair-bias) family.

Fused triangular self-attention over a pair activation with an additive pair bias, carrying
its own backward. This module is the family's public door: importers name it rather than the
``triton/`` layout.

The forward is the portable Triton one (``triton/main.py``, pure Triton kernels that never reach for CUDA); the H100 hand-CUDA
TriangleAttention lives at the module level (``integrations/triattn_h100_sm90``).
"""

from __future__ import annotations

from miniworld_engine.kernels.triangle_attention.triton.main import (
    TritonTriangleAttentionPairBiasFunction,
)

__all__ = ["triton_triangle_attention_pair_bias"]


def triton_triangle_attention_pair_bias(q, k, v, bias):
    """``(B, H, L, L2, D)`` triangle attention with a pair bias (autograd), on the portable Triton kernels."""
    return TritonTriangleAttentionPairBiasFunction.apply(q, k, v, bias)
