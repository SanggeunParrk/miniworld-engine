"""Public entry point for the triangle-attention (pair-bias) family.

Fused triangular self-attention over a pair activation with an additive pair bias, carrying
its own backward. This module is the family's public door: importers name it rather than the
``triton/`` layout.
"""

from __future__ import annotations

from miniworld_engine.kernels.triangle_attention.triton.main import (
    triton_triangle_attention_pair_bias,
)

__all__ = ["triton_triangle_attention_pair_bias"]
