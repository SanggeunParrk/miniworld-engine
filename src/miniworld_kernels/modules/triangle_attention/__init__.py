"""Triangle attention (model-level op connecting the triangle-attention kernel)."""

from .module import TriangleAttention, TrianglePairAttention

__all__ = ["TriangleAttention", "TrianglePairAttention"]
