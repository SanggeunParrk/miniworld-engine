"""Triangle attention (model-level op connecting the triangle-attention kernel)."""

from .bidirectional import BidirectionalTriangleAttention
from .module import TriangleAttention, TrianglePairAttention

__all__ = [
    "BidirectionalTriangleAttention",
    "TriangleAttention",
    "TrianglePairAttention",
]
