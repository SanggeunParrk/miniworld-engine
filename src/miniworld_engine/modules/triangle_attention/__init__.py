"""Triangle attention (model-level op connecting the triangle-attention kernel)."""

from miniworld_engine.modules.triangle_attention.bidirectional import (
    BidirectionalTriangleAttention,
)
from miniworld_engine.modules.triangle_attention.module import (
    TriangleAttention,
    TrianglePairAttention,
)

__all__ = [
    "BidirectionalTriangleAttention",
    "TriangleAttention",
    "TrianglePairAttention",
]
