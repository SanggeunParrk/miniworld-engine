"""Triangle multiplicative update (model-level op connecting tm1 / tm2 kernels)."""

from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_engine.modules.triangle_multiplication.module import (
    TriangleMultiplication,
)

__all__ = ["BidirectionalTriangleMultiplication", "TriangleMultiplication"]
