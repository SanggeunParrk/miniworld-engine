"""Triangle multiplicative update (model-level op connecting tm1 / tm2 kernels)."""

from .bidirectional import BidirectionalTriangleMultiplication
from .module import TriangleMultiplication

__all__ = ["BidirectionalTriangleMultiplication", "TriangleMultiplication"]
