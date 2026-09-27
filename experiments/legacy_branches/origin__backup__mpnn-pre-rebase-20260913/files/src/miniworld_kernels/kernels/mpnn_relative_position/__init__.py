"""Relative-position embedding whose backward is a bucket reduction, not a scatter."""

from .interface import (
    RelativePositionBackend,
    relative_position_embed,
    relative_position_supported,
)
from .reference import relative_position_embed_pytorch

__all__ = [
    "RelativePositionBackend",
    "relative_position_embed",
    "relative_position_embed_pytorch",
    "relative_position_supported",
]
