"""Fused encoder edge tail: packed W1 block, edge MLP, dropout, residual, norm."""

from .interface import EdgeTailBackend, edge_tail_supported, edge_tail_update
from .reference import edge_tail_update_pytorch

__all__ = [
    "EdgeTailBackend",
    "edge_tail_supported",
    "edge_tail_update",
    "edge_tail_update_pytorch",
]
