"""Fused encoder node message: packed W1 block, both GELUs, W2, masked reduction."""

from .interface import (
    NodeMessageBackend,
    node_message_reduce,
    node_message_supported,
)
from .reference import node_message_reduce_pytorch

__all__ = [
    "NodeMessageBackend",
    "node_message_reduce",
    "node_message_reduce_pytorch",
    "node_message_supported",
]
