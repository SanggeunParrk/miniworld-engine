"""Fused encoder node message: packed W1 block, both GELUs, W2, masked reduction."""

from miniworld_engine.kernels.mpnn_node_message.interface import (
    NodeMessageBackend,
    node_message_reduce,
    node_message_supported,
)
from miniworld_engine.kernels.mpnn_node_message.reference import (
    node_message_reduce_pytorch,
)

__all__ = [
    "NodeMessageBackend",
    "node_message_reduce",
    "node_message_reduce_pytorch",
    "node_message_supported",
]
