"""Fused hidden projection and neighbor reduction for ProteinMPNN messages."""

from miniworld_engine.kernels.mpnn_message.interface import (
    MessageBackend,
    message_hidden_reduce,
)
from miniworld_engine.kernels.mpnn_message.reference import (
    message_hidden_reduce_pytorch,
)

__all__ = [
    "MessageBackend",
    "message_hidden_reduce",
    "message_hidden_reduce_pytorch",
]
