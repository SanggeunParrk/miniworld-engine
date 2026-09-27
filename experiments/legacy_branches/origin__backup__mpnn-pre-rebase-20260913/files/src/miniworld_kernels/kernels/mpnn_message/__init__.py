"""Fused hidden projection and neighbor reduction for ProteinMPNN messages."""

from .interface import MessageBackend, message_hidden_reduce
from .reference import message_hidden_reduce_pytorch

__all__ = [
    "MessageBackend",
    "message_hidden_reduce",
    "message_hidden_reduce_pytorch",
]
