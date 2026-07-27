"""Triton implementation of the fused ProteinMPNN encoder node message."""

from .main import triton_node_message_reduce

__all__ = ["triton_node_message_reduce"]
