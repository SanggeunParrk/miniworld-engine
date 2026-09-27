"""Triton implementation of the fused ProteinMPNN encoder edge tail."""

from .main import triton_edge_tail_update

__all__ = ["triton_edge_tail_update"]
