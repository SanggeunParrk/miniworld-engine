"""Triton implementation of the fused ProteinMPNN encoder edge tail."""

from miniworld_engine.kernels.mpnn_edge_tail.triton.main import triton_edge_tail_update

__all__ = ["triton_edge_tail_update"]
