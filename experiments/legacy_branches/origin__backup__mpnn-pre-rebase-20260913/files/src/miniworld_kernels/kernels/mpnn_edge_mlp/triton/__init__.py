"""Triton implementation of the ProteinMPNN edge-message MLP."""

from .main import triton_edge_mlp_update
from .compute import triton_edge_mlp_update_compute

__all__ = ["triton_edge_mlp_update", "triton_edge_mlp_update_compute"]
