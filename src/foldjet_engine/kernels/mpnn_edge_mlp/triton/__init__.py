"""Triton implementation of the ProteinMPNN edge-message MLP."""

from foldjet_engine.kernels.mpnn_edge_mlp.triton.main import triton_edge_mlp_update
from foldjet_engine.kernels.mpnn_edge_mlp.triton.compute import triton_edge_mlp_update_compute

__all__ = ["triton_edge_mlp_update", "triton_edge_mlp_update_compute"]
