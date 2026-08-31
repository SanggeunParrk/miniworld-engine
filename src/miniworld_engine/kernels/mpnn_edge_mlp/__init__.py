"""Fused edge-message MLP used by the ProteinMPNN encoder."""

from miniworld_engine.kernels.mpnn_edge_mlp.interface import (
    EdgeMLPBackend,
    edge_mlp_update,
)
from miniworld_engine.kernels.mpnn_edge_mlp.reference import edge_mlp_update_pytorch

__all__ = ["EdgeMLPBackend", "edge_mlp_update", "edge_mlp_update_pytorch"]
