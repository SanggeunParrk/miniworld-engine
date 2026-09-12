"""Accuracy checks for the ``mpnn_edge_mlp`` family."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _fixed
from miniworld_engine.kernels.drivers.mpnn_edge_tail import _graph
from miniworld_engine.kernels.mpnn_edge_mlp import EdgeMLPBackend


def _pair(backend: EdgeMLPBackend):
    from miniworld_engine.kernels.mpnn_edge_mlp.interface import edge_mlp_update
    from miniworld_engine.kernels.mpnn_edge_mlp.reference import edge_mlp_update_pytorch

    _fixed()
    t = _graph()
    args = (t["edge_states"], t["hidden_weight"], t["hidden_bias"],
            t["output_weight"], t["output_bias"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = edge_mlp_update(*args, backend=backend)
    return out, edge_mlp_update_pytorch(*(a.float() for a in args))


def mpnn_edge_mlp_fwd_gemm_b2b_triton():
    """_edge_mlp_fwd_kernel: both projections in one launch, so no column tile."""
    return _pair("triton_memory")


def mpnn_edge_mlp_fwd_gemm_triton():
    """_compute_stage_fwd_kernel: one projection per launch, and it takes the full axis set."""
    return _pair("triton_compute")
