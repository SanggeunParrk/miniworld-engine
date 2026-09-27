"""Drivers for the ``mpnn_node_message`` family."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.mpnn_edge_tail import (
    _NEIGHBORS,
    _graph,
    _nodes,
)


def _node_message(*, backward: bool, compute: bool = False) -> None:
    from miniworld_engine.kernels.mpnn_node_message.triton.main import (
        triton_node_message_reduce,
        triton_node_message_reduce_compute,
    )

    t = _graph(grad=backward or compute)
    d = dev()
    edge_mask = (torch.rand(1, _nodes(), _NEIGHBORS, device=d) > 0.2).to(BF16)
    fn = triton_node_message_reduce_compute if compute else triton_node_message_reduce
    with torch.enable_grad():
        out = fn(
            t["edge_states"], t["query_projection"], t["neighbor_projection"],
            t["flat_neighbor_indices"], t["edge_weight"], t["hidden_weight"], t["hidden_bias"],
            edge_mask, _NEIGHBORS,
        )
        if backward:
            out.sum().backward()


def mpnn_node_message_fwd_gemm_triton() -> None:
    _node_message(backward=False)
    _node_message(backward=False, compute=True)


def mpnn_node_message_bwd_recompute_triton() -> None:
    _node_message(backward=True)


def mpnn_node_message_bwd_dx_triton() -> None:
    _node_message(backward=True)
