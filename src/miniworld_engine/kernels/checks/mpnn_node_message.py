"""Accuracy checks for the ``mpnn_node_message`` family."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _grads
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.mpnn_edge_tail import _NEIGHBORS, _graph, _nodes


def _node_message_inputs():
    t = _graph(grad=True)
    mask = (torch.rand(1, _nodes(), _NEIGHBORS, device=dev()) > 0.2).to(BF16)
    names = ("edge_states", "query_projection", "neighbor_projection", "edge_weight",
             "hidden_weight", "hidden_bias")
    return t, mask, names, [t[n] for n in names]


def _node_message_pair():
    from miniworld_engine.kernels.mpnn_node_message.reference import (
        node_message_reduce_pytorch,
    )
    from miniworld_engine.kernels.mpnn_node_message.triton.main import (
        triton_node_message_reduce,
    )

    t, mask, names, _leaves = _node_message_inputs()
    index = t["flat_neighbor_indices"]
    args = [t[n] for n in names]
    out = triton_node_message_reduce(
        args[0], args[1], args[2], index, args[3], args[4], args[5], mask, _NEIGHBORS)
    ref = node_message_reduce_pytorch(
        *(a.float() for a in args[:3]), index, *(a.float() for a in args[3:]),
        mask.float(), _NEIGHBORS)
    return out, ref


def _node_message_grads():
    from miniworld_engine.kernels.mpnn_node_message.reference import (
        node_message_reduce_pytorch,
    )
    from miniworld_engine.kernels.mpnn_node_message.triton.main import (
        triton_node_message_reduce,
    )

    t, mask, names, leaves = _node_message_inputs()
    index = t["flat_neighbor_indices"]

    def kernel(*a):
        return triton_node_message_reduce(
            a[0], a[1], a[2], index, a[3], a[4], a[5], mask, _NEIGHBORS)

    def reference(*a):
        return node_message_reduce_pytorch(
            a[0], a[1], a[2], index, a[3], a[4], a[5], mask.float(), _NEIGHBORS)

    return _grads(kernel, leaves, reference, names)


def mpnn_node_message_fwd_gemm_triton():
    """_node_message_fwd_kernel: project, GELU, and reduce over the k neighbours."""
    return _node_message_pair()


def mpnn_node_message_bwd_recompute_triton():
    """_node_message_replay_kernel: the forward recomputed in backward, from its gradients."""
    return _node_message_grads()


def mpnn_node_message_bwd_dx_triton():
    """_node_message_dx_kernel: the dX pass, from the same gradients."""
    return _node_message_grads()
