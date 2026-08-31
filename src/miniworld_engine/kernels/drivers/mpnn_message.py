"""Drivers for the ``mpnn_message`` family.

The shape is the mpnn graph, so it comes from `drivers/mpnn_edge_tail.py` like the other two
families' -- see `DRIVER_SHAPE_OWNERS`. What differs is the operand set: this family projects the
edge states and reduces over the neighbour axis, so it takes one square weight and a mask rather
than the edge tail's three weights and norm parameters.

All four drivers run the same launch. `_forward_op` reaches the projection and the forward reduce;
the backward reaches the dX and the bias reduce. That is not a shortcut -- those launches are the
only way production reaches these kernels, so a driver that got to them any other way would be
tuning a launch nothing makes.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.mpnn_edge_tail import _NEIGHBORS, _graph, _nodes


def _inputs(*, grad: bool):
    t = _graph(grad=grad)
    mask = (torch.rand(1, _nodes(), _NEIGHBORS, device=dev()) > 0.2).to(BF16)
    return t["edge_states"], t["edge_weight"], t["hidden_bias"], mask


def _message(*, backward: bool) -> None:
    from miniworld_engine.kernels.mpnn_message.interface import message_hidden_reduce

    preactivation, weight, bias, mask = _inputs(grad=backward)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = message_hidden_reduce(
            preactivation, weight, bias, mask, _NEIGHBORS, backend="triton",
        )
    if backward:
        out.sum().backward()


def mpnn_message_fwd_gemm_triton() -> None:
    _message(backward=False)


def mpnn_message_fwd_gelu_reduce_triton() -> None:
    _message(backward=False)


def mpnn_message_bwd_dx_triton() -> None:
    _message(backward=True)


def mpnn_message_bwd_reduce_dbias_triton() -> None:
    _message(backward=True)
