"""Drivers for the ``mpnn_edge_dropout`` family.

Elementwise over the edge tensor, so the only shape that matters is its element count -- which is
the mpnn graph's, taken from `drivers/mpnn_edge_tail.py` like every other family here.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import dev
from miniworld_engine.kernels.drivers.mpnn_edge_tail import _graph


def _mask() -> torch.Tensor:
    # The kernel's operand is the BOOL this comparison produces; fp32 is only what
    # `torch.rand` draws in, and the activation dtype never reaches these kernels.
    return torch.rand(_graph()["edge_states"].shape, device=dev()) > 0.1


def mpnn_edge_dropout_fwd_packed_triton() -> None:
    from miniworld_engine.kernels.mpnn_edge_dropout.triton.main import _pack_op

    _pack_op(_mask().contiguous())


def mpnn_edge_dropout_bwd_packed_triton() -> None:
    from miniworld_engine.kernels.mpnn_edge_dropout.triton.main import (
        _backward_op,
        _pack_op,
    )

    mask = _mask().contiguous()
    grad = torch.randn_like(_graph()["edge_states"])
    _backward_op(grad.contiguous(), _pack_op(mask), 1.0 / 0.9)
