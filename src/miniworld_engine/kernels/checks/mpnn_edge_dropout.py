"""Accuracy checks for the ``mpnn_edge_dropout`` family.

The mask is the input, not a draw: both kernels take a mask someone else produced -- the forward
packs it to one bit per element and the backward reads it back -- so the reference is the same
packing and the same gather, in fp32, over the same mask.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _fixed
from miniworld_engine.kernels.drivers.mpnn_edge_dropout import _mask
from miniworld_engine.kernels.drivers.mpnn_edge_tail import _graph


def mpnn_edge_dropout_fwd_packed_triton():
    """_pack_bool_kernel: one bit per element, checked by unpacking it again."""
    from miniworld_engine.kernels.mpnn_edge_dropout.triton.main import _pack_op

    _fixed()
    mask = _mask().contiguous()
    packed = _pack_op(mask)
    bits = torch.arange(mask.numel(), device=mask.device)
    unpacked = ((packed[bits // 8].to(torch.int32) >> (bits % 8)) & 1) != 0
    return unpacked.float(), mask.reshape(-1).float()


def mpnn_edge_dropout_bwd_packed_triton():
    """_packed_dropout_backward_kernel: the gradient straight off the packed mask."""
    from miniworld_engine.kernels.mpnn_edge_dropout.triton.main import (
        _backward_op,
        _pack_op,
    )

    _fixed()
    mask = _mask().contiguous()
    grad = torch.randn_like(_graph()["edge_states"]).contiguous()
    scale = 1.0 / 0.9
    got = _backward_op(grad, _pack_op(mask), scale)
    ref = torch.where(mask, grad.float() * scale, torch.zeros((), device=grad.device))
    return got, ref
