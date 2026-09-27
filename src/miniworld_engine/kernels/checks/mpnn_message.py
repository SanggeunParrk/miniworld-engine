"""Accuracy checks for the ``mpnn_message`` family.

The inputs come from this family's DRIVER, so the reference answers the question the build measured.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _fixed, _grads
from miniworld_engine.kernels.drivers.mpnn_message import _NEIGHBORS, _inputs


def _pair():
    from miniworld_engine.kernels.mpnn_message.interface import message_hidden_reduce
    from miniworld_engine.kernels.mpnn_message.reference import (
        message_hidden_reduce_pytorch,
    )

    _fixed()
    preactivation, weight, bias, mask = _inputs(grad=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = message_hidden_reduce(
            preactivation, weight, bias, mask, _NEIGHBORS, backend="triton")
    ref = message_hidden_reduce_pytorch(
        preactivation.float(), weight.float(), bias.float(), mask, _NEIGHBORS)
    return out, ref


def _gradients():
    from miniworld_engine.kernels.mpnn_message.interface import message_hidden_reduce
    from miniworld_engine.kernels.mpnn_message.reference import (
        message_hidden_reduce_pytorch,
    )

    _fixed()
    preactivation, weight, bias, mask = _inputs(grad=True)
    names = ("preactivation", "weight", "bias")

    def kernel(preactivation, weight, bias):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return message_hidden_reduce(
                preactivation, weight, bias, mask, _NEIGHBORS, backend="triton")

    def reference(preactivation, weight, bias):
        return message_hidden_reduce_pytorch(
            preactivation, weight, bias, mask, _NEIGHBORS)

    return _grads(kernel, [preactivation, weight, bias], reference, names)


def mpnn_message_fwd_gemm_triton():
    """_projection_fwd_kernel: the square projection every other stage reads."""
    return _pair()


def mpnn_message_fwd_gelu_reduce_triton():
    """_gelu_reduce_fwd_kernel: GELU, then the masked reduction over the k neighbours."""
    return _pair()


def mpnn_message_bwd_dx_triton():
    """_projection_dx_kernel, through every gradient the family produces."""
    return _gradients()


def mpnn_message_bwd_reduce_dbias_triton():
    """_gelu_reduce_db_bwd_kernel: the reduction's backward and the bias gradient with it."""
    return _gradients()
