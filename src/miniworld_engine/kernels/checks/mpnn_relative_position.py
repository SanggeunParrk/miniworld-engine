"""Accuracy checks for the ``mpnn_relative_position`` family.

The shapes come from this family's DRIVER, so the reference answers the question the build
actually measured rather than one of its own.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _fixed
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.mpnn_edge_tail import (
    _BUCKET_WIDTH,
    _BUCKETS,
    _NEIGHBORS,
    _nodes,
)


def mpnn_relative_position_bwd_reduce_triton():
    """_bucket_reduce_kernel: (grad_table, grad_bias) against index_add and a plain column sum.

    Not `F.embedding`'s backward: that is the thing this kernel replaces, and it sorts the index
    to do the reduction, so comparing against it would compare two orderings rather than the
    arithmetic. `index_add_` on an fp32 copy is the same sum in the obvious order.
    """
    _fixed()
    from miniworld_engine.kernels.mpnn_relative_position.triton.main import (
        triton_bucket_reduce,
    )

    d, rows = dev(), _nodes() * _NEIGHBORS
    grad_output = torch.randn(1, rows, _BUCKET_WIDTH, device=d, dtype=BF16)
    bucket = torch.randint(0, _BUCKETS, (1, rows), device=d, dtype=torch.long)

    table, bias = triton_bucket_reduce(grad_output, bucket, _BUCKETS)
    flat = grad_output.float().reshape(-1, _BUCKET_WIDTH)
    ref_table = torch.zeros(_BUCKETS, _BUCKET_WIDTH, device=d, dtype=torch.float32)
    ref_table.index_add_(0, bucket.reshape(-1), flat)
    return {"grad_table": (table, ref_table), "grad_bias": (bias, flat.sum(dim=0))}
