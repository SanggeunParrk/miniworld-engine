"""Drivers for the ``mpnn_relative_position`` family."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.mpnn_edge_tail import (
    _BUCKET_WIDTH,
    _BUCKETS,
    _NEIGHBORS,
    _nodes,
)


def mpnn_relative_position_bwd_reduce_triton() -> None:
    from miniworld_engine.kernels.mpnn_relative_position.triton.main import (
        triton_bucket_reduce,
    )

    d, rows = dev(), _nodes() * _NEIGHBORS
    grad_output = torch.randn(1, rows, _BUCKET_WIDTH, device=d, dtype=BF16)
    bucket = torch.randint(0, _BUCKETS, (1, rows), device=d, dtype=torch.long)
    triton_bucket_reduce(grad_output, bucket, _BUCKETS)
