"""Dispatch boundary for the relative-position embedding.

Three backends, because the measurement did not pick the obvious winner. ``index_add``
carries no kernel at all -- it is the same autograd boundary with ``Tensor.index_add_``
underneath -- and on the real index distribution it beat a first hand-written Triton
reduction. It is kept as a first-class choice rather than a fallback for exactly that
reason: the point is to stop paying 30.8 ms per call, not to run a kernel.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


# No ``auto``: the boundary changes which reduction runs in backward, and every
# reduction orders its additions differently, so it is reached only from an explicit
# policy. ``off`` keeps ``F.embedding`` and whatever the compiler makes of it.
RelativePositionBackend = Literal["off", "index_add", "triton"]

# The reduction privatises a whole table per program, so a table that is not small
# stops fitting. 4096 rows at 128 channels is 2 MiB of accumulator, far past any
# sensible point; the real table is 66 x 16.
_MAX_BUCKETS = 512
_MAX_WIDTH = 128


def relative_position_supported(
    bucket: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    """Check the contract without allocating anything."""
    if bias is None:
        return False
    tensors = (bucket, table, bias)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not all(tensor.device == bucket.device for tensor in tensors):
        return False
    if bucket.dtype != torch.long or table.ndim != 2 or bias.ndim != 1:
        return False
    buckets, width = table.shape
    return (
        bucket.numel() > 0
        and bucket.is_contiguous()
        and table.is_contiguous()
        and bias.is_contiguous()
        and bias.shape[0] == width
        and 0 < buckets <= _MAX_BUCKETS
        and 0 < width <= _MAX_WIDTH
        and table.dtype == bias.dtype
        and table.dtype in {torch.float32, torch.bfloat16}
    )


def relative_position_embed(
    bucket: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
    backend: RelativePositionBackend = "off",
) -> torch.Tensor:
    """Look up one table row per edge and add the shared bias."""
    if backend == "off":
        return F.embedding(bucket, table) + bias
    if not relative_position_supported(bucket, table, bias):
        raise ValueError(
            "the fused MPNN relative-position embedding requires a contiguous CUDA "
            "INT64 bucket index, a contiguous 2-D FP32 or BF16 table of at most "
            f"{_MAX_BUCKETS} rows and {_MAX_WIDTH} channels, and a contiguous bias "
            "of matching width and dtype"
        )
    # Keep Triton out of CPU and import-only users; only supported CUDA tensors here.
    from .triton import relative_position_embed_op

    return relative_position_embed_op(bucket, table, bias, backend)


__all__ = [
    "RelativePositionBackend",
    "relative_position_embed",
    "relative_position_supported",
]
