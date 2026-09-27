"""PyTorch reference for the relative-position embedding lookup."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_position_embed_pytorch(
    bucket: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Look up one row per edge and add the shared bias.

    This is the whole operation. The fused path exists for its *backward*, which is a
    reduction of one row per edge into a table of a few dozen, not for the forward
    gather.
    """
    return F.embedding(bucket, table) + bias


__all__ = ["relative_position_embed_pytorch"]
