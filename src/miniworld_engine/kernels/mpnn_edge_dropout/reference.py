"""PyTorch reference for ProteinMPNN encoder edge dropout.

This is what `interface.py` falls back to, and what the bit-packed backend has to agree with. It
was inline in the interface: the same function was both the public door's default branch and the
oracle a checker would compare against, so there was nothing to compare against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_dropout_pytorch(
    values: torch.Tensor,
    probability: float,
    *,
    training: bool,
    inplace: bool = False,
) -> torch.Tensor:
    """Native dropout, keeping ATen's boolean mask for the backward.

    The bit-packed backend produces the identical forward -- it calls this same native dropout --
    and differs only in storing that mask one bit per element instead of one byte.
    """
    return F.dropout(values, p=probability, training=training, inplace=inplace)


__all__ = ["edge_dropout_pytorch"]
