"""PyTorch reference for the ProteinMPNN encoder edge LayerNorm.

What `interface.py` falls back to, and what the memory-saving backend has to agree with. It used
to be an `F.layer_norm` call inline in the interface, which left the family with a public door and
no oracle behind it: the same expression was both the default branch and the thing a checker would
have compared against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_layer_norm_pytorch(
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """LayerNorm over the last dimension, the shape ProteinMPNN's edge tensors carry.

    `normalized_shape` is `values.shape[-1:]` and not a caller-supplied tuple on purpose: every
    edge tensor in this family normalises its feature axis and nothing else, so a caller that
    could pass something else could only pass something wrong.
    """
    return F.layer_norm(values, values.shape[-1:], weight, bias, eps)


__all__ = ["edge_layer_norm_pytorch"]
