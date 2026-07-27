"""Explicit compatibility boundary for the frozen CSSB forward signature."""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import ProteinMPNN


class CSSBForwardAdapter(nn.Module):
    """Accept the legacy unused ``loss_mask`` argument around a clean model."""

    def __init__(self, model: ProteinMPNN) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        backbone: torch.Tensor,
        sequence: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        decoding_order: torch.Tensor,
        patch_index: torch.Tensor,
        loss_mask: torch.Tensor,
        segment_lengths: torch.Tensor | None,
        *,
        use_checkpoint: bool = False,
        return_log_prob: bool = False,
    ) -> torch.Tensor:
        del loss_mask
        return self.model(
            backbone,
            sequence,
            residue_mask,
            residue_index,
            chain_index,
            decoding_order,
            patch_index,
            segment_lengths,
            checkpoint_layers=use_checkpoint,
            return_log_prob=return_log_prob,
        )
