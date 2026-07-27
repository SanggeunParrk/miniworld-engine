"""Item-balanced training objectives for ProteinMPNN."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ItemBalancedLossStatistics:
    """Detached, additive statistics suitable for sum reduction across ranks."""

    item_loss_sum: torch.Tensor
    active_item_count: torch.Tensor
    supervised_token_count: torch.Tensor
    supervision_weight: torch.Tensor

    @property
    def mean_loss(self) -> torch.Tensor:
        denominator = self.active_item_count.to(self.item_loss_sum.dtype).clamp_min(1)
        return self.item_loss_sum / denominator

    def __add__(self, other: ItemBalancedLossStatistics) -> ItemBalancedLossStatistics:
        if not isinstance(other, ItemBalancedLossStatistics):
            return NotImplemented
        return type(self)(
            item_loss_sum=self.item_loss_sum + other.item_loss_sum,
            active_item_count=self.active_item_count + other.active_item_count,
            supervised_token_count=(
                self.supervised_token_count + other.supervised_token_count
            ),
            supervision_weight=self.supervision_weight + other.supervision_weight,
        )


@dataclass(frozen=True)
class ItemBalancedLoss:
    """Differentiable local loss plus explicit item-level reduction state."""

    loss: torch.Tensor
    item_loss_sum: torch.Tensor
    per_item_loss: torch.Tensor
    per_item_supervision_weight: torch.Tensor
    active_item_count: torch.Tensor
    supervised_token_count: torch.Tensor
    supervision_weight: torch.Tensor

    def for_ddp_backward(
        self,
        global_item_count: int | torch.Tensor,
        *,
        world_size: int,
    ) -> torch.Tensor:
        """Scale a local item sum for DDP's subsequent gradient averaging.

        ``global_item_count`` should be the sum of ``active_item_count`` over
        ranks. With standard DDP, which averages gradients over ``world_size``,
        this produces the gradient of the global mean over training items even
        when ranks receive different physical batch sizes.
        """
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if isinstance(global_item_count, Integral) and global_item_count <= 0:
            raise ValueError("global_item_count must be positive")
        denominator = torch.as_tensor(
            global_item_count,
            dtype=self.item_loss_sum.dtype,
            device=self.item_loss_sum.device,
        )
        if denominator.ndim != 0:
            raise ValueError("global_item_count must be a scalar")
        return self.item_loss_sum * world_size / denominator.clamp_min(1)

    def statistics(self) -> ItemBalancedLossStatistics:
        """Return additive values for logging or distributed sum reduction."""
        return ItemBalancedLossStatistics(
            item_loss_sum=self.item_loss_sum.detach(),
            active_item_count=self.active_item_count.detach(),
            supervised_token_count=self.supervised_token_count.detach(),
            supervision_weight=self.supervision_weight.detach(),
        )


def item_balanced_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    residue_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> ItemBalancedLoss:
    """Average masked token CE within each item, then average active items.

    ``logits`` follows the production model's raw-output layout ``[B, C, L]``.
    An item with no supervised residues is excluded from the item denominator;
    its per-item loss is zero. Reduction is accumulated in FP32 for half and
    bfloat16 model outputs.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, C, L], got {logits.shape}")
    if target.ndim != 2:
        raise ValueError(f"target must have shape [B, L], got {target.shape}")
    batch, _, length = logits.shape
    expected = (batch, length)
    if target.shape != expected:
        raise ValueError(
            f"target must have shape {expected}, got {tuple(target.shape)}"
        )
    if target.dtype != torch.long:
        raise TypeError("target must have dtype torch.long")
    if loss_mask.shape != expected:
        raise ValueError(
            f"loss_mask must have shape {expected}, got {tuple(loss_mask.shape)}"
        )
    if residue_mask is not None and residue_mask.shape != expected:
        raise ValueError(
            f"residue_mask must have shape {expected}, got {tuple(residue_mask.shape)}"
        )

    loss_logits = (
        logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    )
    token_loss = F.cross_entropy(
        loss_logits,
        target,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    reduction_dtype = (
        torch.float32
        if token_loss.dtype in (torch.float16, torch.bfloat16)
        else token_loss.dtype
    )
    token_loss = token_loss.to(reduction_dtype)
    supervision_weight = loss_mask.to(reduction_dtype)
    if residue_mask is not None:
        supervision_weight = supervision_weight * residue_mask.to(reduction_dtype)
    valid_target = target != ignore_index
    supervision_weight = supervision_weight * valid_target

    per_item_supervision_weight = supervision_weight.sum(dim=-1)
    active_items = per_item_supervision_weight > 0
    per_item_numerator = (token_loss * supervision_weight).sum(dim=-1)
    per_item_loss = torch.where(
        active_items,
        per_item_numerator / per_item_supervision_weight.clamp_min(1),
        torch.zeros_like(per_item_numerator),
    )
    item_loss_sum = per_item_loss.sum()
    active_item_count = active_items.sum()
    loss = item_loss_sum / active_item_count.to(reduction_dtype).clamp_min(1)

    return ItemBalancedLoss(
        loss=loss,
        item_loss_sum=item_loss_sum,
        per_item_loss=per_item_loss,
        per_item_supervision_weight=per_item_supervision_weight,
        active_item_count=active_item_count,
        supervised_token_count=(supervision_weight > 0).sum(),
        supervision_weight=per_item_supervision_weight.sum(),
    )


__all__ = [
    "ItemBalancedLoss",
    "ItemBalancedLossStatistics",
    "item_balanced_cross_entropy",
]
