"""Static ProteinMPNN benchmark inputs and training objectives.

This module deliberately owns only workload semantics.  Model construction,
compilation, CUDA Graph capture, timing, and numerical comparisons remain in
the shared benchmark runner.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from miniworld_kernels.modules.mpnn import item_balanced_cross_entropy

MPNNLayout = Literal["single", "batch", "packed"]
MPNNTrainingObjective = Literal["output_grad", "item_ce"]
Backward = Callable[[torch.Tensor, torch.Tensor | None], None]


@dataclass(frozen=True)
class MPNNWorkload:
    """One static-shape workload shared by reference and production models."""

    coordinates: torch.Tensor
    sequence: torch.Tensor
    mask: torch.Tensor
    residue_index: torch.Tensor
    chain_index: torch.Tensor
    decoding_order: torch.Tensor
    patch_index: torch.Tensor
    segment_lengths: torch.Tensor | None
    upstream_gradient: torch.Tensor | None
    objective: MPNNTrainingObjective
    label_smoothing: float
    logical_batch_size: int
    physical_batch_size: int
    total_length: int

    def model_inputs(
        self,
        *,
        production_signature: bool,
    ) -> tuple[torch.Tensor | None, ...]:
        """Return inputs after coordinates in the requested model signature."""
        common = (
            self.sequence,
            self.mask,
            self.residue_index,
            self.chain_index,
            self.decoding_order,
            self.patch_index,
        )
        if production_signature:
            return (*common, self.segment_lengths)
        return (*common, self.mask, self.segment_lengths)

    def call_model(
        self,
        candidate: nn.Module,
        coordinates: torch.Tensor,
        *,
        production_signature: bool,
    ) -> torch.Tensor:
        """Invoke a reference or production model with identical workload data."""
        return candidate(
            coordinates,
            *self.model_inputs(production_signature=production_signature),
        )

    def backward(self, output: torch.Tensor, backward: Backward) -> None:
        """Apply the configured benchmark objective to ``output``."""
        if self.objective == "item_ce":
            loss = item_balanced_cross_entropy(
                output,
                self.sequence,
                self.mask,
                residue_mask=self.mask,
                label_smoothing=self.label_smoothing,
            ).loss
            backward(loss, None)
            return
        if self.upstream_gradient is None:
            msg = "output-gradient objective requires an upstream gradient"
            raise RuntimeError(msg)
        backward(output, self.upstream_gradient)


def build_mpnn_workload(
    *,
    seq_len: int,
    batch_size: int,
    layout: MPNNLayout,
    patch_size: int,
    k_neighbors: int,
    training: bool,
    coordinate_grad: bool,
    objective: MPNNTrainingObjective,
    label_smoothing: float,
    use_amp: bool,
    device: torch.device | str,
) -> MPNNWorkload:
    """Construct the static tensors used by one ProteinMPNN benchmark row.

    ``packed`` reproduces ProteinMPNN_CSSB's physical ``[1, B * L]`` collate
    convention. ``single`` and ``batch`` use independent physical rows
    ``[B, L]``.  Random tensors are intentionally created in the same order as
    the original in-runner implementation so seeded benchmark rows are stable.
    """
    if batch_size < 1:
        msg = "MPNN batch_size must be positive"
        raise ValueError(msg)
    if patch_size < 1:
        msg = "MPNN patch size must be positive"
        raise ValueError(msg)
    if layout == "single" and batch_size != 1:
        msg = "MPNN single layout requires batch_size=1; use mpnn_layout=batch"
        raise ValueError(msg)
    if not 0.0 <= label_smoothing <= 1.0:
        msg = "MPNN label smoothing must satisfy 0 <= value <= 1"
        raise ValueError(msg)
    if training and objective == "item_ce" and layout == "packed":
        msg = (
            "item-balanced CE requires a true [B, L] batch; packed layout has "
            "physical batch size 1 and cannot preserve per-item reduction"
        )
        raise ValueError(msg)

    if layout == "packed":
        if seq_len < k_neighbors:
            msg = (
                "packed MPNN benchmark requires each segment length to be at "
                f"least K ({seq_len=} < {k_neighbors=})"
            )
            raise ValueError(msg)
        physical_batch_size = 1
        total_length = batch_size * seq_len
    else:
        physical_batch_size = batch_size
        total_length = seq_len

    coordinates = torch.randn(
        physical_batch_size,
        total_length,
        4,
        3,
        device=device,
        requires_grad=training and coordinate_grad,
    )
    sequence = torch.randint(
        0,
        21,
        (physical_batch_size, total_length),
        device=device,
    )
    mask = torch.ones(physical_batch_size, total_length, device=device)
    local_patch_index = torch.arange(
        (seq_len + patch_size - 1) // patch_size,
        device=device,
    ).repeat_interleave(patch_size)[:seq_len]

    if layout == "packed":
        residue_index = torch.arange(seq_len, device=device).repeat(batch_size)
        residue_index = residue_index.unsqueeze(0)
        chain_index = torch.zeros(
            1,
            total_length,
            dtype=torch.long,
            device=device,
        )
        decoding_order = torch.cat(
            [
                torch.randperm(seq_len, device=device) + segment * seq_len
                for segment in range(batch_size)
            ]
        ).unsqueeze(0)
        patch_index = torch.cat(
            [local_patch_index + segment * seq_len for segment in range(batch_size)]
        ).unsqueeze(0)
        segment_lengths = torch.full(
            (batch_size,),
            seq_len,
            dtype=torch.long,
            device=device,
        )
    else:
        residue_index = torch.arange(seq_len, device=device).expand(
            physical_batch_size,
            -1,
        )
        chain_index = torch.zeros(
            physical_batch_size,
            seq_len,
            dtype=torch.long,
            device=device,
        )
        decoding_order = torch.stack(
            [torch.randperm(seq_len, device=device) for _ in range(physical_batch_size)]
        )
        patch_index = local_patch_index.expand(physical_batch_size, -1)
        segment_lengths = None

    upstream_gradient = None
    if objective == "output_grad":
        gradient_dtype = torch.bfloat16 if use_amp else torch.float32
        upstream_gradient = torch.randn(
            physical_batch_size,
            21,
            total_length,
            device=device,
            dtype=gradient_dtype,
        )

    return MPNNWorkload(
        coordinates=coordinates,
        sequence=sequence,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        decoding_order=decoding_order,
        patch_index=patch_index,
        segment_lengths=segment_lengths,
        upstream_gradient=upstream_gradient,
        objective=objective,
        label_smoothing=label_smoothing,
        logical_batch_size=batch_size,
        physical_batch_size=physical_batch_size,
        total_length=total_length,
    )


__all__ = [
    "MPNNLayout",
    "MPNNTrainingObjective",
    "MPNNWorkload",
    "build_mpnn_workload",
]
