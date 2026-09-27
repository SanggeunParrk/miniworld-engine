"""True-batch data structures and collation utilities for ProteinMPNN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Iterator, Sequence

import torch
from torch.utils.data import Sampler


def _require_long_vector(name: str, value: torch.Tensor, length: int) -> None:
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {tuple(value.shape)}")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {value.dtype}")


def _require_mask(name: str, value: torch.Tensor, length: int) -> None:
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {tuple(value.shape)}")
    if value.dtype != torch.bool and not value.is_floating_point():
        raise TypeError(f"{name} must be boolean or floating point, got {value.dtype}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MPNNTrainingSample:
    """One unpadded protein or complex used as a single training item.

    ``decoding_order`` is a permutation of residue indices. ``patch_index`` is
    indexed by decoding step (not residue index), and must therefore have the
    same length as ``decoding_order``.  Masks may be omitted to mark every
    residue as present and supervised.
    """

    backbone: torch.Tensor
    sequence: torch.Tensor
    residue_index: torch.Tensor
    chain_index: torch.Tensor
    decoding_order: torch.Tensor
    patch_index: torch.Tensor
    residue_mask: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    fixed_decoding_order_length: int = 0

    def __post_init__(self) -> None:
        if self.backbone.ndim != 3 or self.backbone.shape[1:] != (4, 3):
            raise ValueError(
                "backbone must have shape [length, 4, 3], got "
                f"{tuple(self.backbone.shape)}"
            )
        if not self.backbone.is_floating_point():
            raise TypeError("backbone must be floating point")

        length = self.backbone.shape[0]
        if length == 0:
            raise ValueError("an MPNN training sample cannot be empty")
        for name in (
            "sequence",
            "residue_index",
            "chain_index",
            "decoding_order",
            "patch_index",
        ):
            _require_long_vector(name, getattr(self, name), length)

        devices = {
            value.device
            for value in (
                self.backbone,
                self.sequence,
                self.residue_index,
                self.chain_index,
                self.decoding_order,
                self.patch_index,
                self.residue_mask,
                self.loss_mask,
            )
            if value is not None
        }
        if len(devices) != 1:
            raise ValueError(
                "all tensors in an MPNN training sample must share a device"
            )

        expected_order = torch.arange(length, device=self.decoding_order.device)
        if not torch.equal(self.decoding_order.sort().values, expected_order):
            raise ValueError("decoding_order must be a permutation of [0, length)")
        if bool((self.patch_index < 0).any()):
            raise ValueError("patch_index must be non-negative")
        if length > 1 and bool((self.patch_index[1:] < self.patch_index[:-1]).any()):
            raise ValueError("patch_index must be non-decreasing by decoding step")

        if self.residue_mask is not None:
            _require_mask("residue_mask", self.residue_mask, length)
            if bool(((self.residue_mask != 0) & (self.residue_mask != 1)).any()):
                raise ValueError("residue_mask must be binary")
        if self.loss_mask is not None:
            _require_mask("loss_mask", self.loss_mask, length)
        if not isinstance(self.fixed_decoding_order_length, Integral):
            raise TypeError("fixed_decoding_order_length must be an integer")
        if not 0 <= self.fixed_decoding_order_length <= length:
            raise ValueError(
                "fixed_decoding_order_length must satisfy "
                "0 <= fixed_decoding_order_length <= length"
            )

    @property
    def length(self) -> int:
        """Number of unpadded residues in this item."""
        return self.backbone.shape[0]


@dataclass(frozen=True)
class MPNNTrainingBatch:
    """A padded true-``B`` MPNN batch with one independent graph per row."""

    backbone: torch.Tensor
    sequence: torch.Tensor
    residue_mask: torch.Tensor
    residue_index: torch.Tensor
    chain_index: torch.Tensor
    decoding_order: torch.Tensor
    patch_index: torch.Tensor
    loss_mask: torch.Tensor
    lengths: torch.Tensor
    fixed_decoding_order_length: torch.Tensor

    def __post_init__(self) -> None:
        if self.backbone.ndim != 4 or self.backbone.shape[2:] != (4, 3):
            raise ValueError(
                "backbone must have shape [batch, length, 4, 3], got "
                f"{tuple(self.backbone.shape)}"
            )
        batch, length = self.backbone.shape[:2]
        if batch == 0 or length == 0:
            raise ValueError("an MPNN training batch cannot be empty")
        expected_shape = (batch, length)
        for name in (
            "sequence",
            "residue_mask",
            "residue_index",
            "chain_index",
            "decoding_order",
            "patch_index",
            "loss_mask",
        ):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
                )
        if self.lengths.shape != (batch,):
            raise ValueError(
                f"lengths must have shape [{batch}], got {tuple(self.lengths.shape)}"
            )
        if self.fixed_decoding_order_length.shape != (batch,):
            raise ValueError(
                "fixed_decoding_order_length must have shape "
                f"[{batch}], got {tuple(self.fixed_decoding_order_length.shape)}"
            )
        for name in (
            "sequence",
            "residue_index",
            "chain_index",
            "decoding_order",
            "patch_index",
            "lengths",
            "fixed_decoding_order_length",
        ):
            if getattr(self, name).dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")

    @property
    def batch_size(self) -> int:
        return self.backbone.shape[0]

    @property
    def padded_length(self) -> int:
        return self.backbone.shape[1]

    @property
    def supervision_mask(self) -> torch.Tensor:
        """Loss weights after removing absent or padded residues."""
        return self.residue_mask * self.loss_mask

    def model_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return positional arguments accepted by :class:`ProteinMPNN`."""
        return (
            self.backbone,
            self.sequence,
            self.residue_mask,
            self.residue_index,
            self.chain_index,
            self.decoding_order,
            self.patch_index,
        )

    def model_keyword_arguments(self) -> dict[str, torch.Tensor]:
        """Return per-item metadata passed to the model by keyword."""
        return {
            "fixed_decoding_order_length": self.fixed_decoding_order_length,
        }

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> MPNNTrainingBatch:
        """Move every tensor without changing its semantic dtype."""
        values = {
            name: getattr(self, name).to(device=device, non_blocking=non_blocking)
            for name in self.__dataclass_fields__
        }
        return type(self)(**values)


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Visit every item once while grouping nearby lengths into fixed-size batches.

    Bucket membership changes only execution padding, never item multiplicity.
    Batch order and within-bucket order are deterministically shuffled by
    ``seed + epoch``. Pair this with ``collate_mpnn_samples(...,
    pad_to_multiple=bucket_width)`` to expose a small set of CUDA-Graph shapes.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        bucket_width: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if not lengths:
            raise ValueError("lengths cannot be empty")
        if not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(bucket_width, Integral) or bucket_width <= 0:
            raise ValueError("bucket_width must be a positive integer")
        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        normalized_lengths = []
        for index, length in enumerate(lengths):
            if not isinstance(length, Integral) or length <= 0:
                raise ValueError(f"lengths[{index}] must be a positive integer")
            normalized_lengths.append(int(length))

        self.lengths = tuple(normalized_lengths)
        self.batch_size = int(batch_size)
        self.bucket_width = int(bucket_width)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic shuffle for the next epoch."""
        if not isinstance(epoch, Integral):
            raise TypeError("epoch must be an integer")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        bucket_counts: dict[int, int] = {}
        for length in self.lengths:
            bucket = (length - 1) // self.bucket_width
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if self.drop_last:
            return sum(count // self.batch_size for count in bucket_counts.values())
        return sum(
            math.ceil(count / self.batch_size) for count in bucket_counts.values()
        )

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        buckets: dict[int, list[int]] = {}
        for index, length in enumerate(self.lengths):
            bucket = (length - 1) // self.bucket_width
            buckets.setdefault(bucket, []).append(index)

        batches = []
        for bucket in sorted(buckets):
            indices = buckets[bucket]
            if self.shuffle and len(indices) > 1:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[position] for position in permutation]
            bucket_batches = [
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            ]
            if (
                self.drop_last
                and bucket_batches
                and len(bucket_batches[-1]) < self.batch_size
            ):
                bucket_batches.pop()
            batches.extend(bucket_batches)
        if self.shuffle and len(batches) > 1:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in batch_order]
        yield from batches


def _normalize_length_buckets(length_buckets: Sequence[int]) -> tuple[int, ...]:
    if not length_buckets:
        raise ValueError("length_buckets cannot be empty")
    normalized = []
    for position, bucket in enumerate(length_buckets):
        if not isinstance(bucket, Integral) or bucket <= 0:
            raise ValueError(f"length_buckets[{position}] must be a positive integer")
        normalized.append(int(bucket))
    if sorted(set(normalized)) != normalized:
        raise ValueError("length_buckets must be strictly increasing and unique")
    return tuple(normalized)


def bucketed_padded_length(length: int, length_buckets: Sequence[int]) -> int:
    """Return the smallest bucket that holds ``length``.

    Lengths above the largest bucket keep their own exact length, which adds one
    shape per such length. Size the top bucket for the crop limit to avoid that.
    """
    if not isinstance(length, Integral) or length <= 0:
        raise ValueError("length must be a positive integer")
    for bucket in _normalize_length_buckets(length_buckets):
        if length <= bucket:
            return bucket
    return int(length)


def make_bucketed_collate_fn(
    length_buckets: Sequence[int],
) -> Callable[[Sequence[MPNNTrainingSample]], MPNNTrainingBatch]:
    """Return a ``collate_fn`` that pads each batch to its length bucket.

    Pair this with :class:`TokenBudgetBatchSampler` built on the same buckets so
    the padded width the sampler budgeted for is the width the batch actually
    gets, and the DataLoader emits only the sampler's planned shapes.
    """
    buckets = _normalize_length_buckets(length_buckets)

    def collate(samples: Sequence[MPNNTrainingSample]) -> MPNNTrainingBatch:
        longest = max(sample.length for sample in samples)
        return collate_mpnn_samples(
            samples,
            pad_to_length=bucketed_padded_length(longest, buckets),
        )

    return collate


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Fill a padded-token budget instead of a fixed item count.

    Memory and compute for this model are linear in *padded* tokens -- the graph
    is fixed-K, so a padded residue costs a full K-edge row whose result is only
    masked away afterwards. A fixed item count therefore spends a fixed fraction
    of the budget on padding whenever the batch is shorter than the crop limit.
    This sampler instead groups items into length buckets and emits
    ``batch_size = token_budget // bucket`` items per batch, so every batch
    carries about the same number of padded tokens regardless of length.

    Shapes stay static, which is what makes the trade free: because the per-bucket
    item count of a dataset is fixed, the set of emitted ``(batch, padded_length)``
    pairs is identical in every epoch and is reported by :meth:`shape_plan`.
    Shuffling changes which items share a batch, never the shapes. With
    ``drop_last=True`` there is exactly one shape per occupied bucket.

    Unlike the upstream ProteinMPNN loader this never discards an item: upstream
    flushes a batch when the budget is exceeded and then starts the next batch
    *without* the item that triggered the flush, silently dropping one protein per
    batch boundary per epoch.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        token_budget: int,
        *,
        length_buckets: Sequence[int],
        shuffle: bool = True,
        drop_last: bool = False,
        max_batch_size: int | None = None,
        seed: int = 0,
    ) -> None:
        if not lengths:
            raise ValueError("lengths cannot be empty")
        if not isinstance(token_budget, Integral) or token_budget <= 0:
            raise ValueError("token_budget must be a positive integer")
        if max_batch_size is not None and (
            not isinstance(max_batch_size, Integral) or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        normalized_lengths = []
        for index, length in enumerate(lengths):
            if not isinstance(length, Integral) or length <= 0:
                raise ValueError(f"lengths[{index}] must be a positive integer")
            normalized_lengths.append(int(length))

        self.lengths = tuple(normalized_lengths)
        self.token_budget = int(token_budget)
        self.length_buckets = _normalize_length_buckets(length_buckets)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_batch_size = None if max_batch_size is None else int(max_batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic shuffle for the next epoch."""
        if not isinstance(epoch, Integral):
            raise TypeError("epoch must be an integer")
        self.epoch = int(epoch)

    def _capacity(self, padded_length: int) -> int:
        """Items that fit the budget at this padded width, at least one.

        A single item wider than the whole budget still has to be trained on, so
        it forms a batch of one and exceeds the budget. Choose the budget for the
        crop limit if that must never happen.
        """
        capacity = max(1, self.token_budget // padded_length)
        if self.max_batch_size is not None:
            capacity = min(capacity, self.max_batch_size)
        return capacity

    def _grouped_indices(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for index, length in enumerate(self.lengths):
            padded = bucketed_padded_length(length, self.length_buckets)
            groups.setdefault(padded, []).append(index)
        return groups

    def _batch_sizes(self, padded_length: int, count: int) -> list[int]:
        capacity = self._capacity(padded_length)
        sizes = [capacity] * (count // capacity)
        remainder = count % capacity
        if remainder and not self.drop_last:
            sizes.append(remainder)
        return sizes

    def shape_plan(self) -> list[tuple[int, int]]:
        """Return every ``(batch, padded_length)`` this sampler can emit.

        Use it to pre-warm ``torch.compile`` and CUDA Graph capture, and to see
        how many distinct graphs a bucket choice implies.
        """
        shapes = {
            (size, padded_length)
            for padded_length, indices in self._grouped_indices().items()
            for size in self._batch_sizes(padded_length, len(indices))
        }
        return sorted(shapes)

    def budget_occupancy(self) -> float:
        """Mean fraction of the token budget each emitted batch actually uses.

        This is the quantity a fixed item count gives up. ``batch_size`` has to be
        chosen so that ``batch_size * crop_limit`` fits in memory, so a batch of
        short items occupies only ``batch_size * their_length`` and leaves the rest
        of the device idle. Filling the budget instead turns that headroom into
        items per step.
        """
        batches = 0
        occupied = 0
        for padded_length, indices in self._grouped_indices().items():
            for size in self._batch_sizes(padded_length, len(indices)):
                batches += 1
                occupied += size * padded_length
        if batches == 0:
            return 0.0
        return occupied / (batches * self.token_budget)

    def token_utilization(self) -> float:
        """Fraction of budgeted padded tokens that carry real residues."""
        real = 0
        padded = 0
        for padded_length, indices in self._grouped_indices().items():
            sizes = self._batch_sizes(padded_length, len(indices))
            kept = sum(sizes)
            padded += kept * padded_length
            lengths = sorted((self.lengths[index] for index in indices), reverse=True)
            real += sum(lengths[:kept])
        if padded == 0:
            return 0.0
        return real / padded

    def __len__(self) -> int:
        return sum(
            len(self._batch_sizes(padded_length, len(indices)))
            for padded_length, indices in self._grouped_indices().items()
        )

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for padded_length in sorted(self._grouped_indices()):
            indices = self._grouped_indices()[padded_length]
            if self.shuffle and len(indices) > 1:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[position] for position in permutation]
            start = 0
            for size in self._batch_sizes(padded_length, len(indices)):
                batches.append(indices[start : start + size])
                start += size
        if self.shuffle and len(batches) > 1:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in batch_order]
        yield from batches


def collate_mpnn_samples(
    samples: Sequence[MPNNTrainingSample],
    *,
    pad_to_length: int | None = None,
    pad_to_multiple: int | None = None,
) -> MPNNTrainingBatch:
    """Pad independent samples into a true batch without joining their graphs.

    Valid decoding steps are preserved exactly. Padding residues are appended to
    each decoding permutation and assigned later, distinct patch ids, so every
    row remains a complete permutation over the padded tensor width.
    """
    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    if any(not isinstance(sample, MPNNTrainingSample) for sample in samples):
        raise TypeError("collate_mpnn_samples expects MPNNTrainingSample objects")

    if pad_to_length is not None and pad_to_multiple is not None:
        raise ValueError("pad_to_length and pad_to_multiple are mutually exclusive")
    maximum_length = max(sample.length for sample in samples)
    if pad_to_multiple is not None:
        if not isinstance(pad_to_multiple, Integral) or pad_to_multiple <= 0:
            raise ValueError("pad_to_multiple must be a positive integer")
        multiple = int(pad_to_multiple)
        padded_length = math.ceil(maximum_length / multiple) * multiple
    elif pad_to_length is None:
        padded_length = maximum_length
    else:
        if not isinstance(pad_to_length, Integral):
            raise TypeError("pad_to_length must be an integer")
        padded_length = int(pad_to_length)
        if padded_length < maximum_length:
            raise ValueError(
                f"pad_to_length={padded_length} is smaller than the longest "
                f"sample ({maximum_length})"
            )

    first = samples[0]
    tensor_specs = {
        "backbone": (first.backbone.dtype, first.backbone.device),
        "sequence": (first.sequence.dtype, first.sequence.device),
        "residue_index": (first.residue_index.dtype, first.residue_index.device),
        "chain_index": (first.chain_index.dtype, first.chain_index.device),
        "decoding_order": (
            first.decoding_order.dtype,
            first.decoding_order.device,
        ),
        "patch_index": (first.patch_index.dtype, first.patch_index.device),
    }
    for sample in samples[1:]:
        for name, expected in tensor_specs.items():
            value = getattr(sample, name)
            if (value.dtype, value.device) != expected:
                raise ValueError(
                    f"all samples must share {name} dtype and device; expected "
                    f"{expected}, got {(value.dtype, value.device)}"
                )

    batch_size = len(samples)
    backbone = first.backbone.new_zeros((batch_size, padded_length, 4, 3))
    sequence = first.sequence.new_zeros((batch_size, padded_length))
    residue_index = first.residue_index.new_zeros((batch_size, padded_length))
    chain_index = first.chain_index.new_zeros((batch_size, padded_length))
    decoding_order = first.decoding_order.new_empty((batch_size, padded_length))
    patch_index = first.patch_index.new_empty((batch_size, padded_length))

    # Floating masks are required by the feature path's ``1.0 - mask`` math.
    mask_dtype = first.backbone.dtype
    residue_mask = torch.zeros(
        (batch_size, padded_length), dtype=mask_dtype, device=first.backbone.device
    )
    loss_mask = torch.zeros_like(residue_mask)
    lengths = torch.empty(batch_size, dtype=torch.long, device=first.backbone.device)
    fixed_decoding_order_length = torch.empty_like(lengths)

    for row, sample in enumerate(samples):
        length = sample.length
        lengths[row] = length
        fixed_decoding_order_length[row] = sample.fixed_decoding_order_length
        backbone[row, :length].copy_(sample.backbone)
        sequence[row, :length].copy_(sample.sequence)
        residue_index[row, :length].copy_(sample.residue_index)
        chain_index[row, :length].copy_(sample.chain_index)
        decoding_order[row, :length].copy_(sample.decoding_order)
        patch_index[row, :length].copy_(sample.patch_index)

        if sample.residue_mask is None:
            residue_mask[row, :length] = 1
        else:
            residue_mask[row, :length].copy_(sample.residue_mask)
        if sample.loss_mask is None:
            loss_mask[row, :length] = 1
        else:
            loss_mask[row, :length].copy_(sample.loss_mask)
        if not bool((residue_mask[row, :length] * loss_mask[row, :length]).sum() > 0):
            raise ValueError(f"sample {row} has no supervised residues")

        padding = padded_length - length
        if padding:
            decoding_order[row, length:] = torch.arange(
                length,
                padded_length,
                device=decoding_order.device,
                dtype=decoding_order.dtype,
            )
            first_padding_patch = sample.patch_index[-1] + 1
            patch_index[row, length:] = first_padding_patch + torch.arange(
                padding,
                device=patch_index.device,
                dtype=patch_index.dtype,
            )

    return MPNNTrainingBatch(
        backbone=backbone,
        sequence=sequence,
        residue_mask=residue_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        decoding_order=decoding_order,
        patch_index=patch_index,
        loss_mask=loss_mask,
        lengths=lengths,
        fixed_decoding_order_length=fixed_decoding_order_length,
    )


__all__ = [
    "LengthBucketBatchSampler",
    "MPNNTrainingBatch",
    "MPNNTrainingSample",
    "TokenBudgetBatchSampler",
    "bucketed_padded_length",
    "collate_mpnn_samples",
    "make_bucketed_collate_fn",
]
