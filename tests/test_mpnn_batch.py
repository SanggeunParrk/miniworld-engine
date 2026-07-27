"""True-batch collation and item-balanced MPNN loss tests."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from miniworld_kernels.modules.mpnn import (
    ItemBalancedLossStatistics,
    LengthBucketBatchSampler,
    TokenBudgetBatchSampler,
    bucketed_padded_length,
    make_bucketed_collate_fn,
    MPNNTrainingSample,
    ProteinMPNN,
    ProteinMPNNConfig,
    collate_mpnn_samples,
    item_balanced_cross_entropy,
)


def _sample(
    length: int,
    *,
    seed: int,
    decoding_order: torch.Tensor | None = None,
    patch_index: torch.Tensor | None = None,
    residue_mask: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    fixed_decoding_order_length: int = 0,
) -> MPNNTrainingSample:
    generator = torch.Generator().manual_seed(seed)
    if decoding_order is None:
        decoding_order = torch.randperm(length, generator=generator)
    if patch_index is None:
        patch_index = torch.arange(length) // 2
    return MPNNTrainingSample(
        backbone=torch.randn(length, 4, 3, generator=generator),
        sequence=torch.randint(0, 21, (length,), generator=generator),
        residue_index=torch.arange(length),
        chain_index=(torch.arange(length) >= max(1, length // 2)).long(),
        decoding_order=decoding_order,
        patch_index=patch_index,
        residue_mask=residue_mask,
        loss_mask=loss_mask,
        fixed_decoding_order_length=fixed_decoding_order_length,
    )


def test_collate_variable_lengths_builds_full_padded_permutations() -> None:
    first = _sample(
        3,
        seed=1,
        decoding_order=torch.tensor([2, 0, 1]),
        patch_index=torch.tensor([0, 0, 1]),
        residue_mask=torch.tensor([1.0, 0.0, 1.0]),
        loss_mask=torch.tensor([0.0, 1.0, 0.5]),
        fixed_decoding_order_length=1,
    )
    second = _sample(
        5,
        seed=2,
        decoding_order=torch.tensor([1, 4, 0, 3, 2]),
        patch_index=torch.tensor([0, 0, 1, 1, 2]),
    )

    batch = collate_mpnn_samples([first, second])

    assert batch.backbone.shape == (2, 5, 4, 3)
    assert batch.sequence.shape == (2, 5)
    assert batch.lengths.tolist() == [3, 5]
    assert batch.batch_size == 2
    assert batch.padded_length == 5
    torch.testing.assert_close(batch.decoding_order[0], torch.tensor([2, 0, 1, 3, 4]))
    torch.testing.assert_close(batch.patch_index[0], torch.tensor([0, 0, 1, 2, 3]))
    torch.testing.assert_close(batch.decoding_order[1], second.decoding_order)
    torch.testing.assert_close(batch.patch_index[1], second.patch_index)
    for row in batch.decoding_order:
        torch.testing.assert_close(row.sort().values, torch.arange(5))

    torch.testing.assert_close(
        batch.residue_mask[0], torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        batch.loss_mask[0], torch.tensor([0.0, 1.0, 0.5, 0.0, 0.0])
    )
    torch.testing.assert_close(
        batch.supervision_mask[0], torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0])
    )
    torch.testing.assert_close(batch.fixed_decoding_order_length, torch.tensor([1, 0]))
    torch.testing.assert_close(batch.backbone[0, :3], first.backbone)
    assert torch.count_nonzero(batch.backbone[0, 3:]) == 0
    assert torch.count_nonzero(batch.sequence[0, 3:]) == 0
    assert torch.isfinite(batch.backbone).all()
    assert batch.model_inputs() == (
        batch.backbone,
        batch.sequence,
        batch.residue_mask,
        batch.residue_index,
        batch.chain_index,
        batch.decoding_order,
        batch.patch_index,
    )
    assert batch.model_keyword_arguments() == {
        "fixed_decoding_order_length": batch.fixed_decoding_order_length
    }


def test_collate_can_target_a_fixed_bucket_width() -> None:
    batch = collate_mpnn_samples(
        [_sample(3, seed=3), _sample(4, seed=4)], pad_to_length=8
    )
    assert batch.backbone.shape[:2] == (2, 8)
    assert (
        batch.decoding_order[0, :3].tolist()
        == _sample(3, seed=3).decoding_order.tolist()
    )
    assert batch.decoding_order[0, 3:].tolist() == [3, 4, 5, 6, 7]
    assert batch.patch_index[0].tolist() == [0, 0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="smaller than the longest"):
        collate_mpnn_samples([_sample(5, seed=5)], pad_to_length=4)

    rounded = collate_mpnn_samples(
        [_sample(3, seed=3), _sample(5, seed=5)], pad_to_multiple=4
    )
    assert rounded.padded_length == 8
    with pytest.raises(ValueError, match="mutually exclusive"):
        collate_mpnn_samples([_sample(3, seed=3)], pad_to_length=4, pad_to_multiple=4)


def test_length_bucket_sampler_visits_every_item_once_per_epoch() -> None:
    lengths = [2, 7, 3, 9, 16, 5, 12, 15, 1, 8, 11]
    sampler = LengthBucketBatchSampler(
        lengths,
        batch_size=3,
        bucket_width=4,
        shuffle=True,
        seed=41,
    )

    first_epoch = list(sampler)
    assert len(first_epoch) == len(sampler) == 4
    assert sorted(index for batch in first_epoch for index in batch) == list(
        range(len(lengths))
    )
    assert sorted(len(batch) for batch in first_epoch) == [2, 3, 3, 3]
    for batch in first_epoch:
        assert len({(lengths[index] - 1) // 4 for index in batch}) == 1

    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert second_epoch != first_epoch
    replica = LengthBucketBatchSampler(
        lengths,
        batch_size=3,
        bucket_width=4,
        shuffle=True,
        seed=41,
    )
    replica.set_epoch(1)
    assert list(replica) == second_epoch

    dropped = LengthBucketBatchSampler(
        lengths,
        batch_size=3,
        bucket_width=4,
        shuffle=False,
        drop_last=True,
    )
    dropped_batches = list(dropped)
    assert len(dropped_batches) == len(dropped) == 3
    assert all(len(batch) == 3 for batch in dropped_batches)


def test_token_budget_sampler_keeps_every_item_and_a_fixed_shape_set() -> None:
    # 3 items land in bucket 8, 4 in bucket 16, 2 in bucket 32.
    lengths = [5, 8, 3, 9, 16, 12, 30, 25, 14]
    buckets = (8, 16, 32)
    sampler = TokenBudgetBatchSampler(
        lengths,
        token_budget=32,
        length_buckets=buckets,
        shuffle=True,
        seed=41,
    )

    # capacity: 32//8 = 4, 32//16 = 2, 32//32 = 1
    first_epoch = list(sampler)
    assert len(first_epoch) == len(sampler)
    assert sorted(index for batch in first_epoch for index in batch) == list(
        range(len(lengths))
    )
    for batch in first_epoch:
        padded = bucketed_padded_length(max(lengths[index] for index in batch), buckets)
        # Every batch stays inside the padded-token budget.
        assert len(batch) * padded <= 32
        # A batch never mixes buckets, or its padded width would rise for all.
        assert {bucketed_padded_length(lengths[index], buckets) for index in batch} == {
            padded
        }

    # The emitted shapes are a property of the dataset, not of the shuffle.
    plan = sampler.shape_plan()
    assert plan == sorted(
        {
            (
                len(batch),
                bucketed_padded_length(max(lengths[index] for index in batch), buckets),
            )
            for batch in first_epoch
        }
    )
    sampler.set_epoch(7)
    seventh_epoch = list(sampler)
    assert seventh_epoch != first_epoch
    assert sampler.shape_plan() == plan
    assert sorted(index for batch in seventh_epoch for index in batch) == list(
        range(len(lengths))
    )

    replica = TokenBudgetBatchSampler(
        lengths,
        token_budget=32,
        length_buckets=buckets,
        shuffle=True,
        seed=41,
    )
    replica.set_epoch(7)
    assert list(replica) == seventh_epoch

    # drop_last leaves exactly one shape per occupied bucket.
    dropped = TokenBudgetBatchSampler(
        lengths,
        token_budget=32,
        length_buckets=buckets,
        shuffle=False,
        drop_last=True,
    )
    assert len({padded for _size, padded in dropped.shape_plan()}) == len(
        dropped.shape_plan()
    )


def test_token_budget_sampler_fills_the_device_a_fixed_count_leaves_idle() -> None:
    """What a fixed item count gives up is budget occupancy, not batch padding.

    ``batch_size`` must be chosen so that ``batch_size * crop_limit`` fits, so a
    batch of short items uses a small fraction of the device and the epoch needs
    many more steps for the same coverage.
    """
    # 1024 short items fill one budgeted batch; 64 long ones fill two.
    lengths = [64] * 1024 + [2048] * 64
    buckets = (64, 256, 1024, 2048)
    budget = 32 * 2048

    fixed = LengthBucketBatchSampler(
        lengths, batch_size=32, bucket_width=64, shuffle=False
    )
    budgeted = TokenBudgetBatchSampler(
        lengths,
        token_budget=budget,
        length_buckets=buckets,
        shuffle=False,
    )

    fixed_batches = list(fixed)
    budgeted_batches = list(budgeted)
    assert sorted(index for batch in fixed_batches for index in batch) == sorted(
        index for batch in budgeted_batches for index in batch
    )

    # Same coverage, far fewer steps: 1024 short items go 32-at-a-time versus
    # 1024-at-a-time inside the same memory budget.
    assert len(budgeted_batches) < len(fixed_batches) / 10

    def occupancy(batches: list[list[int]]) -> float:
        return sum(
            len(batch)
            * bucketed_padded_length(max(lengths[index] for index in batch), buckets)
            for batch in batches
        ) / (len(batches) * budget)

    assert occupancy(fixed_batches) < 0.1
    # Every budgeted batch is exactly full for this length distribution.
    assert occupancy(budgeted_batches) == pytest.approx(1.0)
    assert budgeted.budget_occupancy() == pytest.approx(occupancy(budgeted_batches))

    # Padding inside a batch stays negligible, and no batch exceeds the budget.
    assert budgeted.token_utilization() > 0.99
    for batch in budgeted_batches:
        padded = bucketed_padded_length(max(lengths[index] for index in batch), buckets)
        assert len(batch) * padded <= budget


def test_token_budget_sampler_never_drops_a_boundary_item() -> None:
    """Upstream ProteinMPNN loses one protein per batch boundary; we must not.

    ``StructureLoader`` flushes the batch when the token budget is exceeded and
    then starts the next batch without the item that triggered the flush.
    """
    lengths = [7, 7, 7, 7, 7, 7, 7]

    def upstream_batches(sequence: list[int], budget: int) -> list[list[int]]:
        order = sorted(range(len(sequence)), key=lambda index: sequence[index])
        clusters: list[list[int]] = []
        batch: list[int] = []
        for index in order:
            if sequence[index] * (len(batch) + 1) <= budget:
                batch.append(index)
            else:
                clusters.append(batch)
                batch = []
        if batch:
            clusters.append(batch)
        return clusters

    upstream = upstream_batches(lengths, 16)
    assert sorted(index for batch in upstream for index in batch) != list(
        range(len(lengths))
    )

    sampler = TokenBudgetBatchSampler(
        lengths,
        token_budget=16,
        length_buckets=(8,),
        shuffle=False,
    )
    assert sorted(index for batch in sampler for index in batch) == list(
        range(len(lengths))
    )


def test_bucketed_collate_pads_to_the_width_the_sampler_budgeted() -> None:
    buckets = (8, 16)
    collate = make_bucketed_collate_fn(buckets)
    batch = collate([_sample(3, seed=1), _sample(6, seed=2)])
    assert batch.padded_length == 8

    wider = collate([_sample(9, seed=3), _sample(5, seed=4)])
    assert wider.padded_length == 16

    # Above the top bucket the exact length is kept rather than silently cropped.
    assert bucketed_padded_length(40, buckets) == 40


def test_token_budget_sampler_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="token_budget"):
        TokenBudgetBatchSampler([4], token_budget=0, length_buckets=(8,))
    with pytest.raises(ValueError, match="strictly increasing"):
        TokenBudgetBatchSampler([4], token_budget=8, length_buckets=(16, 8))
    with pytest.raises(ValueError, match="length_buckets cannot be empty"):
        TokenBudgetBatchSampler([4], token_budget=8, length_buckets=())
    with pytest.raises(ValueError, match="lengths cannot be empty"):
        TokenBudgetBatchSampler([], token_budget=8, length_buckets=(8,))

    # An item wider than the whole budget still has to be visited.
    lonely = TokenBudgetBatchSampler(
        [64], token_budget=8, length_buckets=(64,), shuffle=False
    )
    assert list(lonely) == [[0]]

    capped = TokenBudgetBatchSampler(
        [4] * 8,
        token_budget=64,
        length_buckets=(8,),
        max_batch_size=3,
        shuffle=False,
    )
    assert [len(batch) for batch in capped] == [3, 3, 2]


def test_sample_rejects_invalid_decoding_metadata() -> None:
    with pytest.raises(ValueError, match="permutation"):
        _sample(3, seed=6, decoding_order=torch.tensor([0, 0, 2]))
    with pytest.raises(ValueError, match="non-decreasing"):
        _sample(3, seed=6, patch_index=torch.tensor([0, 2, 1]))
    with pytest.raises(ValueError, match="fixed_decoding_order_length"):
        _sample(3, seed=6, fixed_decoding_order_length=4)


def test_collate_rejects_an_item_without_supervised_residues() -> None:
    empty = _sample(3, seed=7, residue_mask=torch.zeros(3))
    with pytest.raises(ValueError, match="no supervised residues"):
        collate_mpnn_samples([empty])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_length_true_batch_matches_independent_graph_execution(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    samples = [
        _sample(2, seed=11, fixed_decoding_order_length=1),
        _sample(6, seed=12, fixed_decoding_order_length=2),
    ]
    batch = collate_mpnn_samples(samples).to(device)
    model = (
        ProteinMPNN(
            ProteinMPNNConfig(
                node_width=8,
                edge_width=8,
                hidden_width=8,
                encoder_depth=1,
                decoder_depth=1,
                k_neighbors=4,
                coordinate_noise=0,
                dropout=0,
                block_linear_min_edges=1_000_000,
            )
        )
        .eval()
        .to(device)
    )
    generator = torch.Generator().manual_seed(13)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
        actual = model(*batch.model_inputs(), **batch.model_keyword_arguments())
        for row, sample in enumerate(samples):
            single = collate_mpnn_samples([sample]).to(device)
            expected = model(*single.model_inputs(), **single.model_keyword_arguments())
            torch.testing.assert_close(
                actual[row, :, : sample.length],
                expected[0],
                atol=2e-5,
                rtol=2e-5,
            )

        perturbed_backbone = batch.backbone.clone()
        perturbed_sequence = batch.sequence.clone()
        perturbed_residue_index = batch.residue_index.clone()
        perturbed_chain_index = batch.chain_index.clone()
        short_length = samples[0].length
        perturbed_backbone[0, short_length:] = 10_000
        perturbed_sequence[0, short_length:] = 20
        perturbed_residue_index[0, short_length:] = 1_000_000
        perturbed_chain_index[0, short_length:] = 999
        perturbed = model(
            perturbed_backbone,
            perturbed_sequence,
            batch.residue_mask,
            perturbed_residue_index,
            perturbed_chain_index,
            batch.decoding_order,
            batch.patch_index,
            **batch.model_keyword_arguments(),
        )
        torch.testing.assert_close(
            perturbed[0, :, :short_length],
            actual[0, :, :short_length],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "block_linear_min_edges", [0, 1_000_000], ids=["block", "dense"]
)
def test_item_balanced_batch_gradients_match_mean_of_independent_graphs(
    device: str,
    block_linear_min_edges: int,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    samples = [_sample(2, seed=21), _sample(6, seed=22)]
    batch = collate_mpnn_samples(samples).to(device)
    config = ProteinMPNNConfig(
        node_width=8,
        edge_width=8,
        hidden_width=8,
        encoder_depth=1,
        decoder_depth=1,
        k_neighbors=4,
        coordinate_noise=0,
        dropout=0,
        block_linear_min_edges=block_linear_min_edges,
    )
    batched_model = ProteinMPNN(config).train().to(device)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for parameter in batched_model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    independent_model = copy.deepcopy(batched_model)

    batched_backbone = batch.backbone.detach().clone().requires_grad_(True)
    batched_logits = batched_model(
        batched_backbone,
        *batch.model_inputs()[1:],
        **batch.model_keyword_arguments(),
    )
    batched_loss = item_balanced_cross_entropy(
        batched_logits,
        batch.sequence,
        batch.loss_mask,
        residue_mask=batch.residue_mask,
        label_smoothing=0.1,
    ).loss
    batched_loss.backward()

    independent_losses = []
    independent_backbones = []
    for sample in samples:
        single = collate_mpnn_samples([sample]).to(device)
        backbone = single.backbone.detach().clone().requires_grad_(True)
        logits = independent_model(
            backbone,
            *single.model_inputs()[1:],
            **single.model_keyword_arguments(),
        )
        independent_losses.append(
            item_balanced_cross_entropy(
                logits,
                single.sequence,
                single.loss_mask,
                residue_mask=single.residue_mask,
                label_smoothing=0.1,
            ).loss
        )
        independent_backbones.append(backbone)
    torch.stack(independent_losses).mean().backward()

    torch.testing.assert_close(
        batched_loss,
        torch.stack([loss.detach() for loss in independent_losses]).mean(),
        atol=2e-6,
        rtol=2e-6,
    )
    for row, (sample, independent_backbone) in enumerate(
        zip(samples, independent_backbones, strict=True)
    ):
        torch.testing.assert_close(
            batched_backbone.grad[row, : sample.length],
            independent_backbone.grad[0],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            batched_backbone.grad[row, sample.length :],
            torch.zeros_like(batched_backbone.grad[row, sample.length :]),
            atol=0,
            rtol=0,
        )
    for batched_parameter, independent_parameter in zip(
        batched_model.parameters(), independent_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            batched_parameter.grad,
            independent_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )


def test_true_batch_rejects_packed_segment_metadata() -> None:
    batch = collate_mpnn_samples([_sample(3, seed=31), _sample(3, seed=32)])
    model = ProteinMPNN(
        ProteinMPNNConfig(
            node_width=8,
            edge_width=8,
            hidden_width=8,
            encoder_depth=1,
            decoder_depth=1,
            k_neighbors=2,
            coordinate_noise=0,
            dropout=0,
        )
    )
    with pytest.raises(ValueError, match="physical batch size 1"):
        model(*batch.model_inputs(), segment_lengths=torch.tensor([3, 3]))


def test_item_balanced_cross_entropy_averages_items_not_tokens() -> None:
    logits = torch.tensor(
        [
            [[-2.0, 9.0, 9.0], [2.0, -9.0, -9.0]],
            [[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]],
        ],
        requires_grad=True,
    )
    target = torch.tensor([[0, 0, 0], [0, 0, 0]])
    loss_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    result = item_balanced_cross_entropy(logits, target, loss_mask)
    token_loss = F.cross_entropy(logits, target, reduction="none")
    expected_per_item = torch.stack((token_loss[0, 0], token_loss[1].mean()))
    expected = expected_per_item.mean()

    torch.testing.assert_close(result.per_item_loss, expected_per_item)
    torch.testing.assert_close(result.loss, expected)
    assert result.active_item_count.item() == 2
    assert result.supervised_token_count.item() == 4
    assert result.supervision_weight.item() == 4
    assert not torch.allclose(result.loss, (token_loss * loss_mask).sum() / 4)

    result.loss.backward()
    assert logits.grad is not None


def test_loss_intersects_masks_and_excludes_items_without_targets() -> None:
    logits = torch.tensor(
        [
            [[1.0, 1.0], [-1.0, -1.0]],
            [[1.0, 1.0], [-1.0, -1.0]],
        ]
    )
    target = torch.tensor([[0, 0], [-100, 0]])
    loss_mask = torch.ones(2, 2)
    residue_mask = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    result = item_balanced_cross_entropy(
        logits,
        target,
        loss_mask,
        residue_mask=residue_mask,
    )

    torch.testing.assert_close(result.per_item_loss, torch.zeros(2))
    assert result.loss.item() == 0
    assert result.active_item_count.item() == 0
    assert result.supervised_token_count.item() == 0


def test_ddp_scaling_and_statistics_recover_global_item_mean() -> None:
    first = item_balanced_cross_entropy(
        torch.tensor([[[2.0], [-2.0]]]),
        torch.tensor([[0]]),
        torch.ones(1, 1),
    )
    second = item_balanced_cross_entropy(
        torch.tensor(
            [
                [[-2.0], [2.0]],
                [[0.0], [0.0]],
            ]
        ),
        torch.tensor([[0], [0]]),
        torch.ones(2, 1),
    )
    global_item_count = first.active_item_count + second.active_item_count

    rank_zero_loss = first.for_ddp_backward(global_item_count, world_size=2)
    rank_one_loss = second.for_ddp_backward(global_item_count, world_size=2)
    ddp_averaged = (rank_zero_loss + rank_one_loss) / 2
    expected = torch.cat((first.per_item_loss, second.per_item_loss)).mean()
    torch.testing.assert_close(ddp_averaged, expected)

    statistics = first.statistics() + second.statistics()
    assert isinstance(statistics, ItemBalancedLossStatistics)
    torch.testing.assert_close(statistics.mean_loss, expected)
    assert statistics.active_item_count.item() == 3
    assert statistics.supervised_token_count.item() == 3
    assert not statistics.item_loss_sum.requires_grad

    with pytest.raises(ValueError, match="world_size"):
        first.for_ddp_backward(global_item_count, world_size=0)
    with pytest.raises(ValueError, match="global_item_count"):
        first.for_ddp_backward(0, world_size=2)


def test_half_precision_loss_reduces_in_float32() -> None:
    logits = torch.randn(2, 3, 7, dtype=torch.bfloat16)
    target = torch.zeros(2, 7, dtype=torch.long)
    result = item_balanced_cross_entropy(logits, target, torch.ones(2, 7))
    assert result.loss.dtype == torch.float32
    assert result.per_item_supervision_weight.dtype == torch.float32
