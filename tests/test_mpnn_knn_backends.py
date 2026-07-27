"""Neighbour-search backends: exact chunking, and the cutoff-capped grid."""

from __future__ import annotations

import pytest
import torch

from miniworld_kernels.modules.mpnn import (
    BackboneFeatures,
    ProteinMPNN,
    ProteinMPNNConfig,
)


def _features(**overrides) -> BackboneFeatures:
    settings = dict(edge_width=16, num_rbf=4, k_neighbors=8, coordinate_noise=0.0)
    settings.update(overrides)
    torch.manual_seed(0)
    return BackboneFeatures(**settings)


def _coordinates(
    batch: int,
    length: int,
    *,
    seed: int,
    device: str = "cpu",
    spread: float = 6.0,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, length, 3, generator=generator) * spread).to(device)


def test_chunked_backend_is_independent_of_the_chunk_size() -> None:
    """The point of chunking is a smaller intermediate, not a different answer.

    The block size must therefore not be observable. That requires pinning
    ``cdist`` to its direct form: its default switches to a matmul expansion above
    a size threshold, so with the default a 5-row block and a 16-row block
    disagree.
    """
    length = 37
    coordinates = _coordinates(2, length, seed=1)
    residue_mask = torch.ones(2, length)
    residue_mask[1, length // 2 :] = 0.0

    baseline = None
    for chunk in (1, 5, 16, length, length + 9):
        candidate = _features(knn_backend="chunked", knn_query_chunk=chunk)
        actual = candidate._nearest_neighbors(coordinates, residue_mask, None)
        if baseline is None:
            baseline = actual
            continue
        for name, left, right in zip(
            ("distances", "indices", "edge_mask"), actual, baseline, strict=True
        ):
            torch.testing.assert_close(
                left, right, atol=0, rtol=0, msg=f"{name}@{chunk}"
            )


def test_chunked_backend_is_more_accurate_than_the_default_cdist() -> None:
    """``torch.cdist``'s default expansion loses precision on near pairs.

    It reports a nonzero distance from a residue to itself; the direct form used
    by the chunked backend does not. The two backends therefore select slightly
    different neighbours near a tie, which is why ``cdist`` stays the default:
    it is the one that reproduces the frozen reference bit for bit.
    """
    length = 37
    coordinates = _coordinates(1, length, seed=1)
    residue_mask = torch.ones(1, length)

    reference_distances = _features()._nearest_neighbors(
        coordinates, residue_mask, None
    )[0]
    chunked_distances = _features(
        knn_backend="chunked", knn_query_chunk=8
    )._nearest_neighbors(coordinates, residue_mask, None)[0]

    # Slot 0 is the residue itself for every row.
    assert float(chunked_distances[:, :, 0].abs().max()) == 0.0
    assert float(reference_distances[:, :, 0].abs().max()) > 0.0
    torch.testing.assert_close(
        chunked_distances, reference_distances, atol=5e-3, rtol=1e-3
    )


def test_chunked_backend_matches_across_segments_of_a_packed_row() -> None:
    lengths = torch.tensor([9, 13, 7])
    total = int(lengths.sum())
    coordinates = _coordinates(1, total, seed=2)
    residue_mask = torch.ones(1, total)

    whole = _features(knn_backend="chunked", knn_query_chunk=total)._nearest_neighbors(
        coordinates, residue_mask, lengths
    )
    chunked = _features(knn_backend="chunked", knn_query_chunk=4)._nearest_neighbors(
        coordinates, residue_mask, lengths
    )
    for left, right in zip(chunked, whole, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    # Segment isolation is preserved by the block mask.
    segment = torch.arange(lengths.numel()).repeat_interleave(lengths)
    neighbour_segment = segment[chunked[1]]
    crossing = (neighbour_segment != segment[None, :, None]) & (chunked[2] > 0)
    assert int(crossing.sum()) == 0


def test_chunked_backend_defers_to_the_single_shot_path_under_coordinate_grad() -> None:
    """Chunking cannot bound the peak once cdist retains its output for backward."""
    length = 21
    coordinates = _coordinates(1, length, seed=3).requires_grad_()
    residue_mask = torch.ones(1, length)
    candidate = _features(knn_backend="chunked", knn_query_chunk=4)
    distances, _indices, _mask = candidate._nearest_neighbors(
        coordinates, residue_mask, None
    )
    assert distances.requires_grad
    expected = _features()._nearest_neighbors(coordinates, residue_mask, None)[0]
    torch.testing.assert_close(distances, expected, atol=0, rtol=0)
    distances.sum().backward()
    assert coordinates.grad is not None


def _brute_force_capped(
    coordinates: torch.Tensor,
    residue_mask: torch.Tensor,
    segment_lengths: torch.Tensor | None,
    k: int,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rule stated directly: the k nearest that are strictly within cutoff."""
    length = coordinates.shape[1]
    pair = residue_mask[:, None] * residue_mask[:, :, None]
    if segment_lengths is not None:
        segment = torch.arange(segment_lengths.numel()).repeat_interleave(
            segment_lengths
        )
        pair = pair * (segment[:, None] == segment[None, :])
    distances = torch.cdist(coordinates, coordinates)
    inside = (pair > 0) & (distances <= cutoff)
    ranked = torch.where(inside, distances, torch.full_like(distances, float("inf")))
    values, indices = torch.topk(ranked, min(k, length), dim=-1, largest=False)
    return values, torch.isfinite(values), indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("spread", [4.0, 12.0])
def test_grid_cutoff_backend_is_exact_not_approximate(spread: float) -> None:
    """The 3x3x3 block provably contains everything within one cell width."""
    length = 300
    cutoff = 16.0
    coordinates = _coordinates(1, length, seed=4, device="cuda", spread=spread)
    residue_mask = torch.ones(1, length, device="cuda")
    grid = _features(k_neighbors=48, knn_backend="grid_cutoff", knn_cutoff=cutoff)
    distances, indices, edge_mask = grid._nearest_neighbors(
        coordinates, residue_mask, None
    )

    expected_values, expected_valid, expected_indices = _brute_force_capped(
        coordinates, residue_mask, None, 48, cutoff
    )
    torch.testing.assert_close(edge_mask > 0, expected_valid, atol=0, rtol=0)
    # Selected sets must agree exactly; compare as sorted sets of kept edges.
    kept_actual = torch.where(edge_mask > 0, indices, torch.full_like(indices, -1))
    kept_expected = torch.where(
        expected_valid, expected_indices, torch.full_like(expected_indices, -1)
    )
    torch.testing.assert_close(
        kept_actual.sort(dim=-1).values,
        kept_expected.sort(dim=-1).values,
        atol=0,
        rtol=0,
    )
    assert float((distances * (edge_mask > 0)).max()) <= cutoff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grid_cutoff_backend_never_crosses_a_segment() -> None:
    lengths = torch.tensor([120, 90, 140], device="cuda")
    total = int(lengths.sum())
    # Segments overlap in space, so only the key can keep them apart.
    coordinates = _coordinates(1, total, seed=5, device="cuda", spread=8.0)
    residue_mask = torch.ones(1, total, device="cuda")
    grid = _features(k_neighbors=48, knn_backend="grid_cutoff", knn_cutoff=16.0)
    _distances, indices, edge_mask = grid._nearest_neighbors(
        coordinates, residue_mask, lengths
    )

    segment = torch.arange(lengths.numel(), device="cuda").repeat_interleave(lengths)
    neighbour_segment = segment[indices]
    crossing = (neighbour_segment != segment[None, :, None]) & (edge_mask > 0)
    assert int(crossing.sum()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grid_cutoff_backend_masks_unused_slots_and_stays_in_bounds() -> None:
    """A residue with few neighbours must produce masked, in-bounds slots."""
    length = 64
    coordinates = _coordinates(1, length, seed=6, device="cuda", spread=200.0)
    residue_mask = torch.ones(1, length, device="cuda")
    residue_mask[0, length // 2 :] = 0.0
    grid = _features(k_neighbors=48, knn_backend="grid_cutoff", knn_cutoff=16.0)
    distances, indices, edge_mask = grid._nearest_neighbors(
        coordinates, residue_mask, None
    )

    assert int(indices.min()) >= 0
    assert int(indices.max()) < length
    # Far apart at this spread, so each valid residue keeps only itself.
    assert torch.equal(
        edge_mask[0, : length // 2, 0],
        torch.ones(length // 2, device="cuda"),
    )
    assert float(edge_mask[0, : length // 2, 1:].sum()) == 0.0
    assert float(distances[edge_mask > 0].max()) == 0.0
    # Invalid residues are excluded from the graph entirely.
    assert float(edge_mask[0, length // 2 :].sum()) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grid_cutoff_ordering_is_defined_by_distance_then_index() -> None:
    """Duplicate coordinates make every tie explicit; the lower index must win."""
    base = _coordinates(1, 40, seed=7, device="cuda", spread=3.0)
    coordinates = torch.cat((base, base), dim=1)
    residue_mask = torch.ones(1, coordinates.shape[1], device="cuda")
    grid = _features(k_neighbors=48, knn_backend="grid_cutoff", knn_cutoff=16.0)
    distances, indices, edge_mask = grid._nearest_neighbors(
        coordinates, residue_mask, None
    )

    kept = edge_mask > 0
    # Within a row the kept distances are non-decreasing, and equal distances are
    # ordered by index.
    for row in range(coordinates.shape[1]):
        selected = indices[0, row][kept[0, row]]
        selected_distances = distances[0, row][kept[0, row]]
        for position in range(1, selected.numel()):
            previous = float(selected_distances[position - 1])
            current = float(selected_distances[position])
            assert previous <= current
            if previous == current:
                assert int(selected[position - 1]) < int(selected[position])


def test_model_rejects_unknown_knn_settings() -> None:
    with pytest.raises(ValueError, match="knn_backend"):
        ProteinMPNNConfig(knn_backend="grid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="knn_query_chunk"):
        ProteinMPNNConfig(knn_query_chunk=0)
    with pytest.raises(ValueError, match="knn_cutoff"):
        ProteinMPNNConfig(knn_cutoff=0.0)


def test_chunked_backend_leaves_the_model_output_untouched() -> None:
    common = dict(
        node_width=16,
        edge_width=16,
        hidden_width=16,
        encoder_depth=1,
        decoder_depth=1,
        k_neighbors=6,
        coordinate_noise=0.0,
        dropout=0.0,
    )
    torch.manual_seed(11)
    reference = ProteinMPNN(ProteinMPNNConfig(**common)).eval()
    candidate = ProteinMPNN(
        ProteinMPNNConfig(**common, knn_backend="chunked", knn_query_chunk=3)
    ).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)

    length = 17
    generator = torch.Generator().manual_seed(12)
    values = (
        torch.randn(1, length, 4, 3, generator=generator),
        torch.randint(0, 21, (1, length), generator=generator),
        torch.ones(1, length),
        torch.arange(length).unsqueeze(0),
        torch.zeros(1, length, dtype=torch.long),
        torch.randperm(length, generator=generator).unsqueeze(0),
        (torch.arange(length) // 4).unsqueeze(0),
    )
    with torch.no_grad():
        torch.testing.assert_close(
            candidate(*values), reference(*values), atol=0, rtol=0
        )
