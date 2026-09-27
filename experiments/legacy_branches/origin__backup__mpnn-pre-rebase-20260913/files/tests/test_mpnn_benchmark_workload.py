from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.modules.mpnn.workload import build_mpnn_workload  # noqa: E402
from miniworld_kernels.modules.mpnn import (  # noqa: E402
    item_balanced_cross_entropy,
)


def _legacy_seeded_tensors(
    *,
    seq_len: int,
    batch_size: int,
    layout: Literal["batch", "packed"],
    patch_size: int,
) -> tuple[torch.Tensor, ...]:
    if layout == "packed":
        physical_batch_size = 1
        total_length = batch_size * seq_len
    else:
        physical_batch_size = batch_size
        total_length = seq_len

    coordinates = torch.randn(physical_batch_size, total_length, 4, 3)
    sequence = torch.randint(0, 21, (physical_batch_size, total_length))
    mask = torch.ones(physical_batch_size, total_length)
    local_patch_index = torch.arange(
        (seq_len + patch_size - 1) // patch_size,
    ).repeat_interleave(patch_size)[:seq_len]
    if layout == "packed":
        residue_index = torch.arange(seq_len).repeat(batch_size).unsqueeze(0)
        chain_index = torch.zeros(1, total_length, dtype=torch.long)
        decoding_order = torch.cat(
            [
                torch.randperm(seq_len) + segment * seq_len
                for segment in range(batch_size)
            ]
        ).unsqueeze(0)
        patch_index = torch.cat(
            [local_patch_index + segment * seq_len for segment in range(batch_size)]
        ).unsqueeze(0)
        segment_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    else:
        residue_index = torch.arange(seq_len).expand(physical_batch_size, -1)
        chain_index = torch.zeros(
            physical_batch_size,
            seq_len,
            dtype=torch.long,
        )
        decoding_order = torch.stack(
            [torch.randperm(seq_len) for _ in range(physical_batch_size)]
        )
        patch_index = local_patch_index.expand(physical_batch_size, -1)
        segment_lengths = None
    upstream_gradient = torch.randn(physical_batch_size, 21, total_length)
    return (
        coordinates,
        sequence,
        mask,
        residue_index,
        chain_index,
        decoding_order,
        patch_index,
        segment_lengths,
        upstream_gradient,
    )


@pytest.mark.parametrize("layout", ["batch", "packed"])
def test_seeded_workload_matches_original_runner_tensor_stream(
    layout: Literal["batch", "packed"],
) -> None:
    kwargs = {
        "seq_len": 7,
        "batch_size": 3,
        "layout": layout,
        "patch_size": 4,
    }
    torch.manual_seed(123)
    expected = _legacy_seeded_tensors(**kwargs)
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(123)
    actual = build_mpnn_workload(
        **kwargs,
        k_neighbors=7,
        training=True,
        coordinate_grad=False,
        objective="output_grad",
        label_smoothing=0.1,
        use_amp=False,
        device="cpu",
    )
    actual_rng_state = torch.get_rng_state()

    actual_tensors = (
        actual.coordinates,
        actual.sequence,
        actual.mask,
        actual.residue_index,
        actual.chain_index,
        actual.decoding_order,
        actual.patch_index,
        actual.segment_lengths,
        actual.upstream_gradient,
    )
    for actual_tensor, expected_tensor in zip(actual_tensors, expected, strict=True):
        if expected_tensor is None:
            assert actual_tensor is None
        else:
            assert actual_tensor is not None
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    torch.testing.assert_close(actual_rng_state, expected_rng_state, rtol=0, atol=0)


def test_model_signatures_share_the_same_workload_tensors() -> None:
    workload = build_mpnn_workload(
        seq_len=8,
        batch_size=2,
        layout="batch",
        patch_size=4,
        k_neighbors=8,
        training=False,
        coordinate_grad=False,
        objective="output_grad",
        label_smoothing=0.1,
        use_amp=True,
        device="cpu",
    )

    production = workload.model_inputs(production_signature=True)
    reference = workload.model_inputs(production_signature=False)
    assert len(production) == 7
    assert len(reference) == 8
    assert all(
        production_tensor is reference_tensor
        for production_tensor, reference_tensor in zip(
            production[:6],
            reference[:6],
            strict=True,
        )
    )
    assert reference[6] is workload.mask
    assert production[-1] is reference[-1] is None
    assert workload.upstream_gradient is not None
    assert workload.upstream_gradient.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runner requires CUDA")
def test_edge_w1_runner_resolver_matches_runtime_policy_boundaries() -> None:
    # The benchmark runner intentionally rejects CPU-only imports, so keep this
    # policy test allocation-free but run it only in the existing CUDA suite.
    from benchmarks.runners.bench import (
        BenchConfig,
        _resolved_mpnn_edge_w1_recompute,
    )

    base = {
        "kernel": "mpnn",
        "mode": "training",
        "metric": "time",
        "batch_size": 1,
        "k_neighbors": 48,
        "d_pair": 128,
        "precision": "bf16-mixed",
        "mpnn_edge_mlp_backend": "triton_memory",
        "mpnn_edge_w1_recompute": "checkpoint",
    }

    def resolved(seq_len: int, **updates) -> str:
        return _resolved_mpnn_edge_w1_recompute(
            BenchConfig(**(base | updates)),
            seq_len,
        )

    assert resolved(1023) == "off"
    assert resolved(1024) == "checkpoint"
    assert resolved(1024, mode="inference") == "off"
    assert resolved(1024, precision=32) == "off"
    assert resolved(1024, d_pair=64) == "off"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runner requires CUDA")
def test_encoder_node_w1_runner_resolver_matches_runtime_policy_boundaries() -> None:
    from benchmarks.runners.bench import (
        BenchConfig,
        _resolved_mpnn_encoder_node_w1_recompute,
    )

    base = {
        "kernel": "mpnn",
        "mode": "training",
        "metric": "time",
        "batch_size": 1,
        "k_neighbors": 48,
        "d_pair": 128,
        "precision": "bf16-mixed",
        "mpnn_message_backend": "triton_memory",
        "mpnn_encoder_node_w1_recompute": "checkpoint",
    }

    def resolved(seq_len: int, **updates) -> str:
        return _resolved_mpnn_encoder_node_w1_recompute(
            BenchConfig(**(base | updates)),
            seq_len,
        )

    assert resolved(1023) == "off"
    assert resolved(1024) == "checkpoint"
    assert resolved(1024, mode="inference") == "off"
    assert resolved(1024, precision=32) == "off"
    assert resolved(1024, d_pair=64) == "off"
    assert resolved(1024, k_neighbors=32) == "off"
    assert resolved(1024, mpnn_message_backend="triton_compute") == "off"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runner requires CUDA")
def test_edge_dropout_runner_resolver_agrees_with_runtime_dispatch() -> None:
    """The reported dropout policy must be the one the library actually picks.

    The runner mirrors dispatch only to label rows, so a looser mirror would
    silently record ``bitpack`` for a run that fell back to native dropout.
    """
    from benchmarks.runners.bench import (
        MPNN_BENCH_DROPOUT,
        BenchConfig,
        _resolved_mpnn_edge_dropout_backend,
    )
    from miniworld_kernels.kernels.mpnn_edge_dropout.interface import (
        _bitpack_supported,
        _select_backend,
    )

    base = {
        "kernel": "mpnn",
        "mode": "training",
        "metric": "time",
        "batch_size": 1,
        "k_neighbors": 48,
        "d_pair": 128,
        "precision": "bf16-mixed",
        "mpnn_edge_dropout_backend": "bitpack",
    }

    def resolved(seq_len: int, **updates) -> str:
        return _resolved_mpnn_edge_dropout_backend(
            BenchConfig(**(base | updates)),
            seq_len,
        )

    assert resolved(256) == "bitpack"
    assert resolved(256, mode="inference") == "pytorch"
    assert resolved(256, mpnn_edge_dropout_backend="auto") == "pytorch"

    # The runner's label and the library's own decision must agree on a tensor
    # shaped exactly like one encoder edge update of that configuration.
    values = torch.randn(
        1, 256, 48, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    supported = _bitpack_supported(
        values,
        MPNN_BENCH_DROPOUT,
        training=True,
        inplace=False,
    )
    assert _select_backend("bitpack", supported=supported) == resolved(256)

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        assert resolved(256) == "pytorch"
        assert (
            _select_backend(
                "bitpack",
                supported=_bitpack_supported(
                    values,
                    MPNN_BENCH_DROPOUT,
                    training=True,
                    inplace=False,
                ),
            )
            == "pytorch"
        )
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runner requires CUDA")
def test_transition_recompute_runner_resolver_is_training_only() -> None:
    from benchmarks.runners.bench import (
        BenchConfig,
        _resolved_mpnn_transition_recompute,
    )

    base = {
        "kernel": "mpnn",
        "mode": "training",
        "metric": "time",
        "mpnn_transition_recompute": "update",
    }
    assert _resolved_mpnn_transition_recompute(BenchConfig(**base)) == "update"
    assert (
        _resolved_mpnn_transition_recompute(
            BenchConfig(**(base | {"mode": "inference"}))
        )
        == "off"
    )
    assert (
        _resolved_mpnn_transition_recompute(
            BenchConfig(**(base | {"mpnn_transition_recompute": "off"}))
        )
        == "off"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runner requires CUDA")
def test_edge_dropout_runner_resolver_is_explicit_and_training_only() -> None:
    from benchmarks.runners.bench import (
        BenchConfig,
        _resolved_mpnn_edge_dropout_backend,
    )
    from miniworld_kernels.kernels.mpnn_edge_dropout.interface import (
        _INT32_MAX,
        _PADDED_TILE_ELEMENTS,
    )

    base = {
        "kernel": "mpnn",
        "mode": "training",
        "metric": "time",
        "batch_size": 1,
        "k_neighbors": 48,
        "d_pair": 128,
        "mpnn_edge_dropout_backend": "bitpack",
    }

    def resolved(seq_len: int, **updates) -> str:
        return _resolved_mpnn_edge_dropout_backend(
            BenchConfig(**(base | updates)),
            seq_len,
        )

    assert resolved(2048) == "bitpack"
    assert resolved(2048, mpnn_edge_dropout_backend="auto") == "pytorch"
    assert resolved(2048, mpnn_edge_dropout_backend="pytorch") == "pytorch"
    assert resolved(2048, mode="inference") == "pytorch"
    largest = _INT32_MAX - (_PADDED_TILE_ELEMENTS - 1)
    overflowing_batch = largest // (2048 * 48 * 128) + 1
    assert resolved(2048, batch_size=overflowing_batch) == "pytorch"


def test_output_gradient_objective_passes_static_gradient_to_backward() -> None:
    workload = build_mpnn_workload(
        seq_len=5,
        batch_size=2,
        layout="batch",
        patch_size=2,
        k_neighbors=5,
        training=True,
        coordinate_grad=False,
        objective="output_grad",
        label_smoothing=0.1,
        use_amp=False,
        device="cpu",
    )
    output = torch.randn(2, 21, 5, requires_grad=True)
    backward_calls = 0

    def backward(tensor: torch.Tensor, gradient: torch.Tensor | None) -> None:
        nonlocal backward_calls
        backward_calls += 1
        tensor.backward(gradient)

    workload.backward(output, backward)

    assert backward_calls == 1
    assert workload.upstream_gradient is not None
    torch.testing.assert_close(output.grad, workload.upstream_gradient)


def test_item_ce_objective_matches_public_item_balanced_loss() -> None:
    workload = build_mpnn_workload(
        seq_len=6,
        batch_size=3,
        layout="batch",
        patch_size=4,
        k_neighbors=6,
        training=True,
        coordinate_grad=False,
        objective="item_ce",
        label_smoothing=0.2,
        use_amp=False,
        device="cpu",
    )
    actual = torch.randn(3, 21, 6, requires_grad=True)
    expected = actual.detach().clone().requires_grad_(True)

    workload.backward(actual, lambda tensor, gradient: tensor.backward(gradient))
    expected_loss = item_balanced_cross_entropy(
        expected,
        workload.sequence,
        workload.mask,
        residue_mask=workload.mask,
        label_smoothing=0.2,
    ).loss
    expected_loss.backward()

    assert workload.upstream_gradient is None
    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"patch_size": 0}, "patch size must be positive"),
        (
            {"layout": "single", "batch_size": 2},
            "single layout requires batch_size=1",
        ),
        ({"label_smoothing": 1.1}, "label smoothing must satisfy"),
        (
            {"layout": "packed", "objective": "item_ce"},
            "item-balanced CE requires a true",
        ),
    ],
)
def test_invalid_workloads_are_rejected(
    overrides: dict[str, object],
    match: str,
) -> None:
    kwargs: dict[str, object] = {
        "seq_len": 8,
        "batch_size": 2,
        "layout": "batch",
        "patch_size": 4,
        "k_neighbors": 8,
        "training": True,
        "coordinate_grad": False,
        "objective": "output_grad",
        "label_smoothing": 0.1,
        "use_amp": False,
        "device": "cpu",
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        build_mpnn_workload(**kwargs)  # type: ignore[arg-type]
