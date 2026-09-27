from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.mpnn_edge_layernorm import edge_layer_norm
from miniworld_kernels.kernels.mpnn_edge_layernorm.interface import (
    _INT32_MAX,
    _select_backend,
)
from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig


def test_mpnn_edge_layernorm_cpu_and_unsupported_memory_fall_back() -> None:
    torch.manual_seed(73)
    values = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    expected = F.layer_norm(values, (7,), weight, bias, 1e-5)
    actual = edge_layer_norm(values, weight, bias, 1e-5, backend="memory")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mpnn_edge_layernorm_policy_is_explicit_and_grad_aware() -> None:
    values = torch.empty(2048 * 48, 128, device="meta")
    with torch.enable_grad():
        assert _select_backend(values, "auto", supported=True) == "pytorch"
        assert _select_backend(values, "pytorch", supported=True) == "pytorch"
        assert _select_backend(values, "memory", supported=True) == "memory"
        assert _select_backend(values, "memory", supported=False) == "pytorch"
    with torch.no_grad():
        assert _select_backend(values, "memory", supported=True) == "pytorch"


def test_mpnn_edge_layernorm_signed_int_limit_is_conservative() -> None:
    assert _INT32_MAX == 2**31 - 1


@pytest.mark.parametrize("backend", ["auto", "pytorch"])
def test_mpnn_edge_layernorm_compute_policy_preserves_module_hooks(
    backend: str,
) -> None:
    model = ProteinMPNN(ProteinMPNNConfig(edge_norm_backend=backend))
    message = model.encoder.layers[0].edge_message
    calls: list[torch.Tensor] = []
    handle = message.norm.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output)
    )
    try:
        output = message.apply_edgewise_update(
            torch.randn(2, 3, 128),
            torch.randn(2, 3, 128),
        )
    finally:
        handle.remove()
    assert len(calls) == 1
    assert calls[0] is output


def _cuda_inputs(rows: int, dtype: torch.dtype = torch.float32):
    torch.manual_seed(79)
    values = torch.randn(
        rows,
        128,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    weight = torch.randn(128, device="cuda", dtype=torch.float32).requires_grad_()
    bias = torch.randn(128, device="cuda", dtype=torch.float32).requires_grad_()
    upstream = torch.randn(rows, 128, device="cuda", dtype=torch.float32)
    return values, weight, bias, upstream


def _clone_inputs(values):
    return tuple(
        value.detach().clone().requires_grad_(value.requires_grad) for value in values
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.detach().float() - expected.detach().float()
    denominator = expected.detach().float().norm().clamp_min(1e-20)
    return float(difference.norm() / denominator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mpnn_edge_layernorm_preserves_forward_and_tracks_native_gradients(
    dtype: torch.dtype,
) -> None:
    actual_inputs = _cuda_inputs(257, dtype)
    expected_inputs = _clone_inputs(actual_inputs[:3])
    upstream = actual_inputs[3]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = edge_layer_norm(
            *actual_inputs[:3],
            1e-5,
            backend="memory",
        )
        expected = F.layer_norm(
            expected_inputs[0],
            (128,),
            expected_inputs[1],
            expected_inputs[2],
            1e-5,
        )

    actual_gradients = torch.autograd.grad(actual, actual_inputs[:3], upstream)
    expected_gradients = torch.autograd.grad(
        expected,
        expected_inputs,
        upstream,
    )

    # The selected path calls the same native forward under the same autocast
    # context. Only its backward save is compressed.
    assert torch.equal(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert _relative_error(actual_gradient, expected_gradient) < 5e-3
        assert (
            torch.nn.functional.cosine_similarity(
                actual_gradient.float().flatten(),
                expected_gradient.float().flatten(),
                dim=0,
            )
            > 0.99998
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_layernorm_memory_saves_only_bf16_edge_input() -> None:
    values, weight, bias, _ = _cuda_inputs(257)
    saved: list[torch.Tensor] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = edge_layer_norm(
                values,
                weight,
                bias,
                1e-5,
                backend="memory",
            )

    edge_sized = [tensor for tensor in saved if tensor.shape == values.shape]
    assert len(edge_sized) == 1
    assert edge_sized[0].dtype == torch.bfloat16
    assert edge_sized[0].data_ptr() != values.data_ptr()
    output.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_layernorm_module_boundary_matches_native() -> None:
    torch.manual_seed(83)
    reference_model = ProteinMPNN(
        ProteinMPNNConfig(dropout=0.0, edge_norm_backend="pytorch")
    ).cuda()
    candidate_model = ProteinMPNN(
        ProteinMPNNConfig(dropout=0.0, edge_norm_backend="memory")
    ).cuda()
    candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
    reference = reference_model.encoder.layers[0].edge_message
    candidate = candidate_model.encoder.layers[0].edge_message

    reference_states = torch.randn(
        257,
        128,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_update = torch.randn(
        257,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    candidate_states, candidate_update = _clone_inputs(
        (reference_states, reference_update)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference.apply_edgewise_update(
            reference_states,
            reference_update,
        )
        actual = candidate.apply_edgewise_update(
            candidate_states,
            candidate_update,
        )

    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        actual,
        (
            candidate_states,
            candidate_update,
            candidate.norm.weight,
            candidate.norm.bias,
        ),
        upstream,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (
            reference_states,
            reference_update,
            reference.norm.weight,
            reference.norm.bias,
        ),
        upstream,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert _relative_error(actual_gradient, expected_gradient) < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_layernorm_deterministic_mode_uses_native_fallback() -> None:
    values, weight, bias, _ = _cuda_inputs(257)
    saved: list[torch.Tensor] = []
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = edge_layer_norm(
                    values,
                    weight,
                    bias,
                    1e-5,
                    backend="memory",
                )
        output.sum().backward()
    finally:
        torch.use_deterministic_algorithms(previous)

    edge_sized = [tensor for tensor in saved if tensor.shape == values.shape]
    assert len(edge_sized) == 1
    assert edge_sized[0].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_layernorm_deterministic_toggle_uses_partial_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from miniworld_kernels.kernels.mpnn_edge_layernorm.triton import main

    values, weight, bias, _ = _cuda_inputs(257)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = edge_layer_norm(
            values,
            weight,
            bias,
            1e-5,
            backend="memory",
        )

    def forbidden_atomic(*_args, **_kwargs):
        raise AssertionError("deterministic backward selected the atomic path")

    monkeypatch.setattr(main, "_bwd_atomic_impl", forbidden_atomic)
    try:
        torch.use_deterministic_algorithms(True)
        output.sum().backward()
    finally:
        torch.use_deterministic_algorithms(previous)
    assert values.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_layernorm_memory_is_fullgraph_compilable() -> None:
    values, weight, bias, upstream = _cuda_inputs(257)

    def forward(x, w, b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return edge_layer_norm(x, w, b, 1e-5, backend="memory")

    compiled = torch.compile(forward, fullgraph=True)
    output = compiled(values, weight, bias)
    output.backward(upstream)

    assert output.shape == values.shape
    assert values.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
