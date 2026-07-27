from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_kernels.kernels.mpnn_edge_dropout import edge_dropout
from miniworld_kernels.kernels.mpnn_edge_dropout.interface import (
    _INT32_MAX,
    _PADDED_TILE_ELEMENTS,
    _bitpack_shape_supported,
    _select_backend,
)
from miniworld_kernels.modules.mpnn import (
    EdgeDropout,
    ProteinMPNN,
    ProteinMPNNConfig,
)


def test_mpnn_edge_dropout_policy_is_explicit() -> None:
    assert _select_backend("auto", supported=True) == "pytorch"
    assert _select_backend("pytorch", supported=True) == "pytorch"
    assert _select_backend("bitpack", supported=True) == "bitpack"
    assert _select_backend("bitpack", supported=False) == "pytorch"


def test_mpnn_edge_dropout_signed_int_limit_is_conservative() -> None:
    largest = _INT32_MAX - (_PADDED_TILE_ELEMENTS - 1)
    assert _bitpack_shape_supported(largest)
    assert not _bitpack_shape_supported(0)
    assert not _bitpack_shape_supported(largest + 1)


def test_mpnn_edge_dropout_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="edge dropout backend"):
        edge_dropout(
            torch.randn(7),
            0.1,
            training=True,
            backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="edge_dropout_backend"):
        ProteinMPNNConfig(edge_dropout_backend="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["auto", "pytorch", "bitpack"])
def test_mpnn_edge_dropout_cpu_preserves_native_output_and_rng(backend: str) -> None:
    values = torch.linspace(-2.0, 2.0, 257, requires_grad=True)

    torch.manual_seed(101)
    expected = F.dropout(values, p=0.1, training=True)
    expected_rng = torch.get_rng_state().clone()

    torch.manual_seed(101)
    actual = edge_dropout(
        values,
        0.1,
        training=True,
        backend=backend,  # type: ignore[arg-type]
    )
    actual_rng = torch.get_rng_state().clone()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)


def test_mpnn_edge_dropout_no_grad_and_eval_preserve_native_semantics() -> None:
    values = torch.linspace(-2.0, 2.0, 257)
    with torch.no_grad():
        torch.manual_seed(103)
        expected = F.dropout(values, p=0.1, training=True)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(103)
        actual = edge_dropout(
            values,
            0.1,
            training=True,
            backend="bitpack",
        )
        actual_rng = torch.get_rng_state().clone()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)

    evaluated = edge_dropout(
        values,
        0.1,
        training=False,
        backend="bitpack",
    )
    assert evaluated is values


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_mpnn_edge_dropout_probability_boundaries_fall_back(
    probability: float,
) -> None:
    values = torch.linspace(-2.0, 2.0, 17, requires_grad=True)
    torch.manual_seed(104)
    expected = F.dropout(values, p=probability, training=True)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(104)
    actual = edge_dropout(
        values,
        probability,
        training=True,
        backend="bitpack",
    )
    actual_rng = torch.get_rng_state().clone()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)


def test_mpnn_edge_dropout_lazy_fallback_does_not_import_triton() -> None:
    code = """
import sys
import torch
from miniworld_kernels.kernels.mpnn_edge_dropout import edge_dropout
from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig

model = ProteinMPNN(ProteinMPNNConfig(edge_dropout_backend="bitpack"))
with torch.no_grad():
    edge_dropout(torch.randn(257), 0.1, training=True, backend="bitpack")
model.eval()
model.encoder.layers[0].edge_message.dropout(torch.randn(257))
assert "triton" not in sys.modules
assert "miniworld_kernels.kernels.mpnn_edge_dropout.triton.main" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mpnn_edge_dropout_is_encoder_edge_only() -> None:
    model = ProteinMPNN(ProteinMPNNConfig(edge_dropout_backend="bitpack"))
    for layer in model.encoder.layers:
        assert isinstance(layer.edge_message.dropout, EdgeDropout)
        assert layer.edge_message.dropout.backend == "bitpack"
        assert type(layer.node_message.dropout) is nn.Dropout
        assert type(layer.node_transition.dropout) is nn.Dropout
    for layer in model.decoder.layers:
        assert type(layer.node_message.dropout) is nn.Dropout
        assert type(layer.node_transition.dropout) is nn.Dropout


def test_mpnn_edge_dropout_preserves_state_dict_and_module_hooks() -> None:
    native = ProteinMPNN(ProteinMPNNConfig(edge_dropout_backend="pytorch"))
    packed = ProteinMPNN(ProteinMPNNConfig(edge_dropout_backend="bitpack"))
    native_state = native.state_dict()
    packed_state = packed.state_dict()
    assert native_state.keys() == packed_state.keys()
    assert {name: value.shape for name, value in native_state.items()} == {
        name: value.shape for name, value in packed_state.items()
    }
    packed.load_state_dict(native_state, strict=True)

    dropout = packed.encoder.layers[0].edge_message.dropout
    forward_calls: list[torch.Tensor] = []
    backward_calls: list[torch.Tensor] = []
    forward_handle = dropout.register_forward_hook(
        lambda _module, _inputs, output: forward_calls.append(output)
    )
    try:
        values = torch.randn(257, requires_grad=True)
        output = dropout(values)
    finally:
        forward_handle.remove()
    assert len(forward_calls) == 1
    assert forward_calls[0] is output

    backward_handle = dropout.register_full_backward_hook(
        lambda _module, _grad_input, grad_output: backward_calls.append(grad_output[0])
    )
    try:
        dropout(values).sum().backward()
    finally:
        backward_handle.remove()
    assert len(backward_calls) == 1


def _cuda_contract(
    size: int,
    dtype: torch.dtype,
    *,
    probability: float = 0.1,
    compiled: bool = False,
) -> None:
    torch.manual_seed(107)
    expected_values = torch.randn(
        size,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    actual_values = expected_values.detach().clone().requires_grad_()
    upstream = torch.randn_like(expected_values)

    native = lambda values: F.dropout(values, p=probability, training=True)
    candidate = lambda values: edge_dropout(
        values,
        probability,
        training=True,
        backend="bitpack",
    )
    if compiled:
        native = torch.compile(native, fullgraph=True)
        candidate = torch.compile(candidate, fullgraph=True)

    torch.cuda.manual_seed_all(109)
    expected = native(expected_values)
    expected_forward_rng = torch.cuda.get_rng_state().clone()
    (expected_gradient,) = torch.autograd.grad(expected, expected_values, upstream)
    expected_backward_rng = torch.cuda.get_rng_state().clone()

    torch.cuda.manual_seed_all(109)
    actual = candidate(actual_values)
    actual_forward_rng = torch.cuda.get_rng_state().clone()
    (actual_gradient,) = torch.autograd.grad(actual, actual_values, upstream)
    actual_backward_rng = torch.cuda.get_rng_state().clone()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_forward_rng, expected_forward_rng)
    assert torch.equal(actual_backward_rng, expected_backward_rng)
    assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("size", [1, 7, 8, 9, 257])
def test_mpnn_edge_dropout_matches_native_eager(
    size: int,
    dtype: torch.dtype,
) -> None:
    _cuda_contract(size, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_dropout_matches_native_at_production_size() -> None:
    _cuda_contract(12_582_912, torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_dropout_runtime_scale_matches_native_bf16_rounding() -> None:
    _cuda_contract(257, torch.bfloat16, probability=0.37)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ["float16", "noncontiguous"])
def test_mpnn_edge_dropout_unsupported_cuda_inputs_fall_back(kind: str) -> None:
    if kind == "float16":
        expected_values = torch.randn(
            257,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        actual_values = expected_values.detach().clone().requires_grad_()
    else:
        base = torch.randn(11, 13, device="cuda", dtype=torch.bfloat16)
        expected_values = base.T.detach().requires_grad_()
        actual_values = base.clone().T.detach().requires_grad_()
        assert not expected_values.is_contiguous()
        assert not actual_values.is_contiguous()
    upstream = torch.randn_like(expected_values)

    torch.cuda.manual_seed_all(111)
    expected = F.dropout(expected_values, p=0.1, training=True)
    expected_rng = torch.cuda.get_rng_state().clone()
    (expected_gradient,) = torch.autograd.grad(
        expected,
        expected_values,
        upstream,
    )
    torch.cuda.manual_seed_all(111)
    actual = edge_dropout(
        actual_values,
        0.1,
        training=True,
        backend="bitpack",
    )
    actual_rng = torch.cuda.get_rng_state().clone()
    (actual_gradient,) = torch.autograd.grad(actual, actual_values, upstream)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_dropout_saves_only_packed_mask() -> None:
    values = torch.randn(
        257,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def saved_by(function) -> list[torch.Tensor]:
        saved: list[torch.Tensor] = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            output = function(values)
        output.sum().backward()
        values.grad = None
        return saved

    native = saved_by(lambda value: F.dropout(value, p=0.1, training=True))
    packed = saved_by(
        lambda value: edge_dropout(
            value,
            0.1,
            training=True,
            backend="bitpack",
        )
    )
    assert [(tensor.dtype, tensor.shape) for tensor in native] == [
        (torch.bool, values.shape)
    ]
    assert [(tensor.dtype, tensor.shape) for tensor in packed] == [
        (torch.uint8, torch.Size([(values.numel() + 7) // 8]))
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_dropout_deterministic_mode_uses_native_mask() -> None:
    values = torch.randn(
        257,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    saved: list[torch.Tensor] = []
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            output = edge_dropout(
                values,
                0.1,
                training=True,
                backend="bitpack",
            )
        output.sum().backward()
    finally:
        torch.use_deterministic_algorithms(previous)
    assert [(tensor.dtype, tensor.shape) for tensor in saved] == [
        (torch.bool, values.shape)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_dropout_is_fullgraph_compilable() -> None:
    _cuda_contract(257, torch.bfloat16, compiled=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_dropout_rejects_double_backward() -> None:
    values = torch.randn(
        257,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    upstream = torch.randn_like(values, requires_grad=True)
    output = edge_dropout(values, 0.1, training=True, backend="bitpack")
    (first_gradient,) = torch.autograd.grad(
        output,
        values,
        upstream,
        create_graph=True,
    )
    with pytest.raises(RuntimeError, match="differentiate twice|once_differentiable"):
        first_gradient.sum().backward()
