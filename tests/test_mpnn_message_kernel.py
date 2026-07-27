from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_kernels.kernels.mpnn_message import (
    message_hidden_reduce,
    message_hidden_reduce_pytorch,
)
from miniworld_kernels.kernels.mpnn_message.interface import _should_use_triton
from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig


def test_mpnn_message_reference_contract_and_fallback() -> None:
    torch.manual_seed(0)
    preactivation = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 7, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    edge_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    expected = message_hidden_reduce_pytorch(preactivation, weight, bias, edge_mask, 8)
    actual = message_hidden_reduce(
        preactivation, weight, bias, edge_mask, 8, backend="auto"
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (2, 7)
    assert torch.count_nonzero(actual[1]) == 0


def test_mpnn_message_auto_policy_uses_supported_triton_path() -> None:
    with torch.enable_grad():
        assert _should_use_triton(True, "auto")
        assert _should_use_triton(True, "triton")
        assert _should_use_triton(True, "triton_compute")
        assert _should_use_triton(True, "triton_memory")
    with torch.no_grad():
        assert _should_use_triton(True, "auto")
        assert not _should_use_triton(True, "pytorch")
    assert not _should_use_triton(False, "triton")


def test_mpnn_message_int32_guard_accounts_for_padded_dx_tile() -> None:
    from miniworld_kernels.kernels.mpnn_message._policy import (
        _DX_TILE_ELEMENTS,
        _INT32_MAX,
        _requires_i64_indexing,
    )

    largest_safe_numel = _INT32_MAX - (_DX_TILE_ELEMENTS - 1)
    assert not _requires_i64_indexing(largest_safe_numel)
    assert _requires_i64_indexing(largest_safe_numel + 1)


def test_mpnn_message_inference_int32_guard_accounts_for_odd_group_tail() -> None:
    from miniworld_kernels.kernels.mpnn_message._policy import (
        _INT32_MAX,
        _INFERENCE_PADDED_TAIL_ELEMENTS,
        _inference_int32_elements_supported,
    )

    largest_safe_numel = _INT32_MAX - (_INFERENCE_PADDED_TAIL_ELEMENTS - 1)
    assert _inference_int32_elements_supported(largest_safe_numel)
    assert not _inference_int32_elements_supported(largest_safe_numel + 1)


def test_mpnn_message_policy_has_no_shape_keyed_dispatch() -> None:
    """Backward must select the same operations at every shape.

    A shape-keyed dX policy (``groups in {8192}``) previously changed both the
    gradient rounding and the launch sequence at one calibrated batch size.
    """
    from miniworld_kernels.kernels.mpnn_message import _policy

    assert not hasattr(_policy, "_should_use_pytorch_dx")
    assert not hasattr(_policy, "_PYTORCH_DX_GROUPS")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_backward_survives_dynamic_shape_compilation() -> None:
    """Nothing in backward may hash a shape-derived value.

    Under dynamic-shape compilation ``preactivation.numel()`` is a ``SymInt``.
    Using it as a set/dict key made AOTAutograd tracing fail with
    ``TypeError: unhashable type: non-nested SymInt``, which surfaced only on the
    second compilation of an already-traced frame.
    """

    def reduce(preactivation, weight, bias, edge_mask):
        return message_hidden_reduce(
            preactivation,
            weight,
            bias,
            edge_mask,
            48,
            backend="triton_memory",
        )

    compiled = torch.compile(reduce, fullgraph=True, dynamic=True)
    for groups in (64, 96):
        preactivation, weight, bias, edge_mask, upstream = _cuda_inputs(groups)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reduced = compiled(preactivation, weight, bias, edge_mask)
        gradients = torch.autograd.grad(
            reduced,
            (preactivation, weight, bias),
            upstream,
        )
        assert all(gradient is not None for gradient in gradients)


def _cuda_inputs(groups: int):
    torch.manual_seed(7)
    preactivation = torch.randn(
        groups, 48, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = (
        torch.randn(128, 128, device="cuda", dtype=torch.float32, requires_grad=True)
        / 128**0.5
    )
    weight = weight.detach().requires_grad_(True)
    bias = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
    edge_mask = (torch.rand(groups, 48, device="cuda") > 0.2).float()
    edge_mask[0].zero_()
    upstream = torch.randn(groups, 128, device="cuda", dtype=torch.float32)
    return preactivation, weight, bias, edge_mask, upstream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("groups", [3, 17])
@pytest.mark.parametrize("backend", ["triton", "triton_memory"])
def test_mpnn_message_triton_matches_bf16_reference(
    groups: int,
    backend: str,
) -> None:
    actual_inputs = _cuda_inputs(groups)
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        for value in actual_inputs[:4]
    )
    upstream = actual_inputs[4]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = message_hidden_reduce(*actual_inputs[:4], 48, backend=backend)
        expected = message_hidden_reduce_pytorch(*expected_inputs, 48)

    actual_gradients = torch.autograd.grad(
        actual, actual_inputs[:3], upstream, retain_graph=False
    )
    expected_gradients = torch.autograd.grad(
        expected, expected_inputs[:3], upstream, retain_graph=False
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_auxiliary_is_non_differentiable() -> None:
    from miniworld_kernels.kernels.mpnn_message.triton.main import _forward_op

    preactivation, weight, bias, edge_mask, _ = _cuda_inputs(3)
    reduced, projected = _forward_op(
        preactivation,
        weight,
        bias,
        edge_mask,
        48,
    )

    assert reduced.requires_grad
    assert not projected.requires_grad
    assert projected.grad_fn is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_memory_backend_does_not_save_projected() -> None:
    preactivation, weight, bias, edge_mask, _ = _cuda_inputs(3)
    saved: list[torch.Tensor] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = message_hidden_reduce(
                preactivation,
                weight,
                bias,
                edge_mask,
                48,
                backend="triton_memory",
            )

    full_edge_tensors = [
        tensor for tensor in saved if tensor.shape == preactivation.shape
    ]
    assert len(full_edge_tensors) == 1
    assert full_edge_tensors[0] is preactivation
    output.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_dx_epilogue_emits_bf16_gelu() -> None:
    from miniworld_kernels.kernels.mpnn_message.triton.main import _projection_dx_op

    preactivation, weight, _bias, _edge_mask, _ = _cuda_inputs(3)
    grad_projected = torch.randn_like(preactivation)
    grad_preactivation, activated = _projection_dx_op(
        grad_projected,
        weight,
        preactivation,
    )

    assert grad_preactivation.shape == preactivation.shape
    assert grad_preactivation.dtype == preactivation.dtype
    assert activated.shape == preactivation.shape
    assert activated.dtype == torch.bfloat16
    torch.testing.assert_close(
        activated,
        F.gelu(preactivation),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_backward_emits_fp32_bias_partials() -> None:
    from miniworld_kernels.kernels.mpnn_message.triton.main import (
        _reduce_backward_op,
    )

    groups = 3
    projected = torch.randn(groups, 48, 128, device="cuda", dtype=torch.bfloat16)
    grad_reduced = torch.randn(groups, 128, device="cuda")
    edge_mask = (torch.rand(groups, 48, device="cuda") > 0.2).float()

    grad_projected, grad_bias_partial = _reduce_backward_op(
        grad_reduced,
        projected,
        edge_mask,
        48,
    )

    assert grad_projected.shape == projected.shape
    assert grad_projected.dtype == torch.bfloat16
    assert grad_bias_partial.shape == (groups, 128)
    assert grad_bias_partial.dtype == torch.float32
    torch.testing.assert_close(
        grad_bias_partial,
        grad_projected.float().sum(dim=1),
        atol=2e-2,
        rtol=2e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_atomic_bias_gradient_matches_emitted_dp() -> None:
    from miniworld_kernels.kernels.mpnn_message.triton.main import (
        _reduce_backward_atomic_op,
    )

    groups = 17
    projected = torch.randn(groups, 48, 128, device="cuda", dtype=torch.bfloat16)
    grad_reduced = torch.randn(groups, 128, device="cuda")
    edge_mask = (torch.rand(groups, 48, device="cuda") > 0.2).float()

    grad_projected, grad_bias = _reduce_backward_atomic_op(
        grad_reduced,
        projected,
        edge_mask,
        48,
    )

    assert grad_projected.shape == projected.shape
    assert grad_projected.dtype == torch.bfloat16
    assert grad_bias.shape == (128,)
    assert grad_bias.dtype == torch.float32
    torch.testing.assert_close(
        grad_bias,
        grad_projected.float().sum(dim=(0, 1)),
        atol=3e-2,
        rtol=3e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton", "triton_memory"])
def test_mpnn_message_deterministic_mode_has_repeatable_bias_gradient(
    backend: str,
) -> None:
    preactivation, weight, bias, edge_mask, upstream = _cuda_inputs(17)
    was_deterministic = torch.are_deterministic_algorithms_enabled()

    def forward(x, w, b, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return message_hidden_reduce(x, w, b, mask, 48, backend=backend)

    try:
        torch.use_deterministic_algorithms(True)
        compiled = torch.compile(forward, fullgraph=True)
        output = compiled(preactivation, weight, bias, edge_mask)
        first = torch.autograd.grad(output, bias, upstream, retain_graph=True)[0]
        second = torch.autograd.grad(output, bias, upstream, retain_graph=False)[0]
        assert torch.equal(first, second)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_forced_triton_requires_bf16_projection() -> None:
    preactivation, weight, bias, edge_mask, _ = _cuda_inputs(1)
    with pytest.raises(ValueError, match="BF16 projection math"):
        message_hidden_reduce(
            preactivation,
            weight,
            bias,
            edge_mask,
            48,
            backend="triton",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton", "triton_memory"])
def test_mpnn_message_triton_is_fullgraph_compilable(backend: str) -> None:
    preactivation, weight, bias, edge_mask, upstream = _cuda_inputs(8)

    def forward(x, w, b, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return message_hidden_reduce(x, w, b, mask, 48, backend=backend)

    compiled = torch.compile(forward, fullgraph=True)
    output = compiled(preactivation, weight, bias, edge_mask)
    output.backward(upstream)
    gradients = (preactivation.grad, weight.grad, bias.grad)
    assert [gradient.shape for gradient in gradients if gradient is not None] == [
        preactivation.shape,
        weight.shape,
        bias.shape,
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_message_inference_fusion_is_fullgraph_compilable() -> None:
    preactivation, weight, bias, edge_mask, _ = _cuda_inputs(17)

    def candidate(x, w, b, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return message_hidden_reduce(x, w, b, mask, 48, backend="triton")

    def reference(x, w, b, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return message_hidden_reduce_pytorch(x, w, b, mask, 48)

    compiled = torch.compile(candidate, fullgraph=True)
    with torch.no_grad():
        actual = compiled(preactivation, weight, bias, edge_mask)
        expected = reference(preactivation, weight, bias, edge_mask)

    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton", "triton_memory"])
def test_mpnn_message_backend_matches_full_model_gradients(backend: str) -> None:
    common = dict(
        node_width=128,
        edge_width=128,
        hidden_width=128,
        encoder_depth=1,
        decoder_depth=1,
        k_neighbors=48,
        coordinate_noise=0.0,
        dropout=0.0,
        block_linear_min_edges=0,
    )
    torch.manual_seed(19)
    reference = ProteinMPNN(
        ProteinMPNNConfig(
            **common,
            message_backend="pytorch",
            edge_mlp_backend="pytorch",
        )
    ).cuda()
    nn.init.normal_(reference.output_projection.weight, std=128**-0.5)
    candidate = ProteinMPNN(
        ProteinMPNNConfig(
            **common,
            message_backend=backend,
            edge_mlp_backend="pytorch",
        )
    ).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference.train()
    candidate.train()

    batch, length = 2, 64
    backbone = torch.randn(batch, length, 4, 3, device="cuda")
    sequence = torch.randint(0, 21, (batch, length), device="cuda")
    residue_mask = torch.ones(batch, length, device="cuda")
    residue_mask[1, 53:] = 0
    residue_index = torch.arange(length, device="cuda").expand(batch, -1)
    chain_index = torch.zeros(batch, length, dtype=torch.long, device="cuda")
    decoding_order = torch.stack(
        [torch.randperm(length, device="cuda") for _ in range(batch)]
    )
    patch_index = (torch.arange(length, device="cuda") // 8).expand(batch, -1)

    def run(model, coordinates):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(
                coordinates,
                sequence,
                residue_mask,
                residue_index,
                chain_index,
                decoding_order,
                patch_index,
            )

    reference_backbone = backbone.detach().clone().requires_grad_(True)
    candidate_backbone = backbone.detach().clone().requires_grad_(True)
    expected = run(reference, reference_backbone)
    actual = run(candidate, candidate_backbone)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        candidate_backbone.grad,
        reference_backbone.grad,
        atol=4e-2,
        rtol=4e-2,
    )
    expected_parameter_grad = torch.cat(
        [
            parameter.grad.detach().float().flatten()
            for parameter in reference.parameters()
        ]
    )
    actual_parameter_grad = torch.cat(
        [
            parameter.grad.detach().float().flatten()
            for parameter in candidate.parameters()
        ]
    )
    relative_error = torch.linalg.vector_norm(
        actual_parameter_grad - expected_parameter_grad
    ) / torch.linalg.vector_norm(expected_parameter_grad).clamp_min(1e-12)
    cosine = F.cosine_similarity(actual_parameter_grad, expected_parameter_grad, dim=0)
    assert relative_error < 0.02
    assert cosine > 0.999
