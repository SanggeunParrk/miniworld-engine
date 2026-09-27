from __future__ import annotations

import pytest
import torch

from miniworld_kernels.kernels.mpnn_edge_mlp import (
    edge_mlp_update,
    edge_mlp_update_pytorch,
)
from miniworld_kernels.kernels.mpnn_edge_mlp.interface import _select_backend
from miniworld_kernels.modules.mpnn.layers import EncoderLayer


def test_mpnn_edge_mlp_reference_contract_and_cpu_fallback() -> None:
    torch.manual_seed(11)
    preactivation = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    hidden_weight = torch.randn(7, 7, dtype=torch.float64, requires_grad=True)
    hidden_bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    output_weight = torch.randn(7, 7, dtype=torch.float64, requires_grad=True)
    output_bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    values = (
        preactivation,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )

    expected = edge_mlp_update_pytorch(*values)
    actual = edge_mlp_update(*values, backend="auto")

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mpnn_edge_mlp_auto_policy_is_shape_arch_and_grad_aware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = torch.empty(2048 * 48, 128, device="meta")
    small = torch.empty(2048 * 48 - 1, 128, device="meta")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 6))

    assert _select_backend(crop, "auto", supported=True) == "triton_compute"
    with torch.no_grad():
        assert _select_backend(crop, "auto", supported=True) == "triton_memory"
    assert _select_backend(small, "auto", supported=True) == "pytorch"
    assert _select_backend(crop, "auto", supported=False) == "pytorch"

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    assert _select_backend(crop, "auto", supported=True) == "pytorch"


def _cuda_inputs(rows: int):
    torch.manual_seed(29)
    preactivation = torch.randn(
        rows,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    parameters = []
    for shape in ((128, 128), (128,), (128, 128), (128,)):
        value = torch.randn(shape, device="cuda", dtype=torch.float32) / 128**0.5
        parameters.append(value.detach().requires_grad_(True))
    upstream = torch.randn_like(preactivation)
    return preactivation, *parameters, upstream


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.detach().float() - expected.detach().float()
    return float(difference.norm() / expected.detach().float().norm().clamp_min(1e-20))


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.detach().float().flatten(),
            expected.detach().float().flatten(),
            dim=0,
        )
    )


def _assert_tracks(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    relative: float,
    cosine: float = 0.99999,
) -> None:
    assert _relative_error(actual, expected) < relative
    if expected.detach().float().norm() > 1e-12:
        assert _cosine(actual, expected) > cosine
    else:
        assert actual.detach().float().norm() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton_compute", "triton_memory"])
@pytest.mark.parametrize("rows", [1, 17, 257])
def test_mpnn_edge_mlp_triton_matches_bf16_reference(
    rows: int,
    backend: str,
) -> None:
    actual_values = _cuda_inputs(rows)
    expected_values = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        for value in actual_values[:-1]
    )
    upstream = actual_values[-1]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = edge_mlp_update(*actual_values[:-1], backend=backend)
        expected = edge_mlp_update_pytorch(*expected_values)

    actual_gradients = torch.autograd.grad(
        actual,
        actual_values[:-1],
        upstream,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        expected_values,
        upstream,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_mlp_memory_recompute_drift_stays_bounded() -> None:
    """Pin how far backward's recompute may drift from the forward it replays.

    The fused forward contracts 128 channels in one ``tl.dot``; the recompute
    accumulates eight 16-wide chunks, so backward differentiates a very slightly
    different function. Matching the fused order exactly costs 11x on this launch
    (see ``_recompute_projected_op``), so the drift is accepted and bounded here
    instead: it must stay far below this policy's own 4.04e-5 forward error
    against the PyTorch reference.
    """
    from miniworld_kernels.kernels.mpnn_edge_mlp.triton.main import (
        _recompute_projected_op,
    )

    values = _cuda_inputs(2048 * 48)
    preactivation, hidden_weight, hidden_bias, output_weight, output_bias = (
        value.detach() for value in values[:-1]
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused = edge_mlp_update(
            preactivation,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            backend="triton_memory",
        )
    projected = _recompute_projected_op(preactivation, hidden_weight, hidden_bias)
    staged = _recompute_projected_op(projected, output_weight, output_bias)

    # Observed: 3.8e-5 on A5000/A6000 for the composed update.
    assert _relative_error(staged, fused) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton_compute", "triton_memory"])
def test_mpnn_edge_mlp_rank_one_backward_preserves_bias_shapes(backend: str) -> None:
    values = _cuda_inputs(1)
    rank_one_values = (values[0][0], *values[1:-1])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = edge_mlp_update(*rank_one_values, backend=backend)

    gradients = torch.autograd.grad(output, rank_one_values, values[-1][0])

    assert output.shape == (128,)
    assert gradients[2].shape == (128,)
    assert gradients[4].shape == (128,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_mlp_crop_2048_gradient_quality() -> None:
    actual_values = _cuda_inputs(2048 * 48)
    expected_values = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        for value in actual_values[:-1]
    )
    upstream = actual_values[-1]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = edge_mlp_update(*actual_values[:-1], backend="triton_memory")
        expected = edge_mlp_update_pytorch(*expected_values)
    actual_gradients = torch.autograd.grad(actual, actual_values[:-1], upstream)
    expected_gradients = torch.autograd.grad(expected, expected_values, upstream)

    _assert_tracks(actual, expected, relative=1e-4)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        _assert_tracks(actual_gradient, expected_gradient, relative=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton_compute", "triton_memory"])
def test_mpnn_encoder_edge_backend_matches_pytorch(backend: str) -> None:
    torch.manual_seed(41)
    reference = EncoderLayer(
        128,
        128,
        128,
        0.0,
        48,
        "pytorch",
        "pytorch",
    ).cuda()
    candidate = EncoderLayer(
        128,
        128,
        128,
        0.0,
        48,
        "pytorch",
        "pytorch",
    ).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    candidate.edge_message.edge_mlp_backend = backend

    batch, length, neighbors = 1, 64, 48
    reference_inputs = (
        torch.randn(
            batch,
            length,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            batch,
            length,
            neighbors,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
    )
    candidate_inputs = tuple(
        value.detach().clone().requires_grad_() for value in reference_inputs
    )
    neighbor_indices = torch.randint(
        length,
        (batch, length, neighbors),
        device="cuda",
    )
    residue_mask = torch.ones(batch, length, device="cuda")
    neighbor_mask = torch.ones(batch, length, neighbors, device="cuda")

    def run(layer: EncoderLayer, values: tuple[torch.Tensor, torch.Tensor]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return layer(
                *values,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
            )

    compiled_reference = torch.compile(
        lambda *values: run(reference, values),
        fullgraph=True,
    )
    compiled_candidate = torch.compile(
        lambda *values: run(candidate, values),
        fullgraph=True,
    )
    expected = compiled_reference(*reference_inputs)
    actual = compiled_candidate(*candidate_inputs)
    # An FP32 run of the same layer is the only fair arbiter here. Comparing the
    # two compiled BF16 graphs against each other is not a kernel property: the
    # eager Triton and eager PyTorch policies are bitwise identical at this
    # shape, while Inductor's own fusion of the reference moves it by more than
    # 1e-3. So require the Triton policy to be no less accurate than the
    # reference policy with respect to FP32, not identical to it.
    truth_layer = EncoderLayer(128, 128, 128, 0.0, 48, "pytorch", "pytorch").cuda()
    truth_layer.load_state_dict(reference.state_dict(), strict=True)
    truth_inputs = tuple(
        value.detach().float().requires_grad_() for value in reference_inputs
    )
    truth = truth_layer(
        *truth_inputs,
        neighbor_indices,
        residue_mask,
        neighbor_mask,
    )

    upstream = tuple(torch.randn_like(value, dtype=torch.float32) for value in expected)
    torch.autograd.backward(truth, upstream)
    torch.autograd.backward(
        expected,
        tuple(
            value.to(output.dtype)
            for value, output in zip(upstream, expected, strict=True)
        ),
    )
    torch.autograd.backward(
        actual,
        tuple(
            value.to(output.dtype)
            for value, output in zip(upstream, actual, strict=True)
        ),
    )

    comparisons: list[tuple[str, float, float]] = []
    zero_gradient_failures: list[str] = []

    def record(
        actual_value: torch.Tensor,
        expected_value: torch.Tensor,
        truth_value: torch.Tensor,
        label: str,
    ) -> None:
        if truth_value.float().norm() <= 1e-12:
            # Production zero-initializes the transition output projection, so
            # its expand projection legitimately receives no gradient here, and a
            # relative error or direction is undefined for a zero vector.
            if actual_value.float().norm() != 0:
                zero_gradient_failures.append(label)
            return
        comparisons.append(
            (
                label,
                _relative_error(actual_value, truth_value),
                _relative_error(expected_value, truth_value),
            )
        )
        assert _cosine(actual_value, truth_value) > 0.9999, label

    for index, (actual_output, expected_output, truth_output) in enumerate(
        zip(actual, expected, truth, strict=True)
    ):
        record(actual_output, expected_output, truth_output, f"output[{index}]")
    for index, (actual_input, expected_input, truth_input) in enumerate(
        zip(candidate_inputs, reference_inputs, truth_inputs, strict=True)
    ):
        record(
            actual_input.grad,
            expected_input.grad,
            truth_input.grad,
            f"input_grad[{index}]",
        )
    for (name, actual_parameter), expected_parameter, truth_parameter in zip(
        candidate.named_parameters(),
        reference.parameters(),
        truth_layer.parameters(),
        strict=True,
    ):
        record(
            actual_parameter.grad,
            expected_parameter.grad,
            truth_parameter.grad,
            f"param_grad[{name}]",
        )

    report = "\n".join(
        f"  {label}: triton={triton_error:.3e} pytorch={pytorch_error:.3e} "
        f"ratio={triton_error / max(pytorch_error, 1e-20):.2f}"
        for label, triton_error, pytorch_error in comparisons
    )
    assert not zero_gradient_failures, (
        f"nonzero gradient where FP32 gives exactly zero: {zero_gradient_failures}"
    )
    # The absolute ceiling catches both policies degrading together. The ratio
    # bound is deliberately not tighter than the spread between two
    # independently compiled BF16 graphs: Inductor fuses the reference's own
    # reductions differently, so even quantities the edge MLP never touches (for
    # example the node message's output bias) land 20-30% apart from each other
    # relative to FP32. Bias gradients here are sums of 3072 rows at the BF16
    # accumulation floor, around 4e-3 to 6e-3 for both policies.
    for label, triton_error, pytorch_error in comparisons:
        assert triton_error < 2e-2, f"{label} exceeds the BF16 ceiling\n{report}"
        assert triton_error <= pytorch_error * 1.5 + 1e-3, (
            f"{label} is materially worse than the reference policy\n{report}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_w1_checkpoint_preserves_layer_outputs_and_gradients() -> None:
    torch.manual_seed(53)
    reference = EncoderLayer(
        128,
        128,
        128,
        0.0,
        48,
        "pytorch",
        "triton_memory",
        "off",
    ).cuda()
    candidate = EncoderLayer(
        128,
        128,
        128,
        0.0,
        48,
        "pytorch",
        "triton_memory",
        "checkpoint",
    ).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)

    batch, length, neighbors = 1, 64, 48
    reference_inputs = (
        torch.randn(batch, length, 128, device="cuda", requires_grad=True),
        torch.randn(
            batch,
            length,
            neighbors,
            128,
            device="cuda",
            requires_grad=True,
        ),
    )
    candidate_inputs = tuple(
        value.detach().clone().requires_grad_() for value in reference_inputs
    )
    neighbor_indices = torch.randint(
        length,
        (batch, length, neighbors),
        device="cuda",
    )
    residue_mask = torch.ones(batch, length, device="cuda")
    neighbor_mask = torch.ones(batch, length, neighbors, device="cuda")

    def run(
        layer: EncoderLayer,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return layer(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
            )

    compiled_reference = torch.compile(
        lambda node_states, edge_states: run(
            reference,
            node_states,
            edge_states,
        ),
        fullgraph=True,
    )
    compiled_candidate = torch.compile(
        lambda node_states, edge_states: run(
            candidate,
            node_states,
            edge_states,
        ),
        fullgraph=True,
    )
    expected = compiled_reference(*reference_inputs)
    actual = compiled_candidate(*candidate_inputs)
    upstream = tuple(torch.randn_like(value) for value in expected)
    torch.autograd.backward(expected, upstream)
    torch.autograd.backward(actual, upstream)

    # The forward is bitwise unchanged: the checkpoint replays the same math.
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)

    # Gradients are not. The checkpoint casts its operands to BF16 explicitly,
    # while the ordinary path hands FP32 to an autocast ``F.linear`` and lets it
    # cast internally. The forward values agree, but the checkpoint's gradient
    # path carries one extra BF16 cast node, so everything downstream of the edge
    # update -- that is, the node-side gradients -- differs at the BF16 floor.
    # Measured in eager on an A5000/A6000: 3.3e-3 on the node-state gradient and
    # up to 5.3e-3 on node parameters, with the edge-message gradients (the ones
    # this policy actually rewrites) bitwise identical. An earlier 1e-4 bound only
    # held because Inductor happened to emit identical code for both graphs.
    # The bitwise claim belongs in eager, where it is a property of this policy
    # rather than of Inductor's scheduling: comparing two separately compiled
    # graphs bitwise measures the compiler, as the integration test above also
    # found. Eager is checked separately below.
    for actual_parameter, expected_parameter in zip(
        candidate.parameters(),
        reference.parameters(),
        strict=True,
    ):
        _assert_tracks(
            actual_parameter.grad,
            expected_parameter.grad,
            relative=1e-2,
            cosine=0.9999,
        )
    for actual_input, expected_input in zip(
        candidate_inputs,
        reference_inputs,
        strict=True,
    ):
        _assert_tracks(
            actual_input.grad,
            expected_input.grad,
            relative=1e-2,
            cosine=0.9999,
        )

    # Eager is where the policy's own guarantee is testable: every gradient this
    # boundary rewrites must be bitwise unchanged, and only the node side -- which
    # sees one extra BF16 cast on its gradient path -- may move.
    eager_reference = EncoderLayer(128, 128, 128, 0.0, 48, "pytorch", "triton_memory")
    eager_candidate = EncoderLayer(
        128, 128, 128, 0.0, 48, "pytorch", "triton_memory", "checkpoint"
    )
    eager_candidate.load_state_dict(eager_reference.state_dict(), strict=True)
    eager_reference, eager_candidate = eager_reference.cuda(), eager_candidate.cuda()
    eager_inputs = tuple(
        value.detach().clone().requires_grad_() for value in reference_inputs
    )
    eager_candidate_inputs = tuple(
        value.detach().clone().requires_grad_() for value in reference_inputs
    )
    eager_expected = run(eager_reference, *eager_inputs)
    eager_actual = run(eager_candidate, *eager_candidate_inputs)
    eager_upstream = tuple(torch.randn_like(value) for value in eager_expected)
    torch.autograd.backward(eager_expected, eager_upstream)
    torch.autograd.backward(eager_actual, eager_upstream)
    for (name, actual_parameter), expected_parameter in zip(
        eager_candidate.named_parameters(),
        eager_reference.parameters(),
        strict=True,
    ):
        if name.startswith("edge_message."):
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                atol=0,
                rtol=0,
                msg=f"eager: {name} must be bitwise unchanged by the checkpoint",
            )


def _small_edge_w1_case(
    *,
    dropout: float,
    recompute: str,
) -> EncoderLayer:
    return EncoderLayer(
        node_width=128,
        edge_width=128,
        hidden_width=128,
        dropout=dropout,
        neighbor_scale=16,
        reduction_backend="pytorch",
        edge_mlp_backend="triton_memory",
        edge_w1_recompute=recompute,
    ).cuda()


def _small_edge_w1_inputs() -> tuple[torch.Tensor, ...]:
    batch, length, neighbors = 1, 32, 16
    return (
        torch.randn(batch, length, 128, device="cuda", requires_grad=True),
        torch.randn(
            batch,
            length,
            neighbors,
            128,
            device="cuda",
            requires_grad=True,
        ),
        torch.randint(length, (batch, length, neighbors), device="cuda"),
        torch.ones(batch, length, device="cuda"),
        torch.ones(batch, length, neighbors, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_w1_checkpoint_preserves_dropout_rng_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(59)
    reference = _small_edge_w1_case(dropout=0.1, recompute="off").train()
    candidate = _small_edge_w1_case(dropout=0.1, recompute="checkpoint").train()
    candidate.load_state_dict(reference.state_dict(), strict=True)

    reference_values = _small_edge_w1_inputs()
    candidate_values = (
        reference_values[0].detach().clone().requires_grad_(),
        reference_values[1].detach().clone().requires_grad_(),
        *reference_values[2:],
    )
    upstream = (
        torch.randn(1, 32, 128, device="cuda"),
        torch.randn(1, 32, 16, 128, device="cuda"),
    )

    original_checkpoint = torch.utils.checkpoint.checkpoint
    checkpoint_calls = 0

    def counted_checkpoint(*args, **kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counted_checkpoint)

    def forward(layer: EncoderLayer, values: tuple[torch.Tensor, ...]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return layer(*values)

    torch.cuda.manual_seed_all(997)
    expected = forward(reference, reference_values)
    expected_forward_rng = torch.cuda.get_rng_state().clone()
    torch.autograd.backward(expected, upstream)
    expected_backward_rng = torch.cuda.get_rng_state().clone()

    torch.cuda.manual_seed_all(997)
    actual = forward(candidate, candidate_values)
    actual_forward_rng = torch.cuda.get_rng_state().clone()
    torch.autograd.backward(actual, upstream)
    actual_backward_rng = torch.cuda.get_rng_state().clone()

    assert checkpoint_calls == 1
    assert torch.equal(actual_forward_rng, expected_forward_rng)
    assert torch.equal(actual_backward_rng, expected_backward_rng)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    for actual_input, expected_input in zip(
        candidate_values[:2],
        reference_values[:2],
        strict=True,
    ):
        _assert_tracks(
            actual_input.grad,
            expected_input.grad,
            relative=3e-3,
            cosine=0.99999,
        )
    # Recompute intentionally stores the W1 inputs in BF16. Gradients flowing
    # back into the preceding node stack therefore cross one BF16 rounding
    # boundary; this small shape measures a worst relative error just below 0.5%.
    for actual_parameter, expected_parameter in zip(
        candidate.parameters(),
        reference.parameters(),
        strict=True,
    ):
        _assert_tracks(
            actual_parameter.grad,
            expected_parameter.grad,
            relative=6e-3,
            cosine=0.99998,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_w1_checkpoint_is_bypassed_under_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(61)
    reference = _small_edge_w1_case(dropout=0.1, recompute="off").eval()
    candidate = _small_edge_w1_case(dropout=0.1, recompute="checkpoint").eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    values = _small_edge_w1_inputs()

    def unexpected_checkpoint(*_args, **_kwargs):
        pytest.fail("edge-W1 checkpoint must not run with gradients disabled")

    monkeypatch.setattr(
        torch.utils.checkpoint,
        "checkpoint",
        unexpected_checkpoint,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(*values)
        actual = candidate(*values)

    for actual_output, expected_output in zip(actual, expected, strict=True):
        assert not actual_output.requires_grad
        torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("backend", "expected_edge_tensors"),
    [("triton_compute", 2), ("triton_memory", 1)],
)
def test_mpnn_edge_mlp_forward_saved_tensor_policy(
    backend: str,
    expected_edge_tensors: int,
) -> None:
    values = _cuda_inputs(17)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    def unpack(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = edge_mlp_update(*values[:-1], backend=backend)

    edge_sized = [tensor for tensor in saved if tensor.numel() == values[0].numel()]
    assert output.shape == values[0].shape
    assert len(edge_sized) == expected_edge_tensors
    assert any(tensor.data_ptr() == values[0].data_ptr() for tensor in edge_sized)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton_compute", "triton_memory"])
def test_mpnn_edge_mlp_triton_is_fullgraph_compilable(backend: str) -> None:
    values = _cuda_inputs(257)

    def forward(*inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return edge_mlp_update(*inputs, backend=backend)

    compiled = torch.compile(forward, fullgraph=True)
    output = compiled(*values[:-1])
    output.backward(values[-1])

    assert output.shape == values[0].shape
    assert all(value.grad is not None for value in values[:-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_mlp_auto_crop_policy_is_fullgraph_compilable() -> None:
    if (
        torch.cuda.get_device_capability() != (8, 6)
        or torch.cuda.get_device_name() != "NVIDIA RTX A5000"
    ):
        pytest.skip("the automatic crop policy is calibrated on the A5000")
    values = _cuda_inputs(2048 * 48)

    def forward(*function_inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return edge_mlp_update(*function_inputs, backend="auto")

    compiled = torch.compile(forward, fullgraph=True)
    output = compiled(*values[:-1])
    output.backward(values[-1])
    torch.cuda.synchronize()

    assert output.shape == values[0].shape
    assert all(value.grad is not None for value in values[:-1])
    assert all(torch.isfinite(value.grad).all() for value in values[:-1])
