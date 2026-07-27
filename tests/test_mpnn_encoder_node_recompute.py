from __future__ import annotations

import copy
import warnings
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig
from miniworld_kernels.modules.mpnn.layers import EncoderLayer


def _model_inputs(
    *,
    batch: int,
    length: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    backbone = torch.randn(batch, length, 4, 3, device=device)
    sequence = torch.randint(0, 21, (batch, length), device=device)
    residue_mask = torch.ones(batch, length, device=device)
    residue_index = torch.arange(length, device=device).expand(batch, -1)
    chain_index = torch.zeros(batch, length, dtype=torch.long, device=device)
    decoding_order = torch.stack(
        [torch.randperm(length, device=device) for _ in range(batch)]
    )
    patch_index = (torch.arange(length, device=device) // 8).expand(batch, -1)
    return (
        backbone,
        sequence,
        residue_mask,
        residue_index,
        chain_index,
        decoding_order,
        patch_index,
    )


@pytest.mark.parametrize(
    "execution_path",
    [
        "block",
        "dense_training",
        "zero_node_training",
        "node_recompute",
        "zero_node_recompute",
    ],
)
def test_encoder_execution_paths_preserve_module_hooks_on_cpu(
    execution_path: str,
) -> None:
    layer = EncoderLayer(
        8,
        8,
        8,
        0.0,
        4,
        reduction_backend="pytorch",
        edge_mlp_backend="pytorch",
    )
    node_states = torch.randn(1, 4, 8, requires_grad=True)
    edge_states = torch.randn(1, 4, 4, 8, requires_grad=True)
    neighbor_indices = torch.randint(0, 4, (1, 4, 4))
    residue_mask = torch.ones(1, 4)
    neighbor_mask = torch.ones(1, 4, 4)
    calls = {"pre": 0, "forward": 0}

    def pre_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        calls["pre"] += 1

    def forward_hook(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        calls["forward"] += 1

    handles = (
        layer.register_forward_pre_hook(pre_hook),
        layer.register_forward_hook(forward_hook),
    )
    try:
        # The recompute paths cannot engage on CPU and warn about it; the module
        # boundary under test here is unaffected either way.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            layer(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
                execution_path=execution_path,  # type: ignore[arg-type]
            )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"pre": 1, "forward": 1}


def test_encoder_node_w1_policy_falls_back_and_only_dispatches_for_training_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProteinMPNNConfig(
        node_width=8,
        edge_width=8,
        hidden_width=8,
        encoder_depth=1,
        decoder_depth=1,
        k_neighbors=4,
        coordinate_noise=0.0,
        dropout=0.0,
        block_linear_min_edges=0,
        message_backend="triton_memory",
        encoder_node_w1_recompute="checkpoint",
    )
    candidate = ProteinMPNN(config)
    baseline = ProteinMPNN(replace(config, encoder_node_w1_recompute="off"))
    baseline.load_state_dict(candidate.state_dict(), strict=True)
    # CPU is outside the fused contract. Keep the ordinary fallback executable
    # while retaining the stack-level explicit policy under test.
    for model in (candidate, baseline):
        for layer in (*model.encoder.layers, *model.decoder.layers):
            layer.node_message.reduction_backend = "pytorch"
    values = _model_inputs(batch=1, length=4, device="cpu")

    calls = 0
    layer = candidate.encoder.layers[0]
    original = layer.forward_zero_node_training_recompute

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(layer, "forward_zero_node_training_recompute", tracked)

    candidate.train()
    baseline.train()
    # CPU cannot satisfy the fused contract, so the requested policy falls back
    # and has to say so once.
    with pytest.warns(RuntimeWarning, match="encoder_node_w1_recompute"):
        actual = candidate(*values)
    expected = baseline(*values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert calls == 1

    candidate.eval()
    candidate(*values)
    assert calls == 1  # eval + grad keeps the ordinary inference path

    candidate.train()
    with torch.no_grad():
        candidate(*values)
    assert calls == 1

    candidate(*values, checkpoint_layers=True)
    assert calls == 1  # whole-layer checkpointing bypasses the inner boundary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_encoder_node_w1_checkpoint_preserves_layer_hooks() -> None:
    config = ProteinMPNNConfig(
        node_width=128,
        edge_width=128,
        hidden_width=128,
        encoder_depth=3,
        decoder_depth=1,
        k_neighbors=48,
        coordinate_noise=0.0,
        dropout=0.0,
        block_linear_min_edges=0,
        message_backend="triton_memory",
        edge_mlp_backend="pytorch",
        feature_backend="recompute",
        encoder_node_w1_recompute="checkpoint",
    )
    model = ProteinMPNN(config).cuda().train()
    nn.init.normal_(model.output_projection.weight, std=128**-0.5)
    values = _model_inputs(batch=1, length=64, device="cuda")
    calls = {
        "pre": [0] * config.encoder_depth,
        "forward": [0] * config.encoder_depth,
        "backward": [0] * config.encoder_depth,
    }

    def increment(kind: str, index: int):
        def hook(*_args) -> None:
            calls[kind][index] += 1

        return hook

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for index, layer in enumerate(model.encoder.layers):
        handles.extend(
            [
                layer.register_forward_pre_hook(increment("pre", index)),
                layer.register_forward_hook(increment("forward", index)),
                layer.register_full_backward_hook(increment("backward", index)),
            ]
        )
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(*values)
        output.float().square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {
        "pre": [1] * config.encoder_depth,
        "forward": [1] * config.encoder_depth,
        "backward": [1] * config.encoder_depth,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fallback", ["message_backend", "packed_width"])
def test_encoder_node_w1_checkpoint_gpu_contract_fallback(
    fallback: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_width = 64 if fallback == "packed_width" else 128
    backend = "triton_compute" if fallback == "message_backend" else "triton_memory"
    torch.manual_seed(7)
    layer = EncoderLayer(
        node_width,
        128,
        128,
        0.0,
        48,
        reduction_backend=backend,
        edge_mlp_backend="pytorch",
    ).cuda()
    reference = copy.deepcopy(layer)

    node_states = torch.randn(1, 8, node_width, device="cuda", requires_grad=True)
    edge_states = torch.randn(1, 8, 48, 128, device="cuda", requires_grad=True)
    neighbor_indices = torch.randint(0, 8, (1, 8, 48), device="cuda")
    residue_mask = torch.ones(1, 8, device="cuda")
    neighbor_mask = torch.ones(1, 8, 48, device="cuda")

    def unexpected_checkpoint(*args, **kwargs):
        raise AssertionError("unsupported contracts must use the ordinary path")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", unexpected_checkpoint)
    # The fallback must be audible: a silent no-op would let a configured memory
    # policy be benchmarked as if it had engaged.
    with pytest.warns(RuntimeWarning, match="encoder_node_w1_recompute"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
            )
            actual = layer.forward_node_recompute(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
            )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_encoder_node_w1_checkpoint_fullgraph_parity() -> None:
    common = dict(
        node_width=128,
        edge_width=128,
        hidden_width=128,
        encoder_depth=3,
        decoder_depth=1,
        k_neighbors=48,
        coordinate_noise=0.0,
        dropout=0.0,
        block_linear_min_edges=0,
        message_backend="triton_memory",
        edge_mlp_backend="triton_memory",
        edge_norm_backend="memory",
        feature_backend="recompute",
        edge_w1_recompute="checkpoint",
    )
    torch.manual_seed(11)
    baseline = ProteinMPNN(
        ProteinMPNNConfig(**common, encoder_node_w1_recompute="off")
    ).cuda()
    nn.init.normal_(baseline.output_projection.weight, std=128**-0.5)
    candidate = ProteinMPNN(
        ProteinMPNNConfig(**common, encoder_node_w1_recompute="checkpoint")
    ).cuda()
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.train()
    candidate.train()

    values = _model_inputs(batch=1, length=64, device="cuda")
    baseline_fn = torch.compile(baseline, fullgraph=True)
    candidate_fn = torch.compile(candidate, fullgraph=True)

    def run(model: nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(*values)
        output.float().square().mean().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return output.detach(), gradients

    # Warm both separately so parity exercises stable compiled full graphs,
    # including the checkpoint backward replay.
    run(baseline_fn)
    run(candidate_fn)
    expected_output, expected_gradients = run(baseline_fn)
    actual_output, actual_gradients = run(candidate_fn)

    torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    assert actual_gradients.keys() == expected_gradients.keys()
    expected_flat = torch.cat(
        [gradient.float().flatten() for gradient in expected_gradients.values()]
    )
    actual_flat = torch.cat(
        [gradient.float().flatten() for gradient in actual_gradients.values()]
    )
    relative_error = torch.linalg.vector_norm(actual_flat - expected_flat) / (
        torch.linalg.vector_norm(expected_flat).clamp_min(1e-12)
    )
    assert relative_error < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_encoder_node_w1_checkpoint_drops_w1_activation_from_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    baseline = EncoderLayer(
        128,
        128,
        128,
        0.0,
        48,
        reduction_backend="triton_memory",
        edge_mlp_backend="pytorch",
    ).cuda()
    candidate = copy.deepcopy(baseline)
    node_states = torch.zeros(1, 8, 128, device="cuda", requires_grad=True)
    edge_states = torch.randn(1, 8, 48, 128, device="cuda", requires_grad=True)
    neighbor_indices = torch.randint(0, 8, (1, 8, 48), device="cuda")
    residue_mask = torch.ones(1, 8, device="cuda")
    neighbor_mask = torch.ones(1, 8, 48, device="cuda")

    def w1_is_saved(layer: EncoderLayer, *, recompute: bool) -> bool:
        projection = layer.node_message.input_projection
        original = projection.edge_only
        preactivations: list[torch.Tensor] = []
        saved: list[torch.Tensor] = []

        def capture(edges: torch.Tensor) -> torch.Tensor:
            output = original(edges)
            preactivations.append(output)
            return output

        monkeypatch.setattr(projection, "edge_only", capture)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forward = (
                    layer.forward_zero_node_training_recompute
                    if recompute
                    else layer.forward_zero_node_training
                )
                forward(
                    node_states,
                    edge_states,
                    neighbor_indices,
                    residue_mask,
                    neighbor_mask,
                )

        assert len(preactivations) == 1
        w1_storage = preactivations[0].untyped_storage().data_ptr()
        return any(
            tensor.untyped_storage().data_ptr() == w1_storage for tensor in saved
        )

    assert w1_is_saved(baseline, recompute=False)
    assert not w1_is_saved(candidate, recompute=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_encoder_node_w1_checkpoint_keeps_dropout_rng_outside_replay() -> None:
    torch.manual_seed(31)
    baseline = EncoderLayer(
        128,
        128,
        128,
        0.1,
        48,
        reduction_backend="triton_memory",
        edge_mlp_backend="pytorch",
    ).cuda()
    candidate = copy.deepcopy(baseline)
    node_values = torch.zeros(1, 8, 128, device="cuda")
    edge_values = torch.randn(1, 8, 48, 128, device="cuda")
    neighbor_indices = torch.randint(0, 8, (1, 8, 48), device="cuda")
    residue_mask = torch.ones(1, 8, device="cuda")
    neighbor_mask = torch.ones(1, 8, 48, device="cuda")
    node_upstream = torch.randn_like(node_values)
    edge_upstream = torch.randn_like(edge_values)

    def run(
        layer: EncoderLayer,
        *,
        recompute: bool,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        layer.zero_grad(set_to_none=True)
        nodes = node_values.detach().clone().requires_grad_(True)
        edges = edge_values.detach().clone().requires_grad_(True)
        torch.cuda.manual_seed(991)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            forward = (
                layer.forward_zero_node_training_recompute
                if recompute
                else layer.forward_zero_node_training
            )
            output = forward(
                nodes,
                edges,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
            )
        forward_rng = torch.cuda.get_rng_state()
        torch.autograd.backward(output, (node_upstream, edge_upstream))
        backward_rng = torch.cuda.get_rng_state()
        gradients = (
            nodes.grad.detach().clone(),
            edges.grad.detach().clone(),
            *(
                parameter.grad.detach().clone()
                for parameter in layer.parameters()
                if parameter.grad is not None
            ),
        )
        return (*output, *gradients), forward_rng, backward_rng

    expected, expected_forward_rng, expected_backward_rng = run(
        baseline,
        recompute=False,
    )
    actual, actual_forward_rng, actual_backward_rng = run(
        candidate,
        recompute=True,
    )

    torch.testing.assert_close(actual_forward_rng, expected_forward_rng, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_backward_rng,
        expected_backward_rng,
        atol=0,
        rtol=0,
    )
    for actual_output, expected_output in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    expected_gradients = torch.cat(
        [gradient.float().flatten() for gradient in expected[2:]]
    )
    actual_gradients = torch.cat(
        [gradient.float().flatten() for gradient in actual[2:]]
    )
    relative_error = torch.linalg.vector_norm(
        actual_gradients - expected_gradients
    ) / torch.linalg.vector_norm(expected_gradients).clamp_min(1e-12)
    assert relative_error < 2e-3
