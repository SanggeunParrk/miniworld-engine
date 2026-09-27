"""Numerical checks for the semantic production PyTorch MPNN path."""

from __future__ import annotations

import pytest
import torch

from miniworld_kernels.modules.mpnn import (
    CSSBForwardAdapter,
    EncodedMPNN,
    NaiveProteinMPNN,
    NeighborGraph,
    ProteinMPNN,
    ProteinMPNNConfig,
    convert_cssb_state_dict,
    iter_reference_parameter_pairs,
    load_cssb_weights,
    production_tensor_in_reference_layout,
)
from miniworld_kernels.modules.mpnn.naive import (
    DecLayer as NaiveDecLayer,
    EncLayer as NaiveEncLayer,
)
from miniworld_kernels.modules.mpnn._functional import (
    concatenate_neighbor_features,
    gather_neighbors,
)
from miniworld_kernels.modules.mpnn.conversion import (
    convert_decoder_layer_state_dict,
    convert_encoder_layer_state_dict,
)
from miniworld_kernels.modules.mpnn.layers import (
    DecoderLayer,
    EncoderLayer,
)


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().float().flatten()
    expected = expected.detach().float().flatten()
    denominator = (actual.norm() * expected.norm()).clamp_min(1e-30)
    return actual.dot(expected).div(denominator).item()


def _relative_frobenius(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.detach().float() - expected.detach().float()
    return (
        difference.norm().div(expected.detach().float().norm().clamp_min(1e-30)).item()
    )


def _models(
    device: str = "cpu", *, block_linear_min_edges: int = 49_152
) -> tuple[NaiveProteinMPNN, ProteinMPNN]:
    kwargs = dict(
        node_features=16,
        edge_features=16,
        hidden_dim=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
        augment_trans=0,
        augment_rot=0,
        dropout=0,
    )
    reference = NaiveProteinMPNN(**kwargs).eval()
    generator = torch.Generator().manual_seed(2026)
    with torch.no_grad():
        for parameter in reference.parameters():
            values = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )
            parameter.copy_(values * 0.05)
    optimized = ProteinMPNN(
        ProteinMPNNConfig(
            node_width=16,
            edge_width=16,
            hidden_width=16,
            encoder_depth=1,
            decoder_depth=1,
            k_neighbors=4,
            coordinate_noise=0,
            dropout=0,
            block_linear_min_edges=block_linear_min_edges,
        )
    ).eval()
    load_cssb_weights(optimized, reference)
    return reference.to(device), optimized.to(device)


def _randomize_parameters(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _assert_parameter_gradients_match(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    atol: float = 2e-11,
    rtol: float = 2e-11,
) -> None:
    for (
        expected_name,
        expected_parameter,
        _actual_name,
        actual_parameter,
    ) in iter_reference_parameter_pairs(expected, actual):
        torch.testing.assert_close(
            production_tensor_in_reference_layout(expected_name, actual_parameter.grad),
            expected_parameter.grad,
            atol=atol,
            rtol=rtol,
        )


def _assert_layer_parameter_gradients_match(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    converter,
    *,
    atol: float = 2e-11,
    rtol: float = 2e-11,
) -> None:
    expected_gradients = {
        name: parameter.grad for name, parameter in expected.named_parameters()
    }
    converted = converter(expected_gradients, target=dict(actual.named_parameters()))
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter.grad, converted[name], atol=atol, rtol=rtol
        )


def _assert_matching_parameter_gradients(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    atol: float = 0,
    rtol: float = 0,
) -> None:
    for (actual_name, actual_parameter), (
        expected_name,
        expected_parameter,
    ) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=atol,
            rtol=rtol,
        )


def _inputs(device: str = "cpu", length: int = 12) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    xyz = torch.randn(1, length, 4, 3, generator=generator).to(device)
    seq = torch.randint(0, 21, (1, length), generator=generator).to(device)
    mask = torch.ones(1, length, device=device)
    residue_idx = torch.arange(length, device=device).unsqueeze(0)
    chain_idx = torch.zeros(1, length, dtype=torch.long, device=device)
    chain_idx[:, length // 2 :] = 1
    decoding_order = torch.randperm(length, generator=generator).to(device).unsqueeze(0)
    patch_index = torch.arange(length, device=device).unsqueeze(0)
    return (
        xyz,
        seq,
        mask,
        residue_idx,
        chain_idx,
        decoding_order,
        patch_index,
        mask,
        torch.tensor([length], device=device),
    )


def _production_args(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (*inputs[:7], inputs[8])


def _forward(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    backbone: torch.Tensor | None = None,
    checkpoint_layers: bool = False,
    return_log_prob: bool = False,
) -> torch.Tensor:
    values = list(inputs)
    if backbone is not None:
        values[0] = backbone
    if isinstance(model, NaiveProteinMPNN):
        return model(
            *values,
            use_checkpoint=checkpoint_layers,
            return_log_prob=return_log_prob,
        )
    return model(
        *_production_args(tuple(values)),
        checkpoint_layers=checkpoint_layers,
        return_log_prob=return_log_prob,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimized_features_match_naive(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    reference, optimized = _models(device)
    inputs = _inputs(device)
    with torch.no_grad():
        expected, expected_idx = reference.features(
            inputs[0], inputs[2], inputs[3], inputs[4], inputs[8]
        )
        actual, actual_idx = optimized.backbone_features(
            inputs[0], inputs[2], inputs[3], inputs[4], inputs[8]
        )
    torch.testing.assert_close(actual_idx, expected_idx, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_backbone_features_expose_explicit_neighbor_graph_contract() -> None:
    _, model = _models()
    inputs = list(_inputs(length=7))
    inputs[2][:, -2:] = 0

    with torch.no_grad():
        graph = model.backbone_features.build_graph(
            inputs[0], inputs[2], inputs[3], inputs[4], inputs[8]
        )
        edge_features, neighbor_indices = model.backbone_features(
            inputs[0], inputs[2], inputs[3], inputs[4], inputs[8]
        )

    assert isinstance(graph, NeighborGraph)
    torch.testing.assert_close(graph.edge_features, edge_features, atol=0, rtol=0)
    torch.testing.assert_close(graph.neighbor_indices, neighbor_indices, atol=0, rtol=0)
    gathered_mask = gather_neighbors(
        inputs[2].unsqueeze(-1), graph.neighbor_indices
    ).squeeze(-1)
    expected_edge_mask = inputs[2].unsqueeze(-1) * gathered_mask
    torch.testing.assert_close(graph.edge_mask, expected_edge_mask, atol=0, rtol=0)
    assert (
        torch.count_nonzero(
            graph.edge_features.masked_select(~graph.edge_mask.bool().unsqueeze(-1))
        )
        == 0
    )


@pytest.mark.parametrize("coordinate_grad", [False, True])
def test_feature_recompute_backend_preserves_outputs_and_gradients(
    coordinate_grad: bool,
) -> None:
    _, reference = _models()
    _, candidate = _models()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference.backbone_features.feature_backend = "pytorch"
    candidate.backbone_features.feature_backend = "recompute"
    common = _inputs(length=8)
    reference_backbone = common[0].detach().clone().requires_grad_(coordinate_grad)
    candidate_backbone = common[0].detach().clone().requires_grad_(coordinate_grad)

    expected = _forward(reference, common, backbone=reference_backbone)
    actual = _forward(candidate, common, backbone=candidate_backbone)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    _assert_matching_parameter_gradients(candidate, reference, atol=2e-7, rtol=2e-7)
    if coordinate_grad:
        torch.testing.assert_close(
            candidate_backbone.grad,
            reference_backbone.grad,
            atol=2e-7,
            rtol=2e-7,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_feature_recompute_backend_is_fullgraph_compilable() -> None:
    _, model = _models("cuda")
    model.backbone_features.feature_backend = "recompute"
    common = _inputs("cuda", length=8)
    backbone = common[0].detach().clone().requires_grad_(True)

    def forward(coordinates: torch.Tensor) -> torch.Tensor:
        return _forward(model, common, backbone=coordinates)

    compiled = torch.compile(forward, fullgraph=True)
    output = compiled(backbone)
    output.square().mean().backward()
    assert backbone.grad is not None
    assert torch.isfinite(backbone.grad).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimized_forward_and_backward_match_naive(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    reference, optimized = _models(device)
    common = _inputs(device)
    xyz_reference = common[0].detach().clone().requires_grad_(True)
    xyz_optimized = common[0].detach().clone().requires_grad_(True)
    expected = _forward(reference, common, backbone=xyz_reference)
    actual = _forward(optimized, common, backbone=xyz_optimized)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        xyz_optimized.grad, xyz_reference.grad, atol=5e-5, rtol=5e-5
    )
    _assert_parameter_gradients_match(optimized, reference, atol=5e-5, rtol=5e-5)


def test_default_parameter_count_matches_current_model() -> None:
    reference = NaiveProteinMPNN()
    optimized = ProteinMPNN()
    assert sum(parameter.numel() for parameter in reference.parameters()) == 1_660_485
    assert sum(parameter.numel() for parameter in optimized.parameters()) == 1_660_485


def test_model_can_be_built_from_config() -> None:
    config = ProteinMPNNConfig(
        node_width=16,
        edge_width=16,
        hidden_width=16,
        encoder_depth=1,
        decoder_depth=2,
        k_neighbors=4,
        dropout=0,
    )
    model = ProteinMPNN(config)
    assert len(model.encoder.layers) == 1
    assert len(model.decoder.layers) == 2
    assert model.backbone_features.k_neighbors == 4


def test_production_forward_follows_parameter_dtype() -> None:
    _, production = _models()
    production.double()
    inputs = tuple(
        value.double() if value.is_floating_point() else value for value in _inputs()
    )
    with torch.no_grad():
        output = _forward(production, inputs)
    assert output.dtype == torch.float64


def test_production_state_dict_is_semantic_and_explicitly_converted() -> None:
    reference, production = _models()
    reference_keys = set(reference.state_dict())
    production_keys = set(production.state_dict())
    assert reference_keys != production_keys
    assert {
        "backbone_features.relative_position.embedding.weight",
        "edge_input_projection.weight",
        "sequence_embedding.weight",
        "encoder.layers.0.node_message.input_projection.weight",
        "encoder.layers.0.edge_message.output_projection.weight",
        "decoder.layers.0.node_transition.expand_projection.weight",
        "output_projection.weight",
    } <= production_keys
    assert not any(".W1" in key or ".dense." in key for key in production_keys)
    with pytest.raises(RuntimeError):
        production.load_state_dict(reference.state_dict(), strict=True)
    converted = convert_cssb_state_dict(reference.state_dict(), target=production)
    assert set(converted) == production_keys
    assert converted[
        "backbone_features.relative_position.embedding.weight"
    ].is_contiguous()


def test_production_state_dict_round_trip() -> None:
    _, model = _models()
    clone = ProteinMPNN(model.config)
    clone.load_state_dict(model.state_dict(), strict=True)
    for expected, actual in zip(
        model.state_dict().values(), clone.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_cssb_forward_adapter_is_explicit_and_numerically_transparent() -> None:
    _, model = _models()
    adapter = CSSBForwardAdapter(model)
    inputs = _inputs()
    with torch.no_grad():
        expected = _forward(model, inputs)
        actual = adapter(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("block_linear", [False, True], ids=["dense", "block"])
def test_training_mode_uses_backward_efficient_equivalent_path(
    block_linear: bool,
) -> None:
    threshold = 0 if block_linear else 1_000_000_000
    reference, optimized = _models(block_linear_min_edges=threshold)
    reference.train()
    optimized.train()
    common = _inputs()
    xyz_reference = common[0].detach().clone().requires_grad_(True)
    xyz_optimized = common[0].detach().clone().requires_grad_(True)
    expected = _forward(reference, common, backbone=xyz_reference)
    actual = _forward(optimized, common, backbone=xyz_optimized)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        xyz_optimized.grad, xyz_reference.grad, atol=3e-5, rtol=3e-5
    )
    _assert_parameter_gradients_match(optimized, reference, atol=5e-5, rtol=5e-5)


def test_parameter_only_block_training_path_matches_reference() -> None:
    reference, optimized = _models(block_linear_min_edges=0)
    reference.train()
    optimized.train()
    common = _inputs()
    expected = _forward(reference, common)
    actual = _forward(optimized, common)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    expected.square().mean().backward()
    actual.square().mean().backward()
    _assert_parameter_gradients_match(optimized, reference, atol=5e-5, rtol=5e-5)


def test_encoder_post_reduce_projection_preserves_masked_bias_and_gradients() -> None:
    dimensions = dict(d_node=5, d_edge=4, d_hidden=6, dropout=0.2, scale=7)
    reference = NaiveEncLayer(**dimensions).double()
    optimized = EncoderLayer(
        node_width=5,
        edge_width=4,
        hidden_width=6,
        dropout=0.2,
        neighbor_scale=7,
    ).double()
    _randomize_parameters(reference, seed=101)
    optimized.load_state_dict(
        convert_encoder_layer_state_dict(reference.state_dict(), target=optimized),
        strict=True,
    )

    generator = torch.Generator().manual_seed(102)
    h_v_reference = torch.randn(
        2, 5, 5, generator=generator, dtype=torch.float64, requires_grad=True
    )
    h_e_reference = torch.randn(
        2, 5, 3, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    h_v_optimized = h_v_reference.detach().clone().requires_grad_(True)
    h_e_optimized = h_e_reference.detach().clone().requires_grad_(True)
    edge_idx = torch.randint(0, 5, (2, 5, 3), generator=generator)
    mask_v = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    mask_attend = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 1], [1, 0, 0]],
            [[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        ],
        dtype=torch.float64,
    )

    torch.manual_seed(103)
    expected_v, expected_e = reference(
        h_v_reference, h_e_reference, edge_idx, mask_v, mask_attend
    )
    torch.manual_seed(103)
    actual_v, actual_e = optimized.forward_dense_training(
        h_v_optimized, h_e_optimized, edge_idx, mask_v, mask_attend
    )
    torch.testing.assert_close(actual_v, expected_v, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(actual_e, expected_e, atol=2e-11, rtol=2e-11)

    weight_v = torch.randn(expected_v.shape, generator=generator, dtype=torch.float64)
    weight_e = torch.randn(expected_e.shape, generator=generator, dtype=torch.float64)
    (expected_v * weight_v).sum().add((expected_e * weight_e).sum()).backward()
    (actual_v * weight_v).sum().add((actual_e * weight_e).sum()).backward()
    torch.testing.assert_close(
        h_v_optimized.grad, h_v_reference.grad, atol=2e-11, rtol=2e-11
    )
    torch.testing.assert_close(
        h_e_optimized.grad, h_e_reference.grad, atol=2e-11, rtol=2e-11
    )
    _assert_layer_parameter_gradients_match(
        optimized, reference, convert_encoder_layer_state_dict
    )


def test_decoder_single_context_preserves_source_bias_and_gradients() -> None:
    dimensions = dict(d_node=5, d_edge=4, d_hidden=6, dropout=0.2, scale=7)
    reference = NaiveDecLayer(**dimensions).double()
    optimized = DecoderLayer(
        node_width=5,
        edge_width=4,
        hidden_width=6,
        dropout=0.2,
        neighbor_scale=7,
    ).double()
    _randomize_parameters(reference, seed=201)
    optimized.load_state_dict(
        convert_decoder_layer_state_dict(reference.state_dict(), target=optimized),
        strict=True,
    )

    generator = torch.Generator().manual_seed(202)
    base_shapes = ((2, 5, 5), (2, 5, 3, 4), (2, 5, 5), (2, 5, 5))
    reference_inputs = [
        torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
        for shape in base_shapes
    ]
    optimized_inputs = [
        value.detach().clone().requires_grad_(True) for value in reference_inputs
    ]
    h_v_reference, h_e_reference, h_s_reference, h_encoder_reference = reference_inputs
    h_v_optimized, h_e_optimized, h_s_optimized, h_encoder_optimized = optimized_inputs
    edge_idx = torch.randint(0, 5, (2, 5, 3), generator=generator)
    mask_v = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    past = torch.randint(0, 2, (2, 5, 3, 1), generator=generator).double()
    mask_bw = mask_v[:, :, None, None] * past
    mask_fw = mask_v[:, :, None, None] * (1.0 - past)

    h_es = concatenate_neighbor_features(h_s_reference, h_e_reference, edge_idx)
    h_ex = concatenate_neighbor_features(
        torch.zeros_like(h_s_reference), h_e_reference, edge_idx
    )
    h_exv = concatenate_neighbor_features(h_encoder_reference, h_ex, edge_idx)
    source_context = (
        mask_bw * concatenate_neighbor_features(h_v_reference, h_es, edge_idx)
        + mask_fw * h_exv
    )
    torch.manual_seed(203)
    expected = reference(h_v_reference, source_context, mask_v)

    edge_context = (mask_fw + mask_bw) * h_e_optimized
    sequence_context = mask_bw * gather_neighbors(h_s_optimized, edge_idx)
    encoder_context = mask_fw * gather_neighbors(h_encoder_optimized, edge_idx)
    torch.manual_seed(203)
    actual = optimized.forward_dense_training(
        h_v_optimized,
        edge_context,
        sequence_context,
        encoder_context,
        edge_idx,
        mask_bw,
        mask_v,
    )
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)

    loss_weight = torch.randn(expected.shape, generator=generator, dtype=torch.float64)
    (expected * loss_weight).sum().backward()
    (actual * loss_weight).sum().backward()
    for actual_input, expected_input in zip(
        optimized_inputs, reference_inputs, strict=True
    ):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, atol=2e-11, rtol=2e-11
        )
    _assert_layer_parameter_gradients_match(
        optimized, reference, convert_decoder_layer_state_dict
    )


def test_split_encode_and_score_preserve_forward_and_gradients() -> None:
    _, forward_model = _models()
    _, split_model = _models()
    forward_model.train()
    split_model.train()
    common = _inputs()
    xyz_forward = common[0].detach().clone().requires_grad_(True)
    xyz_split = common[0].detach().clone().requires_grad_(True)

    expected = _forward(forward_model, common, backbone=xyz_forward)
    encoded = split_model.encode_backbone(
        xyz_split,
        common[2],
        common[3],
        common[4],
        common[8],
    )
    actual = split_model.score_sequence(
        encoded,
        common[1],
        common[5],
        common[6],
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(xyz_split.grad, xyz_forward.grad, atol=0, rtol=0)
    _assert_matching_parameter_gradients(split_model, forward_model)


def test_encoded_backbone_is_reused_across_sequence_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model = _models()
    model.eval()
    common = _inputs()
    alternate_seq = (common[1] + 1) % 21

    with torch.no_grad():
        expected = _forward(model, common)
        alternate_inputs = list(common)
        alternate_inputs[1] = alternate_seq
        expected_alternate = _forward(
            model, tuple(alternate_inputs), return_log_prob=True
        )

    feature_calls = 0

    original_build_graph = model.backbone_features.build_graph

    def count_feature_call(*args: torch.Tensor, **kwargs: object) -> NeighborGraph:
        nonlocal feature_calls
        feature_calls += 1
        return original_build_graph(*args, **kwargs)

    monkeypatch.setattr(model.backbone_features, "build_graph", count_feature_call)
    with torch.no_grad():
        encoded = model.encode_backbone(
            common[0], common[2], common[3], common[4], common[8]
        )
        actual = model.score_sequence(encoded, common[1], common[5], common[6])
        actual_alternate = model.score_sequence(
            encoded,
            alternate_seq,
            common[5],
            common[6],
            return_log_prob=True,
        )

    assert isinstance(encoded, EncodedMPNN)
    assert feature_calls == 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_alternate, expected_alternate, atol=0, rtol=0)


def test_training_reuse_combines_losses_before_backward() -> None:
    _, model = _models()
    model.train()
    common = _inputs()
    xyz = common[0].detach().clone().requires_grad_(True)
    encoded = model.encode_backbone(xyz, common[2], common[3], common[4], common[8])
    first = model.score_sequence(encoded, common[1], common[5], common[6])
    second = model.score_sequence(encoded, (common[1] + 1) % 21, common[5], common[6])
    (first.square().mean() + second.square().mean()).backward()
    assert xyz.grad is not None
    assert torch.isfinite(xyz.grad).all()


def test_encoded_backbone_rejects_wrong_owner_or_mode() -> None:
    _, model = _models()
    _, other_model = _models()
    common = _inputs()
    with torch.no_grad():
        encoded = model.encode_backbone(
            common[0], common[2], common[3], common[4], common[8]
        )
        with pytest.raises(ValueError, match="different model instance"):
            other_model.score_sequence(encoded, common[1], common[5], common[6])
        model.train()
        with pytest.raises(ValueError, match="different model mode"):
            model.score_sequence(encoded, common[1], common[5], common[6])


def test_packed_segments_and_invalid_residues_match_reference() -> None:
    reference, optimized = _models()
    common = list(_inputs(length=12))
    common[2][:, 3] = 0
    common[2][:, 10] = 0
    common[7] = common[2].clone()
    common[8] = torch.tensor([5, 7])
    with torch.no_grad():
        expected = _forward(reference, tuple(common))
        actual = _forward(optimized, tuple(common))
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("model_index", [0, 1], ids=["naive", "optimized"])
def test_packed_output_matches_concatenated_independent_segments(
    model_index: int,
) -> None:
    model = _models()[model_index]
    generator = torch.Generator().manual_seed(1701)
    lengths = (5, 7)
    patch_size = 2
    segments: list[tuple[torch.Tensor, ...]] = []
    outputs = []
    for length in lengths:
        xyz = torch.randn(1, length, 4, 3, generator=generator)
        seq = torch.randint(0, 21, (1, length), generator=generator)
        mask = torch.ones(1, length)
        residue_idx = torch.arange(length).unsqueeze(0)
        chain_idx = torch.zeros(1, length, dtype=torch.long)
        decoding_order = torch.randperm(length, generator=generator).unsqueeze(0)
        patch_index = (
            torch.arange((length + patch_size - 1) // patch_size)
            .repeat_interleave(patch_size)[:length]
            .unsqueeze(0)
        )
        segment = (
            xyz,
            seq,
            mask,
            residue_idx,
            chain_idx,
            decoding_order,
            patch_index,
            mask,
            torch.tensor([length]),
        )
        segments.append(segment)
        with torch.no_grad():
            outputs.append(_forward(model, segment))

    offsets = (0, lengths[0])
    packed = (
        torch.cat([segment[0] for segment in segments], dim=1),
        torch.cat([segment[1] for segment in segments], dim=1),
        torch.cat([segment[2] for segment in segments], dim=1),
        torch.cat([segment[3] for segment in segments], dim=1),
        torch.cat([segment[4] for segment in segments], dim=1),
        torch.cat(
            [
                segment[5] + offset
                for segment, offset in zip(segments, offsets, strict=True)
            ],
            dim=1,
        ),
        torch.cat(
            [
                segment[6] + offset
                for segment, offset in zip(segments, offsets, strict=True)
            ],
            dim=1,
        ),
        torch.cat([segment[7] for segment in segments], dim=1),
        torch.tensor(lengths),
    )
    with torch.no_grad():
        actual = _forward(model, packed)
    expected = torch.cat(outputs, dim=-1)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("model_index", [0, 1], ids=["naive", "optimized"])
def test_packed_lengths_must_sum_to_physical_length(model_index: int) -> None:
    model = _models()[model_index]
    inputs = list(_inputs(length=12))
    inputs[8] = torch.tensor([5, 6])
    with pytest.raises(RuntimeError, match="allocated size"):
        _forward(model, tuple(inputs))


def test_optimized_packed_path_is_fullgraph_compilable() -> None:
    _, model = _models()
    inputs = list(_inputs(length=12))
    inputs[3] = torch.cat((torch.arange(5), torch.arange(7))).unsqueeze(0)
    inputs[5] = torch.cat((torch.randperm(5), torch.randperm(7) + 5)).unsqueeze(0)
    inputs[8] = torch.tensor([5, 7])
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    with torch.no_grad():
        expected = _forward(model, tuple(inputs))
        actual = _forward(compiled, tuple(inputs))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("block_linear", [False, True], ids=["dense", "block"])
def test_checkpointed_training_path_is_differentiable(
    block_linear: bool,
) -> None:
    threshold = 0 if block_linear else 1_000_000_000
    _, direct = _models(block_linear_min_edges=threshold)
    _, checkpointed = _models(block_linear_min_edges=threshold)
    direct.train()
    checkpointed.train()
    common = _inputs(length=8)
    xyz_direct = common[0].detach().clone().requires_grad_(True)
    xyz_checkpointed = common[0].detach().clone().requires_grad_(True)
    expected = _forward(direct, common, backbone=xyz_direct)
    actual = _forward(
        checkpointed,
        common,
        backbone=xyz_checkpointed,
        checkpoint_layers=True,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(xyz_checkpointed.grad, xyz_direct.grad, atol=0, rtol=0)
    _assert_matching_parameter_gradients(checkpointed, direct)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bf16_mixed_forward_and_coordinate_gradient_track_reference() -> None:
    reference, optimized = _models("cuda")
    common = _inputs("cuda", length=32)
    xyz_reference = common[0].detach().clone().requires_grad_(True)
    xyz_optimized = common[0].detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _forward(reference, common, backbone=xyz_reference)
        actual = _forward(optimized, common, backbone=xyz_optimized)
    assert _cosine(actual, expected) >= 0.999
    assert _relative_frobenius(actual, expected) <= 0.02
    assert (actual.float() - expected.float()).abs().max().item() <= 0.05

    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    assert _cosine(xyz_optimized.grad, xyz_reference.grad) >= 0.995
    assert _relative_frobenius(xyz_optimized.grad, xyz_reference.grad) <= 0.08
    assert (
        xyz_optimized.grad.float() - xyz_reference.grad.float()
    ).abs().max().item() <= 0.5
