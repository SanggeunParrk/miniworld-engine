"""Checkpoint-schema tests for the independent production MPNN model."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import pytest
import torch

from miniworld_kernels.modules.mpnn import (
    NaiveProteinMPNN,
    ProteinMPNN,
    ProteinMPNNConfig,
    convert_cssb_state_dict,
    load_cssb_weights,
    reference_to_production_key,
)


def _asymmetric_models() -> tuple[NaiveProteinMPNN, ProteinMPNN]:
    reference = NaiveProteinMPNN(
        node_features=5,
        edge_features=7,
        hidden_dim=11,
        num_encoder_layers=2,
        num_decoder_layers=1,
        k_neighbors=4,
        augment_trans=0,
        augment_rot=0,
        dropout=0,
    )
    production = ProteinMPNN(
        ProteinMPNNConfig(
            node_width=5,
            edge_width=7,
            hidden_width=11,
            encoder_depth=2,
            decoder_depth=1,
            k_neighbors=4,
            coordinate_noise=0,
            dropout=0,
        )
    )
    return reference, production


def test_converter_maps_every_tensor_without_aliasing_source_storage() -> None:
    reference, production = _asymmetric_models()
    source = reference.state_dict()
    converted = convert_cssb_state_dict(source, target=production)

    assert set(converted) == set(production.state_dict())
    assert len(converted) == len(source)
    for reference_key, source_value in source.items():
        production_key = reference_to_production_key(reference_key)
        expected = (
            source_value.T
            if reference_key == "features.embeddings.linear.weight"
            else source_value
        )
        torch.testing.assert_close(converted[production_key], expected, atol=0, rtol=0)
        assert converted[production_key].untyped_storage().data_ptr() != (
            source_value.untyped_storage().data_ptr()
        )
    assert converted[
        "backbone_features.relative_position.embedding.weight"
    ].is_contiguous()


def test_converter_rejects_incomplete_unknown_and_wrong_shape_states() -> None:
    reference, production = _asymmetric_models()
    source = reference.state_dict()

    missing = OrderedDict(source)
    missing.pop("W_out.bias")
    with pytest.raises(ValueError, match="missing="):
        convert_cssb_state_dict(missing, target=production)

    unknown = OrderedDict(source)
    unknown["legacy.surprise"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unrecognized frozen CSSB state key"):
        convert_cssb_state_dict(unknown, target=production)

    wrong_shape = OrderedDict(source)
    wrong_shape["W_e.weight"] = source["W_e.weight"][:-1].clone()
    with pytest.raises(ValueError, match="shape mismatch"):
        convert_cssb_state_dict(wrong_shape, target=production)

    with pytest.raises(ValueError, match="unrecognized frozen CSSB state key"):
        convert_cssb_state_dict(production.state_dict(), target=production)


def test_default_production_schema_has_no_legacy_parameter_names() -> None:
    model = ProteinMPNN()
    keys = set(model.state_dict())
    assert len(keys) == 118
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_660_485
    assert not any(
        key.startswith(("W_", "encoder_layers", "decoder_layers"))
        or ".W1" in key
        or ".dense." in key
        for key in keys
    )
    assert "encoder.layers.0.node_message.input_projection.weight" in keys
    assert "decoder.layers.0.node_transition.norm.weight" in keys


@pytest.mark.parametrize("edge_backend", ["triton_compute", "triton_memory"])
def test_execution_backends_do_not_change_checkpoint_schema(edge_backend: str) -> None:
    pytorch_model = ProteinMPNN(
        ProteinMPNNConfig(
            message_backend="pytorch",
            edge_mlp_backend="pytorch",
        )
    )
    triton_model = ProteinMPNN(
        replace(
            pytorch_model.config,
            message_backend="triton",
            edge_mlp_backend=edge_backend,
        )
    )

    pytorch_state = pytorch_model.state_dict()
    triton_state = triton_model.state_dict()
    assert tuple(pytorch_state) == tuple(triton_state)
    assert {
        key: (value.shape, value.dtype) for key, value in pytorch_state.items()
    } == {key: (value.shape, value.dtype) for key, value in triton_state.items()}
    triton_model.load_state_dict(pytorch_state, strict=True)


def test_loader_validates_k_neighbors_missing_from_tensor_schema() -> None:
    reference, production = _asymmetric_models()
    load_cssb_weights(production, reference)

    raw_state = reference.state_dict()
    with pytest.raises(ValueError, match="source_k_neighbors is required"):
        load_cssb_weights(ProteinMPNN(production.config), raw_state)

    clone = ProteinMPNN(production.config)
    load_cssb_weights(clone, raw_state, source_k_neighbors=4)
    for expected, actual in zip(
        production.state_dict().values(), clone.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    wrong_k = ProteinMPNN(
        ProteinMPNNConfig(
            node_width=5,
            edge_width=7,
            hidden_width=11,
            encoder_depth=2,
            decoder_depth=1,
            k_neighbors=5,
            coordinate_noise=0,
            dropout=0,
        )
    )
    with pytest.raises(ValueError, match="k_neighbors differ"):
        load_cssb_weights(wrong_k, reference)
    with pytest.raises(ValueError, match="disagrees"):
        load_cssb_weights(production, reference, source_k_neighbors=5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node_width", 0),
        ("k_neighbors", 0),
        ("coordinate_noise", -0.1),
        ("dropout", 1.0),
        ("block_linear_min_edges", -1),
    ],
)
def test_production_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        ProteinMPNNConfig(**{field: value})


def test_production_config_rejects_unknown_edge_mlp_backend() -> None:
    with pytest.raises(ValueError, match="edge_mlp_backend"):
        ProteinMPNNConfig(edge_mlp_backend="triton")


def test_production_config_rejects_unknown_message_backend() -> None:
    with pytest.raises(ValueError, match="message_backend"):
        ProteinMPNNConfig(message_backend="triton_fastest")


def test_production_config_rejects_unknown_feature_backend() -> None:
    with pytest.raises(ValueError, match="feature_backend"):
        ProteinMPNNConfig(feature_backend="checkpoint_everything")


def test_production_config_rejects_unknown_edge_norm_backend() -> None:
    with pytest.raises(ValueError, match="edge_norm_backend"):
        ProteinMPNNConfig(edge_norm_backend="triton")


def test_edge_w1_checkpoint_requires_memory_edge_mlp() -> None:
    with pytest.raises(ValueError, match="edge_w1_recompute"):
        ProteinMPNNConfig(edge_w1_recompute="checkpoint")


def test_encoder_node_w1_checkpoint_requires_memory_message_backend() -> None:
    with pytest.raises(ValueError, match="encoder_node_w1_recompute"):
        ProteinMPNNConfig(encoder_node_w1_recompute="checkpoint")
    with pytest.raises(ValueError, match="encoder_node_w1_recompute"):
        ProteinMPNNConfig(
            message_backend="triton_memory",
            encoder_node_w1_recompute="always",
        )


def test_production_config_rejects_unknown_transition_recompute() -> None:
    with pytest.raises(ValueError, match="transition_recompute"):
        ProteinMPNNConfig(transition_recompute="checkpoint")


def test_memory_execution_policies_do_not_change_state_dict_schema() -> None:
    compute = ProteinMPNN(ProteinMPNNConfig())
    memory = ProteinMPNN(
        ProteinMPNNConfig(
            message_backend="triton_memory",
            edge_mlp_backend="triton_memory",
            edge_norm_backend="memory",
            edge_dropout_backend="bitpack",
            feature_backend="recompute",
            edge_w1_recompute="checkpoint",
            encoder_node_w1_recompute="checkpoint",
            transition_recompute="update",
        )
    )
    compute_state = compute.state_dict()
    memory_state = memory.state_dict()
    assert compute_state.keys() == memory_state.keys()
    assert {name: tensor.shape for name, tensor in compute_state.items()} == {
        name: tensor.shape for name, tensor in memory_state.items()
    }
    memory.load_state_dict(compute_state, strict=True)
