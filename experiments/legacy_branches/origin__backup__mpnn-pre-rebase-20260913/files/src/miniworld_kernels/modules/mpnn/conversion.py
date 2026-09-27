"""Explicit one-way conversion from the frozen CSSB checkpoint schema."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
import operator
import re

import torch
import torch.nn as nn


_ENCODER_LAYER_RE = re.compile(r"^encoder_layers\.(\d+)\.(.+)$")
_DECODER_LAYER_RE = re.compile(r"^decoder_layers\.(\d+)\.(.+)$")

_ENCODER_COMPONENTS = {
    "W1": "node_message.input_projection",
    "W2": "node_message.hidden_projection",
    "W3": "node_message.output_projection",
    "norm1": "node_message.norm",
    "dense.W_in": "node_transition.expand_projection",
    "dense.W_out": "node_transition.output_projection",
    "norm2": "node_transition.norm",
    "W11": "edge_message.input_projection",
    "W12": "edge_message.hidden_projection",
    "W13": "edge_message.output_projection",
    "norm3": "edge_message.norm",
}

_DECODER_COMPONENTS = {
    "W1": "node_message.input_projection",
    "W2": "node_message.hidden_projection",
    "W3": "node_message.output_projection",
    "norm1": "node_message.norm",
    "dense.W_in": "node_transition.expand_projection",
    "dense.W_out": "node_transition.output_projection",
    "norm2": "node_transition.norm",
}

_TOP_LEVEL_PREFIXES = {
    "features.embeddings.linear": "backbone_features.relative_position",
    "features.edge_embedding": "backbone_features.edge_projection",
    "features.norm_edges": "backbone_features.edge_norm",
    "W_e": "edge_input_projection",
    "W_s": "sequence_embedding",
    "W_out": "output_projection",
}


def _map_component(tail: str, components: Mapping[str, str]) -> str:
    for source, target in components.items():
        prefix = f"{source}."
        if tail.startswith(prefix):
            return f"{target}.{tail.removeprefix(prefix)}"
    raise ValueError(f"unrecognized frozen layer key: {tail!r}")


def reference_to_production_key(reference_key: str) -> str:
    """Map one frozen CSSB parameter/buffer key to the production schema."""
    encoder_match = _ENCODER_LAYER_RE.match(reference_key)
    if encoder_match:
        layer_index, tail = encoder_match.groups()
        return (
            f"encoder.layers.{layer_index}.{_map_component(tail, _ENCODER_COMPONENTS)}"
        )

    decoder_match = _DECODER_LAYER_RE.match(reference_key)
    if decoder_match:
        layer_index, tail = decoder_match.groups()
        return (
            f"decoder.layers.{layer_index}.{_map_component(tail, _DECODER_COMPONENTS)}"
        )

    for source, target in _TOP_LEVEL_PREFIXES.items():
        prefix = f"{source}."
        if reference_key.startswith(prefix):
            suffix = reference_key.removeprefix(prefix)
            if source == "features.embeddings.linear" and suffix == "weight":
                return f"{target}.embedding.weight"
            return f"{target}.{suffix}"
    raise ValueError(f"unrecognized frozen CSSB state key: {reference_key!r}")


def _convert_tensor(reference_key: str, value: torch.Tensor) -> torch.Tensor:
    if reference_key == "features.embeddings.linear.weight":
        return value.detach().T.contiguous().clone()
    return value.detach().clone()


def _validate_target(
    converted: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> None:
    converted_keys = set(converted)
    target_keys = set(target_state)
    missing = sorted(target_keys - converted_keys)
    unexpected = sorted(converted_keys - target_keys)
    if missing or unexpected:
        raise ValueError(
            "frozen CSSB checkpoint does not match the production architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )
    shape_errors = [
        f"{key}: converted={tuple(converted[key].shape)}, "
        f"target={tuple(target_state[key].shape)}"
        for key in target_state
        if converted[key].shape != target_state[key].shape
    ]
    if shape_errors:
        raise ValueError(
            "frozen CSSB checkpoint shape mismatch: " + "; ".join(shape_errors)
        )


def convert_cssb_state_dict(
    reference_state: Mapping[str, torch.Tensor],
    *,
    target: nn.Module | Mapping[str, torch.Tensor] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Convert a frozen ``origin/dev`` state dict without mutating its tensors."""
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for reference_key, value in reference_state.items():
        production_key = reference_to_production_key(reference_key)
        if production_key in converted:
            raise ValueError(f"duplicate converted state key: {production_key!r}")
        converted[production_key] = _convert_tensor(reference_key, value)

    if target is not None:
        target_state = target.state_dict() if isinstance(target, nn.Module) else target
        _validate_target(converted, target_state)
    return converted


def _positive_integer(value: object, *, name: str) -> int:
    try:
        converted = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer, got {converted}")
    return converted


def _infer_reference_k_neighbors(reference: nn.Module) -> int | None:
    candidates: list[tuple[str, int]] = []
    features = getattr(reference, "features", None)
    if features is not None and hasattr(features, "top_k"):
        top_k = getattr(features, "top_k")
        if top_k is not None:
            candidates.append(
                (
                    "reference.features.top_k",
                    _positive_integer(top_k, name="reference.features.top_k"),
                )
            )

    for stack_name in ("encoder_layers", "decoder_layers"):
        layers = getattr(reference, stack_name, ())
        for index, layer in enumerate(layers):
            if hasattr(layer, "scale"):
                name = f"reference.{stack_name}.{index}.scale"
                candidates.append(
                    (name, _positive_integer(getattr(layer, "scale"), name=name))
                )

    if not candidates:
        return None
    distinct = {value for _, value in candidates}
    if len(distinct) != 1:
        details = ", ".join(f"{name}={value}" for name, value in candidates)
        raise ValueError(f"inconsistent frozen CSSB k_neighbors metadata: {details}")
    return candidates[0][1]


def load_cssb_weights(
    model: nn.Module,
    reference: nn.Module | Mapping[str, torch.Tensor],
    *,
    source_k_neighbors: int | None = None,
) -> None:
    """Load frozen CSSB weights and validate non-tensor graph metadata."""
    target_config = getattr(model, "config", None)
    if target_config is None or not hasattr(target_config, "k_neighbors"):
        raise TypeError("production model must expose config.k_neighbors")
    target_k_neighbors = _positive_integer(
        getattr(target_config, "k_neighbors"), name="model.config.k_neighbors"
    )

    inferred_k_neighbors = (
        _infer_reference_k_neighbors(reference)
        if isinstance(reference, nn.Module)
        else None
    )
    if source_k_neighbors is None:
        source_k_neighbors = inferred_k_neighbors
    else:
        source_k_neighbors = _positive_integer(
            source_k_neighbors, name="source_k_neighbors"
        )
        if (
            inferred_k_neighbors is not None
            and source_k_neighbors != inferred_k_neighbors
        ):
            raise ValueError(
                "source_k_neighbors disagrees with the frozen CSSB module: "
                f"explicit={source_k_neighbors}, inferred={inferred_k_neighbors}"
            )
    if source_k_neighbors is None:
        raise ValueError(
            "source_k_neighbors is required when loading a raw CSSB state dict; "
            "the tensor schema does not encode the graph neighbor count"
        )
    if source_k_neighbors != target_k_neighbors:
        raise ValueError(
            "frozen CSSB and production k_neighbors differ: "
            f"source={source_k_neighbors}, target={target_k_neighbors}"
        )

    reference_state = (
        reference.state_dict() if isinstance(reference, nn.Module) else reference
    )
    converted = convert_cssb_state_dict(reference_state, target=model)
    model.load_state_dict(converted, strict=True)


def production_tensor_in_reference_layout(
    reference_key: str, production_tensor: torch.Tensor
) -> torch.Tensor:
    """View a production tensor in the frozen parameter's logical layout."""
    if reference_key == "features.embeddings.linear.weight":
        return production_tensor.T
    return production_tensor


def iter_reference_parameter_pairs(
    reference: nn.Module, production: nn.Module
) -> Iterator[tuple[str, nn.Parameter, str, nn.Parameter]]:
    """Yield parameter pairs in canonical frozen-checkpoint order."""
    production_parameters = dict(production.named_parameters())
    for reference_key, reference_parameter in reference.named_parameters():
        production_key = reference_to_production_key(reference_key)
        try:
            production_parameter = production_parameters[production_key]
        except KeyError as error:
            raise ValueError(
                f"production model has no parameter for {reference_key!r}: "
                f"expected {production_key!r}"
            ) from error
        yield (
            reference_key,
            reference_parameter,
            production_key,
            production_parameter,
        )


def convert_encoder_layer_state_dict(
    reference_state: Mapping[str, torch.Tensor],
    *,
    target: nn.Module | Mapping[str, torch.Tensor] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Convert one frozen encoder layer; intended for focused layer tests."""
    converted = OrderedDict(
        (_map_component(key, _ENCODER_COMPONENTS), value.detach().clone())
        for key, value in reference_state.items()
    )
    if target is not None:
        target_state = target.state_dict() if isinstance(target, nn.Module) else target
        _validate_target(converted, target_state)
    return converted


def convert_decoder_layer_state_dict(
    reference_state: Mapping[str, torch.Tensor],
    *,
    target: nn.Module | Mapping[str, torch.Tensor] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Convert one frozen decoder layer; intended for focused layer tests."""
    converted = OrderedDict(
        (_map_component(key, _DECODER_COMPONENTS), value.detach().clone())
        for key, value in reference_state.items()
    )
    if target is not None:
        target_state = target.state_dict() if isinstance(target, nn.Module) else target
        _validate_target(converted, target_state)
    return converted
