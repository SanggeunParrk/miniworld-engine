"""Semantic message-passing layers for the production MPNN model."""

from __future__ import annotations

from typing import Literal, cast
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from miniworld_kernels.kernels.mpnn_edge_dropout import EdgeDropoutBackend
from miniworld_kernels.kernels.mpnn_edge_layernorm import (
    EdgeNormBackend,
    edge_layer_norm,
)
from miniworld_kernels.kernels.mpnn_edge_mlp import EdgeMLPBackend, edge_mlp_update
from miniworld_kernels.kernels.mpnn_edge_mlp.interface import (
    _triton_contract_supported as _edge_mlp_contract_supported,
)
from miniworld_kernels.kernels.mpnn_edge_tail import (
    EdgeTailBackend,
    edge_tail_supported,
    edge_tail_update,
)
from miniworld_kernels.kernels.mpnn_message import MessageBackend, message_hidden_reduce
from miniworld_kernels.kernels.mpnn_node_message import (
    NodeMessageBackend,
    node_message_reduce,
    node_message_supported,
)
from miniworld_kernels.kernels.mpnn_message.interface import (
    _triton_contract_supported as _message_contract_supported,
)

from ._functional import (
    _flat_neighbor_indices,
    gather_neighbors,
    lecun_normal_,
    project_packed_neighbor_inputs,
)
from .dropout import EdgeDropout


DEFAULT_BLOCK_LINEAR_MIN_EDGES = 49_152
EdgeW1Recompute = Literal["off", "checkpoint"]
EncoderNodeW1Recompute = Literal["off", "checkpoint"]
DecoderNodeW1Recompute = Literal["off", "checkpoint"]
TransitionRecompute = Literal["off", "update"]
EncoderExecutionPath = Literal[
    "block",
    "dense_training",
    "zero_node_training",
    "node_recompute",
    "zero_node_recompute",
]


def _warn_unmet_recompute_contract(module: nn.Module, policy: str) -> None:
    """Warn once per module when an explicit recompute policy cannot engage.

    These policies fall back to the ordinary path on any contract mismatch --
    width, K, dtype, layout, or grad mode. Silence would let a configured memory
    policy be measured as if it were active, so the first fallback is reported.
    Tracing is exempt: warning state is a module mutation and would either break
    a ``fullgraph`` capture or bake a stale guard into the compiled graph.
    """
    if torch.compiler.is_compiling():
        return
    attribute = f"_warned_unmet_{policy}"
    if getattr(module, attribute, False):
        return
    setattr(module, attribute, True)
    warnings.warn(
        f"{policy} was requested but its runtime contract is unmet for these "
        "inputs (width/K, dtype, contiguity, device, or grad mode); this layer "
        "is running the ordinary path with no recomputation",
        RuntimeWarning,
        stacklevel=3,
    )


def _supports_edge_w1_recompute(
    node_states: torch.Tensor,
    edge_states: torch.Tensor,
    edge_projection: nn.Linear,
    edge_message: MessageUpdate,
) -> bool:
    """Return whether the compressed BF16 checkpoint path is valid."""
    w1_parameters = (edge_projection.weight, edge_projection.bias)
    edge_parameters = (
        edge_message.hidden_projection.weight,
        edge_message.hidden_projection.bias,
        edge_message.output_projection.weight,
        edge_message.output_projection.bias,
    )
    parameters = (
        edge_projection.weight,
        edge_projection.bias,
        *edge_parameters,
    )
    if edge_states.ndim == 0 or any(parameter is None for parameter in parameters):
        return False
    tensors = (node_states, edge_states, *parameters)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not all(tensor.device == edge_states.device for tensor in tensors):
        return False

    autocast_bf16 = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    native_w1 = all(
        tensor.dtype == torch.bfloat16
        for tensor in (node_states, edge_states, *w1_parameters)
    )
    autocast_w1 = (
        autocast_bf16
        and node_states.dtype in {torch.float32, torch.bfloat16}
        and edge_states.dtype in {torch.float32, torch.bfloat16}
        and len({parameter.dtype for parameter in w1_parameters}) == 1
        and w1_parameters[0].dtype in {torch.float32, torch.bfloat16}
    )
    w1_contract = (
        (native_w1 or autocast_w1)
        and edge_projection.out_features == 128
        and edge_projection.weight.shape[0] == 128
        and edge_projection.bias.shape == (128,)
        and all(parameter.is_contiguous() for parameter in w1_parameters)
    )
    rows = edge_states.numel() // edge_states.shape[-1]
    preactivation_elements = rows * edge_projection.out_features
    edge_contract = _edge_mlp_contract_supported(
        device=edge_states.device,
        dtype=torch.bfloat16,
        numel=preactivation_elements,
        width=edge_projection.out_features,
        contiguous=True,
        hidden_weight=edge_parameters[0],
        hidden_bias=edge_parameters[1],
        output_weight=edge_parameters[2],
        output_bias=edge_parameters[3],
    )
    return w1_contract and edge_contract


def _supports_encoder_node_w1_recompute(
    node_states: torch.Tensor,
    edge_states: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_mask: torch.Tensor | None,
    input_projection: PackedEncoderProjection,
    node_message: MessageUpdate,
) -> bool:
    """Return whether the encoder node W1/reduction checkpoint is valid."""
    weight = input_projection.weight
    bias = input_projection.bias
    if (
        not torch.is_grad_enabled()
        or neighbor_mask is None
        or node_message.reduction_backend != "triton_memory"
        or bias is None
        or node_states.ndim != 3
        or edge_states.ndim != 4
        or neighbor_indices.shape != edge_states.shape[:-1]
        or node_states.shape[:2] != edge_states.shape[:2]
    ):
        return False

    tensors = (node_states, edge_states, neighbor_indices, weight, bias)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not all(tensor.device == edge_states.device for tensor in tensors):
        return False

    autocast_bf16 = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    native_w1 = all(
        tensor.dtype == torch.bfloat16
        for tensor in (node_states, edge_states, weight, bias)
    )
    autocast_w1 = (
        autocast_bf16
        and node_states.dtype in {torch.float32, torch.bfloat16}
        and edge_states.dtype in {torch.float32, torch.bfloat16}
        and weight.dtype in {torch.float32, torch.bfloat16}
        and bias.dtype == weight.dtype
    )
    w1_contract = (
        (native_w1 or autocast_w1)
        and input_projection.node_width == 128
        and input_projection.edge_width == 128
        and input_projection.in_features == 384
        and input_projection.out_features == 128
        and node_states.shape[-1] == 128
        and edge_states.shape[-2:] == (48, 128)
        and weight.shape == (128, 384)
        and bias.shape == (128,)
        and node_states.is_contiguous()
        and edge_states.is_contiguous()
        and neighbor_indices.is_contiguous()
        and neighbor_indices.dtype == torch.long
        and weight.is_contiguous()
        and bias.is_contiguous()
    )
    preactivation_shape = (*edge_states.shape[:-1], input_projection.out_features)
    message_contract = _message_contract_supported(
        device=edge_states.device,
        dtype=torch.bfloat16,
        shape=preactivation_shape,
        contiguous=True,
        weight=node_message.hidden_projection.weight,
        bias=node_message.hidden_projection.bias,
        edge_mask=neighbor_mask,
    )
    return w1_contract and message_contract


def _supports_decoder_node_w1_recompute(
    node_states: torch.Tensor,
    edge_states: torch.Tensor,
    neighbor_indices: torch.Tensor,
    edge_mask: torch.Tensor | None,
    input_projection: PackedDecoderProjection,
    node_message: MessageUpdate,
) -> bool:
    """Return whether the decoder node W1/reduction checkpoint is valid.

    The decoder replays the same packed projection plus fused reduction as the
    encoder, so it needs the same contract. Its projection is wider -- query,
    edge, sequence and neighbour blocks -- but only the fused reduction's
    contract constrains the widths that matter.
    """
    weight = input_projection.weight
    bias = input_projection.bias
    if (
        not torch.is_grad_enabled()
        or edge_mask is None
        or node_message.reduction_backend != "triton_memory"
        or bias is None
        or node_states.ndim != 3
        or edge_states.ndim != 4
        or neighbor_indices.shape != edge_states.shape[:-1]
        or node_states.shape[:2] != edge_states.shape[:2]
    ):
        return False

    tensors = (node_states, edge_states, neighbor_indices, weight, bias)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not all(tensor.device == edge_states.device for tensor in tensors):
        return False

    autocast_bf16 = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    native_w1 = all(
        tensor.dtype == torch.bfloat16
        for tensor in (node_states, edge_states, weight, bias)
    )
    autocast_w1 = (
        autocast_bf16
        and node_states.dtype in {torch.float32, torch.bfloat16}
        and edge_states.dtype in {torch.float32, torch.bfloat16}
        and weight.dtype in {torch.float32, torch.bfloat16}
        and bias.dtype == weight.dtype
    )
    w1_contract = (
        (native_w1 or autocast_w1)
        and input_projection.out_features == 128
        and node_states.shape[-1] == 128
        and edge_states.shape[-2:] == (48, 128)
        and neighbor_indices.dtype == torch.long
        and weight.is_contiguous()
        and bias.is_contiguous()
    )
    preactivation_shape = (*edge_states.shape[:-1], input_projection.out_features)
    message_contract = _message_contract_supported(
        device=edge_states.device,
        dtype=torch.bfloat16,
        shape=preactivation_shape,
        contiguous=True,
        weight=node_message.hidden_projection.weight,
        bias=node_message.hidden_projection.bias,
        edge_mask=edge_mask,
    )
    return w1_contract and message_contract


class PackedEncoderProjection(nn.Linear):
    """Single packed ``[query, edge, neighbor]`` message projection."""

    def __init__(self, node_width: int, edge_width: int, hidden_width: int) -> None:
        self.node_width = node_width
        self.edge_width = edge_width
        super().__init__(2 * node_width + edge_width, hidden_width, bias=True)

    def block(
        self,
        query_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        return project_packed_neighbor_inputs(
            query_nodes,
            edge_features,
            neighbor_nodes,
            neighbor_indices,
            self,
            self.node_width,
            self.edge_width,
        )

    def dense(
        self,
        query_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        neighbors = gather_neighbors(neighbor_nodes, neighbor_indices)
        queries = query_nodes.unsqueeze(2).expand(-1, -1, edge_features.shape[2], -1)
        return self(torch.cat((queries, edge_features, neighbors), dim=-1))

    def edge_only(self, edge_features: torch.Tensor) -> torch.Tensor:
        start = self.node_width
        stop = start + self.edge_width
        return F.linear(edge_features, self.weight[:, start:stop], self.bias)


class PackedDecoderProjection(nn.Linear):
    """Single packed ``[query, edge, sequence, neighbor]`` projection."""

    def __init__(self, node_width: int, edge_width: int, hidden_width: int) -> None:
        self.node_width = node_width
        self.edge_width = edge_width
        super().__init__(3 * node_width + edge_width, hidden_width, bias=True)

    def block(
        self,
        query_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        sequence_features: torch.Tensor,
        encoder_nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
        future_mask: torch.Tensor,
        past_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight
        node_width = self.node_width
        edge_width = self.edge_width

        sequence_start = node_width + edge_width
        sequence_stop = 2 * node_width + edge_width
        sequence_projection = F.linear(
            sequence_features, weight[:, sequence_start:sequence_stop]
        )
        sequence_neighbors = gather_neighbors(sequence_projection, neighbor_indices)

        output = F.linear(query_nodes, weight[:, :node_width], self.bias).unsqueeze(2)
        edge_projection = F.linear(edge_features, weight[:, node_width:sequence_start])
        edge_mask = (future_mask + past_mask).to(edge_projection.dtype)
        output = output + edge_mask * edge_projection

        past_compute = past_mask.to(sequence_neighbors.dtype)
        output = output + past_compute * sequence_neighbors
        neighbor_weight = weight[:, sequence_stop:]
        current_neighbors = gather_neighbors(
            F.linear(query_nodes, neighbor_weight), neighbor_indices
        )
        encoder_neighbors = gather_neighbors(
            F.linear(encoder_nodes, neighbor_weight), neighbor_indices
        )
        future_compute = future_mask.to(encoder_neighbors.dtype)
        return (
            output
            + past_compute * current_neighbors
            + future_compute * encoder_neighbors
        )

    def dense(
        self,
        query_nodes: torch.Tensor,
        edge_context: torch.Tensor,
        sequence_context: torch.Tensor,
        encoder_context: torch.Tensor,
        neighbor_indices: torch.Tensor,
        past_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = query_nodes.unsqueeze(2).expand(-1, -1, edge_context.shape[2], -1)
        current_context = past_mask * gather_neighbors(query_nodes, neighbor_indices)
        neighbor_context = current_context + encoder_context
        packed = torch.cat(
            (queries, edge_context, sequence_context, neighbor_context), dim=-1
        )
        return self(packed)


class MessageUpdate(nn.Module):
    """Three-projection message MLP plus its residual normalization."""

    def __init__(
        self,
        input_projection: nn.Linear,
        hidden_width: int,
        output_width: int,
        dropout: float,
        reduction_backend: MessageBackend = "auto",
        edge_mlp_backend: EdgeMLPBackend = "auto",
        edge_norm_backend: EdgeNormBackend = "auto",
        edge_dropout_backend: EdgeDropoutBackend | None = None,
    ) -> None:
        super().__init__()
        self.input_projection = input_projection
        self.hidden_projection = nn.Linear(hidden_width, hidden_width, bias=True)
        self.output_projection = nn.Linear(hidden_width, output_width, bias=True)
        self.norm = nn.LayerNorm(output_width)
        if edge_dropout_backend is None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = EdgeDropout(dropout, backend=edge_dropout_backend)
        self.activation = nn.GELU()
        self.reduction_backend = reduction_backend
        self.edge_mlp_backend = edge_mlp_backend
        self.edge_norm_backend = edge_norm_backend
        for projection in (
            self.input_projection,
            self.hidden_projection,
            self.output_projection,
        ):
            lecun_normal_(projection)
            nn.init.zeros_(projection.bias)

    def hidden_features(self, preactivation: torch.Tensor) -> torch.Tensor:
        return self.activation(self.hidden_projection(self.activation(preactivation)))

    def reduced_hidden_features(
        self,
        preactivation: torch.Tensor,
        edge_mask: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        return message_hidden_reduce(
            preactivation,
            self.hidden_projection.weight,
            self.hidden_projection.bias,
            edge_mask,
            scale,
            backend=self.reduction_backend,
        )

    def apply_reduced_hidden(
        self,
        node_states: torch.Tensor,
        reduced_hidden: torch.Tensor,
        bias_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        update = F.linear(reduced_hidden, self.output_projection.weight).to(
            reduced_hidden.dtype
        )
        if self.output_projection.bias is not None:
            update = update + bias_scale * self.output_projection.bias
        return self.norm(node_states + self.dropout(update))

    def apply_edgewise(
        self, edge_states: torch.Tensor, message_hidden: torch.Tensor
    ) -> torch.Tensor:
        update = self.output_projection(message_hidden)
        return self.apply_edgewise_update(edge_states, update)

    def apply_edgewise_preactivation(
        self,
        edge_states: torch.Tensor,
        preactivation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the fused hidden/output edge MLP before residual normalization."""
        update = self.edge_update_from_preactivation(preactivation)
        return self.apply_edgewise_update(edge_states, update)

    def edge_update_from_preactivation(
        self,
        preactivation: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate only the edge MLP update, without residual normalization."""
        return edge_mlp_update(
            preactivation,
            self.hidden_projection.weight,
            self.hidden_projection.bias,
            self.output_projection.weight,
            self.output_projection.bias,
            backend=self.edge_mlp_backend,
        )

    def apply_edgewise_update(
        self,
        edge_states: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """Apply dropout, residual addition, and normalization to an edge update."""
        values = edge_states + self.dropout(update)
        if self.edge_norm_backend != "memory":
            # Preserve nn.LayerNorm's ordinary module call boundary, including
            # user-installed forward hooks, for the compute-oriented policies.
            return self.norm(values)
        return edge_layer_norm(
            values,
            self.norm.weight,
            self.norm.bias,
            self.norm.eps,
            backend="memory",
        )


class ResidualTransition(nn.Module):
    """Position-wise feed-forward transition with residual normalization."""

    def __init__(
        self,
        width: int,
        dropout: float,
        transition_recompute: TransitionRecompute = "off",
    ) -> None:
        super().__init__()
        self.transition_recompute = transition_recompute
        self.expand_projection = nn.Linear(width, width * 4, bias=True)
        self.output_projection = nn.Linear(width * 4, width, bias=True)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        nn.init.kaiming_normal_(self.expand_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.expand_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _update(self, states: torch.Tensor) -> torch.Tensor:
        """Compute the deterministic MLP update kept inside the replay boundary."""
        return self.output_projection(self.activation(self.expand_projection(states)))

    def forward(
        self,
        states: torch.Tensor,
        *,
        allow_recompute: bool = True,
    ) -> torch.Tensor:
        checkpoint_update = (
            self.transition_recompute == "update"
            and allow_recompute
            and torch.is_grad_enabled()
        )
        if checkpoint_update:
            update = torch.utils.checkpoint.checkpoint(
                self._update,
                states,
                use_reentrant=False,
                # Only deterministic projections and GELU are replayed. Dropout
                # remains below, outside this boundary.
                preserve_rng_state=False,
            )
        else:
            update = self._update(states)
        return self.norm(states + self.dropout(update))


class EncoderLayer(nn.Module):
    """Node and edge updates for one MPNN encoder layer."""

    def __init__(
        self,
        node_width: int,
        edge_width: int,
        hidden_width: int,
        dropout: float,
        neighbor_scale: int,
        reduction_backend: MessageBackend = "auto",
        edge_mlp_backend: EdgeMLPBackend = "auto",
        edge_w1_recompute: EdgeW1Recompute = "off",
        edge_norm_backend: EdgeNormBackend = "auto",
        transition_recompute: TransitionRecompute = "off",
        edge_dropout_backend: EdgeDropoutBackend = "auto",
        edge_tail_backend: EdgeTailBackend = "off",
        node_message_backend: NodeMessageBackend = "off",
    ) -> None:
        super().__init__()
        self.neighbor_scale = neighbor_scale
        self.edge_w1_recompute = edge_w1_recompute
        self.edge_tail_backend = edge_tail_backend
        self.node_message_backend = node_message_backend
        self.node_message = MessageUpdate(
            PackedEncoderProjection(node_width, edge_width, hidden_width),
            hidden_width,
            node_width,
            dropout,
            reduction_backend,
        )
        self.node_transition = ResidualTransition(
            node_width,
            dropout,
            transition_recompute,
        )
        self.edge_message = MessageUpdate(
            PackedEncoderProjection(node_width, edge_width, hidden_width),
            hidden_width,
            edge_width,
            dropout,
            reduction_backend,
            edge_mlp_backend,
            edge_norm_backend,
            edge_dropout_backend,
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        *,
        execution_path: EncoderExecutionPath = "block",
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.node_message_backend == "triton" and execution_path != "dense_training":
            reduced_hidden = self._fused_node_message(
                node_states, edge_states, neighbor_indices, neighbor_mask
            )
            if reduced_hidden is not None:
                assert neighbor_mask is not None
                return self._finish_reduced(
                    node_states,
                    edge_states,
                    neighbor_indices,
                    reduced_hidden,
                    neighbor_mask.sum(dim=-1, keepdim=True) / self.neighbor_scale,
                    residue_mask,
                    block_edge_projection=True,
                    allow_transition_recompute=allow_transition_recompute,
                )
            _warn_unmet_recompute_contract(self, "node_message_backend")

        if execution_path == "dense_training":
            return self.forward_dense_training(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
                allow_transition_recompute=allow_transition_recompute,
            )
        if execution_path == "zero_node_training":
            return self.forward_zero_node_training(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
                allow_transition_recompute=allow_transition_recompute,
            )
        if execution_path == "node_recompute":
            return self.forward_node_recompute(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
                allow_transition_recompute=allow_transition_recompute,
            )
        if execution_path == "zero_node_recompute":
            return self.forward_zero_node_training_recompute(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
                allow_transition_recompute=allow_transition_recompute,
            )
        if execution_path != "block":
            raise ValueError(f"unknown encoder execution path: {execution_path!r}")
        return self._forward_block(
            node_states,
            edge_states,
            neighbor_indices,
            residue_mask,
            neighbor_mask,
            allow_transition_recompute=allow_transition_recompute,
        )

    def _forward_block(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        *,
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_projection = cast(
            PackedEncoderProjection, self.node_message.input_projection
        )
        preactivation = input_projection.block(
            node_states, edge_states, node_states, neighbor_indices
        )
        return self._finish(
            node_states,
            edge_states,
            neighbor_indices,
            preactivation,
            residue_mask,
            neighbor_mask,
            block_edge_projection=True,
            allow_transition_recompute=allow_transition_recompute,
        )

    def forward_dense_training(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        *,
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_projection = cast(
            PackedEncoderProjection, self.node_message.input_projection
        )
        preactivation = input_projection.dense(
            node_states, edge_states, node_states, neighbor_indices
        )
        return self._finish(
            node_states,
            edge_states,
            neighbor_indices,
            preactivation,
            residue_mask,
            neighbor_mask,
            block_edge_projection=False,
            allow_transition_recompute=allow_transition_recompute,
        )

    def forward_zero_node_training(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        *,
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_projection = cast(
            PackedEncoderProjection, self.node_message.input_projection
        )
        preactivation = input_projection.edge_only(edge_states)
        return self._finish(
            node_states,
            edge_states,
            neighbor_indices,
            preactivation,
            residue_mask,
            neighbor_mask,
            block_edge_projection=True,
            allow_transition_recompute=allow_transition_recompute,
        )

    def forward_node_recompute(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        *,
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Checkpoint encoder node W1 plus its fused message reduction."""
        input_projection = cast(
            PackedEncoderProjection, self.node_message.input_projection
        )
        if not _supports_encoder_node_w1_recompute(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_mask,
            input_projection,
            self.node_message,
        ):
            if torch.is_grad_enabled():
                _warn_unmet_recompute_contract(self, "encoder_node_w1_recompute")
            return self._forward_block(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
                allow_transition_recompute=allow_transition_recompute,
            )

        assert neighbor_mask is not None
        replay_edges = edge_states.to(torch.bfloat16)

        def node_reduce(nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
            preactivation = input_projection.block(
                nodes,
                edges,
                nodes,
                neighbor_indices,
            )
            return self.node_message.reduced_hidden_features(
                preactivation,
                neighbor_mask,
                self.neighbor_scale,
            )

        reduced_hidden = torch.utils.checkpoint.checkpoint(
            node_reduce,
            node_states,
            replay_edges,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        bias_scale = neighbor_mask.sum(dim=-1, keepdim=True) / self.neighbor_scale
        return self._finish_reduced(
            node_states,
            edge_states,
            neighbor_indices,
            reduced_hidden,
            bias_scale,
            residue_mask,
            block_edge_projection=True,
            allow_transition_recompute=allow_transition_recompute,
        )

    def forward_zero_node_training_recompute(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        *,
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the smaller layer-zero edge-only W1 checkpoint boundary."""
        input_projection = cast(
            PackedEncoderProjection, self.node_message.input_projection
        )
        if not _supports_encoder_node_w1_recompute(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_mask,
            input_projection,
            self.node_message,
        ):
            if torch.is_grad_enabled():
                _warn_unmet_recompute_contract(self, "encoder_node_w1_recompute")
            return self.forward_zero_node_training(
                node_states,
                edge_states,
                neighbor_indices,
                residue_mask,
                neighbor_mask,
                allow_transition_recompute=allow_transition_recompute,
            )

        assert neighbor_mask is not None
        replay_edges = edge_states.to(torch.bfloat16)

        def node_reduce(edges: torch.Tensor) -> torch.Tensor:
            preactivation = input_projection.edge_only(edges)
            return self.node_message.reduced_hidden_features(
                preactivation,
                neighbor_mask,
                self.neighbor_scale,
            )

        reduced_hidden = torch.utils.checkpoint.checkpoint(
            node_reduce,
            replay_edges,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        bias_scale = neighbor_mask.sum(dim=-1, keepdim=True) / self.neighbor_scale
        return self._finish_reduced(
            node_states,
            edge_states,
            neighbor_indices,
            reduced_hidden,
            bias_scale,
            residue_mask,
            block_edge_projection=True,
            allow_transition_recompute=allow_transition_recompute,
        )

    def _finish(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        preactivation: torch.Tensor,
        residue_mask: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
        *,
        block_edge_projection: bool,
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if neighbor_mask is not None:
            reduced_hidden = self.node_message.reduced_hidden_features(
                preactivation, neighbor_mask, self.neighbor_scale
            )
            bias_scale: torch.Tensor | float = (
                neighbor_mask.sum(dim=-1, keepdim=True) / self.neighbor_scale
            )
        else:
            message_hidden = self.node_message.hidden_features(preactivation)
            reduced_hidden = message_hidden.sum(dim=-2) / self.neighbor_scale
            bias_scale = message_hidden.shape[-2] / self.neighbor_scale
        return self._finish_reduced(
            node_states,
            edge_states,
            neighbor_indices,
            reduced_hidden,
            bias_scale,
            residue_mask,
            block_edge_projection=block_edge_projection,
            allow_transition_recompute=allow_transition_recompute,
        )

    def _fused_node_message(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Reduce the node message through one fused kernel, or return ``None``.

        The two node-side blocks of the packed projection stay in PyTorch -- they are
        ``[B, T, 128]`` and cost nothing -- so only the edge-sized remainder moves into
        the kernel. Layer zero needs no special case: its node states are zero, so the
        query block reduces to the bias and the neighbour block to zero on their own,
        and the edge-only shortcut the separate-operation path used exists purely to
        skip two node-sized matmuls.
        """
        if neighbor_mask is None:
            return None
        projection = cast(PackedEncoderProjection, self.node_message.input_projection)
        node_width = projection.node_width
        edge_width = projection.edge_width
        weight = projection.weight
        hidden = self.node_message.hidden_projection
        query_projection = F.linear(
            node_states, weight[:, :node_width], projection.bias
        )
        neighbor_projection = F.linear(
            node_states, weight[:, node_width + edge_width :]
        )
        edge_weight = weight[:, node_width : node_width + edge_width]
        batch, length = node_states.shape[:2]
        flat_indices = (
            neighbor_indices
            if batch == 1
            else _flat_neighbor_indices(neighbor_indices, batch, length)
        )
        if not node_message_supported(
            edge_states,
            query_projection,
            neighbor_projection,
            flat_indices,
            edge_weight,
            hidden.weight,
            hidden.bias,
            neighbor_mask,
        ):
            return None
        return node_message_reduce(
            edge_states,
            query_projection,
            neighbor_projection,
            flat_indices,
            edge_weight,
            hidden.weight,
            hidden.bias,
            neighbor_mask,
            self.neighbor_scale,
        )

    def _fused_edge_tail(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        edge_projection: PackedEncoderProjection,
    ) -> torch.Tensor | None:
        """Run the whole edge update through one fused kernel, or return ``None``.

        The two node-side blocks of the packed projection stay in PyTorch: they are
        ``[B, T, 128]``, cost nothing, and keep their own weight gradients on the
        ordinary autograd path. Only the edge-sized remainder -- the edge block of
        W1, both MLP projections, dropout, the residual add and LayerNorm -- moves
        into the kernel, which is where every edge-sized allocation came from.

        ``self.edge_message.norm`` and ``.dropout`` contribute their parameters and
        probability but are not called, so module hooks installed on them do not
        fire on this path.
        """
        node_width = edge_projection.node_width
        edge_width = edge_projection.edge_width
        weight = edge_projection.weight
        edge_weight = weight[:, node_width : node_width + edge_width]
        dropout = cast(nn.Dropout, self.edge_message.dropout)
        norm = self.edge_message.norm
        query_projection = F.linear(
            node_states, weight[:, :node_width], edge_projection.bias
        )
        neighbor_projection = F.linear(
            node_states, weight[:, node_width + edge_width :]
        )
        batch, length = node_states.shape[:2]
        flat_indices = (
            neighbor_indices
            if batch == 1
            else _flat_neighbor_indices(neighbor_indices, batch, length)
        )
        if not edge_tail_supported(
            edge_states,
            query_projection,
            neighbor_projection,
            flat_indices,
            edge_weight,
            self.edge_message.hidden_projection.weight,
            self.edge_message.hidden_projection.bias,
            self.edge_message.output_projection.weight,
            self.edge_message.output_projection.bias,
            norm.weight,
            norm.bias,
        ):
            return None
        probability = dropout.p if self.training else 0.0
        if probability > 0.0:
            seed = torch.randint(
                0,
                2**31 - 1,
                (1,),
                device=edge_states.device,
                dtype=torch.int64,
            )
        else:
            seed = torch.zeros(1, device=edge_states.device, dtype=torch.int64)
        return edge_tail_update(
            edge_states,
            query_projection,
            neighbor_projection,
            flat_indices,
            edge_weight,
            self.edge_message.hidden_projection.weight,
            self.edge_message.hidden_projection.bias,
            self.edge_message.output_projection.weight,
            self.edge_message.output_projection.bias,
            norm.weight,
            norm.bias,
            seed,
            norm.eps,
            probability,
        )

    def _finish_reduced(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        reduced_hidden: torch.Tensor,
        bias_scale: torch.Tensor | float,
        residue_mask: torch.Tensor | None,
        *,
        block_edge_projection: bool,
        allow_transition_recompute: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finish a layer after the node message has been neighbor-reduced."""
        node_states = self.node_message.apply_reduced_hidden(
            node_states,
            reduced_hidden,
            bias_scale,
        )
        node_states = self.node_transition(
            node_states,
            allow_recompute=allow_transition_recompute,
        )
        if residue_mask is not None:
            node_states = residue_mask.unsqueeze(-1) * node_states

        edge_projection = cast(
            PackedEncoderProjection, self.edge_message.input_projection
        )
        if self.edge_tail_backend == "triton" and block_edge_projection:
            fused_edges = self._fused_edge_tail(
                node_states,
                edge_states,
                neighbor_indices,
                edge_projection,
            )
            if fused_edges is not None:
                return node_states, fused_edges
            _warn_unmet_recompute_contract(self, "edge_tail_backend")

        requested_edge_w1 = (
            self.edge_w1_recompute == "checkpoint"
            and self.edge_message.edge_mlp_backend == "triton_memory"
            and block_edge_projection
            and torch.is_grad_enabled()
        )
        checkpoint_edge_w1 = requested_edge_w1 and _supports_edge_w1_recompute(
            node_states,
            edge_states,
            edge_projection,
            self.edge_message,
        )
        if requested_edge_w1 and not checkpoint_edge_w1:
            _warn_unmet_recompute_contract(self, "edge_w1_recompute")
        if checkpoint_edge_w1:
            w1_nodes = node_states.to(torch.bfloat16)
            w1_edges = edge_states.to(torch.bfloat16)

            def edge_update(nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
                edge_preactivation = edge_projection.block(
                    nodes,
                    edges,
                    nodes,
                    neighbor_indices,
                )
                return self.edge_message.edge_update_from_preactivation(
                    edge_preactivation
                )

            update = torch.utils.checkpoint.checkpoint(
                edge_update,
                w1_nodes,
                w1_edges,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            edge_states = self.edge_message.apply_edgewise_update(edge_states, update)
            return node_states, edge_states

        if block_edge_projection:
            edge_preactivation = edge_projection.block(
                node_states, edge_states, node_states, neighbor_indices
            )
        else:
            edge_preactivation = edge_projection.dense(
                node_states, edge_states, node_states, neighbor_indices
            )
        edge_states = self.edge_message.apply_edgewise_preactivation(
            edge_states,
            edge_preactivation,
        )
        return node_states, edge_states


class DecoderLayer(nn.Module):
    """Teacher-forced node update for one MPNN decoder layer."""

    def __init__(
        self,
        node_width: int,
        edge_width: int,
        hidden_width: int,
        dropout: float,
        neighbor_scale: int,
        reduction_backend: MessageBackend = "auto",
        transition_recompute: TransitionRecompute = "off",
        node_w1_recompute: DecoderNodeW1Recompute = "off",
    ) -> None:
        super().__init__()
        self.neighbor_scale = neighbor_scale
        self.node_w1_recompute = node_w1_recompute
        self.node_message = MessageUpdate(
            PackedDecoderProjection(node_width, edge_width, hidden_width),
            hidden_width,
            node_width,
            dropout,
            reduction_backend,
        )
        self.node_transition = ResidualTransition(
            node_width,
            dropout,
            transition_recompute,
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        sequence_features: torch.Tensor,
        encoder_nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
        future_mask: torch.Tensor,
        past_mask: torch.Tensor,
        residue_mask: torch.Tensor | None,
        edge_mask: torch.Tensor | None = None,
        *,
        allow_transition_recompute: bool = True,
    ) -> torch.Tensor:
        input_projection = cast(
            PackedDecoderProjection, self.node_message.input_projection
        )
        if self.node_w1_recompute == "checkpoint":
            if _supports_decoder_node_w1_recompute(
                node_states,
                edge_states,
                neighbor_indices,
                edge_mask,
                input_projection,
                self.node_message,
            ):
                assert edge_mask is not None

                def node_reduce(
                    nodes: torch.Tensor,
                    edges: torch.Tensor,
                    sequence: torch.Tensor,
                    encoder: torch.Tensor,
                    future: torch.Tensor,
                    past: torch.Tensor,
                ) -> torch.Tensor:
                    preactivation = input_projection.block(
                        nodes,
                        edges,
                        sequence,
                        encoder,
                        neighbor_indices,
                        future,
                        past,
                    )
                    return self.node_message.reduced_hidden_features(
                        preactivation,
                        edge_mask,
                        self.neighbor_scale,
                    )

                # The replayed region is deterministic: dropout, the residual add
                # and both norms stay outside, so no RNG state is captured. The
                # retained operands are node- and mask-sized apart from the edge
                # state, which the encoder output already keeps alive.
                reduced_hidden = torch.utils.checkpoint.checkpoint(
                    node_reduce,
                    node_states,
                    edge_states,
                    sequence_features,
                    encoder_nodes,
                    future_mask,
                    past_mask,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                bias_scale = edge_mask.sum(dim=-1, keepdim=True) / self.neighbor_scale
                return self._finish_reduced(
                    node_states,
                    reduced_hidden,
                    bias_scale,
                    residue_mask,
                    allow_transition_recompute=allow_transition_recompute,
                )
            if torch.is_grad_enabled():
                _warn_unmet_recompute_contract(self, "decoder_node_w1_recompute")

        preactivation = input_projection.block(
            node_states,
            edge_states,
            sequence_features,
            encoder_nodes,
            neighbor_indices,
            future_mask,
            past_mask,
        )
        return self._finish(
            node_states,
            preactivation,
            residue_mask,
            edge_mask,
            allow_transition_recompute=allow_transition_recompute,
        )

    def forward_dense_training(
        self,
        node_states: torch.Tensor,
        edge_context: torch.Tensor,
        sequence_context: torch.Tensor,
        encoder_context: torch.Tensor,
        neighbor_indices: torch.Tensor,
        past_mask: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        *,
        allow_transition_recompute: bool = True,
    ) -> torch.Tensor:
        input_projection = cast(
            PackedDecoderProjection, self.node_message.input_projection
        )
        preactivation = input_projection.dense(
            node_states,
            edge_context,
            sequence_context,
            encoder_context,
            neighbor_indices,
            past_mask,
        )
        return self._finish(
            node_states,
            preactivation,
            residue_mask,
            edge_mask,
            allow_transition_recompute=allow_transition_recompute,
        )

    def _finish(
        self,
        node_states: torch.Tensor,
        preactivation: torch.Tensor,
        residue_mask: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
        *,
        allow_transition_recompute: bool = True,
    ) -> torch.Tensor:
        if edge_mask is not None:
            reduced_hidden = self.node_message.reduced_hidden_features(
                preactivation, edge_mask, self.neighbor_scale
            )
            bias_scale: torch.Tensor | float = (
                edge_mask.sum(dim=-1, keepdim=True) / self.neighbor_scale
            )
        else:
            message_hidden = self.node_message.hidden_features(preactivation)
            reduced_hidden = message_hidden.sum(dim=-2) / self.neighbor_scale
            bias_scale = message_hidden.shape[-2] / self.neighbor_scale
        return self._finish_reduced(
            node_states,
            reduced_hidden,
            bias_scale,
            residue_mask,
            allow_transition_recompute=allow_transition_recompute,
        )

    def _finish_reduced(
        self,
        node_states: torch.Tensor,
        reduced_hidden: torch.Tensor,
        bias_scale: torch.Tensor | float,
        residue_mask: torch.Tensor | None,
        *,
        allow_transition_recompute: bool = True,
    ) -> torch.Tensor:
        """Finish a decoder layer after the node message has been reduced."""
        node_states = self.node_message.apply_reduced_hidden(
            node_states,
            reduced_hidden,
            bias_scale,
        )
        node_states = self.node_transition(
            node_states,
            allow_recompute=allow_transition_recompute,
        )
        if residue_mask is not None:
            node_states = residue_mask.unsqueeze(-1) * node_states
        return node_states
