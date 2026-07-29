"""Production ProteinMPNN model with semantic module and checkpoint structure."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from miniworld_kernels.kernels.mpnn_edge_dropout import EdgeDropoutBackend
from miniworld_kernels.kernels.mpnn_edge_layernorm import EdgeNormBackend
from miniworld_kernels.kernels.mpnn_relative_position import RelativePositionBackend
from miniworld_kernels.kernels.mpnn_edge_mlp import EdgeMLPBackend
from miniworld_kernels.kernels.mpnn_edge_tail import EdgeTailBackend
from miniworld_kernels.kernels.mpnn_message import MessageBackend
from miniworld_kernels.kernels.mpnn_node_message import NodeMessageBackend

from ._functional import gather_neighbors, lecun_normal_
from .features import BackboneFeatures, FeatureBackend, KNNBackend
from .layers import (
    DEFAULT_BLOCK_LINEAR_MIN_EDGES,
    DecoderLayer,
    DecoderNodeW1Recompute,
    EdgeW1Recompute,
    EncoderExecutionPath,
    EncoderNodeW1Recompute,
    EncoderLayer,
    TransitionRecompute,
)
from .masking import build_decoding_masks


@dataclass(frozen=True)
class ProteinMPNNConfig:
    """Architecture and execution policy for :class:`ProteinMPNN`."""

    sequence_vocabulary: int = 21
    output_vocabulary: int = 21
    node_width: int = 128
    edge_width: int = 128
    hidden_width: int = 128
    encoder_depth: int = 3
    decoder_depth: int = 3
    k_neighbors: int = 32
    coordinate_noise: float = 0.5
    dropout: float = 0.1
    block_linear_min_edges: int = DEFAULT_BLOCK_LINEAR_MIN_EDGES
    message_backend: MessageBackend = "auto"
    edge_mlp_backend: EdgeMLPBackend = "auto"
    edge_norm_backend: EdgeNormBackend = "auto"
    feature_backend: FeatureBackend = "auto"
    relative_position_backend: RelativePositionBackend = "off"
    knn_backend: KNNBackend = "cdist"
    knn_query_chunk: int = 2048
    knn_cutoff: float = 16.0
    edge_w1_recompute: EdgeW1Recompute = "off"
    encoder_node_w1_recompute: EncoderNodeW1Recompute = "off"
    decoder_node_w1_recompute: DecoderNodeW1Recompute = "off"
    transition_recompute: TransitionRecompute = "off"
    edge_dropout_backend: EdgeDropoutBackend = "auto"
    edge_tail_backend: EdgeTailBackend = "off"
    node_message_backend: NodeMessageBackend = "off"

    def __post_init__(self) -> None:
        positive = {
            "sequence_vocabulary": self.sequence_vocabulary,
            "output_vocabulary": self.output_vocabulary,
            "node_width": self.node_width,
            "edge_width": self.edge_width,
            "hidden_width": self.hidden_width,
            "encoder_depth": self.encoder_depth,
            "decoder_depth": self.decoder_depth,
            "k_neighbors": self.k_neighbors,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"ProteinMPNN dimensions must be positive: {invalid}")
        if self.coordinate_noise < 0:
            raise ValueError("coordinate_noise must be non-negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.block_linear_min_edges < 0:
            raise ValueError("block_linear_min_edges must be non-negative")
        if self.message_backend not in {
            "auto",
            "pytorch",
            "triton",
            "triton_compute",
            "triton_memory",
        }:
            raise ValueError(
                "message_backend must be one of 'auto', 'pytorch', 'triton', "
                "'triton_compute', or 'triton_memory'"
            )
        if self.edge_mlp_backend not in {
            "auto",
            "pytorch",
            "triton_compute",
            "triton_memory",
        }:
            raise ValueError(
                "edge_mlp_backend must be one of 'auto', 'pytorch', "
                "'triton_compute', or 'triton_memory'"
            )
        if self.edge_norm_backend not in {"auto", "pytorch", "memory"}:
            raise ValueError(
                "edge_norm_backend must be one of 'auto', 'pytorch', or 'memory'"
            )
        if self.edge_dropout_backend not in {"auto", "pytorch", "bitpack"}:
            raise ValueError(
                "edge_dropout_backend must be one of 'auto', 'pytorch', or 'bitpack'"
            )
        if self.feature_backend not in {"auto", "pytorch", "recompute"}:
            raise ValueError(
                "feature_backend must be one of 'auto', 'pytorch', or 'recompute'"
            )
        if self.relative_position_backend not in {"off", "index_add", "triton"}:
            raise ValueError(
                "relative_position_backend must be one of 'off', 'index_add', "
                "or 'triton'"
            )
        if self.knn_backend not in {"cdist", "chunked", "grid_cutoff", "segment"}:
            raise ValueError(
                "knn_backend must be one of 'cdist', 'chunked', 'grid_cutoff', "
                "or 'segment'"
            )
        if self.knn_query_chunk <= 0:
            raise ValueError("knn_query_chunk must be positive")
        if self.knn_cutoff <= 0:
            raise ValueError("knn_cutoff must be positive")
        if self.edge_w1_recompute not in {"off", "checkpoint"}:
            raise ValueError("edge_w1_recompute must be one of 'off' or 'checkpoint'")
        if (
            self.edge_w1_recompute == "checkpoint"
            and self.edge_mlp_backend != "triton_memory"
        ):
            raise ValueError(
                "edge_w1_recompute='checkpoint' requires "
                "edge_mlp_backend='triton_memory'"
            )
        if self.encoder_node_w1_recompute not in {"off", "checkpoint"}:
            raise ValueError(
                "encoder_node_w1_recompute must be one of 'off' or 'checkpoint'"
            )
        if (
            self.encoder_node_w1_recompute == "checkpoint"
            and self.message_backend != "triton_memory"
        ):
            raise ValueError(
                "encoder_node_w1_recompute='checkpoint' requires "
                "message_backend='triton_memory'"
            )
        if self.decoder_node_w1_recompute not in {"off", "checkpoint"}:
            raise ValueError(
                "decoder_node_w1_recompute must be one of 'off' or 'checkpoint'"
            )
        if (
            self.decoder_node_w1_recompute == "checkpoint"
            and self.message_backend != "triton_memory"
        ):
            raise ValueError(
                "decoder_node_w1_recompute='checkpoint' requires "
                "message_backend='triton_memory'"
            )
        if self.transition_recompute not in {"off", "update"}:
            raise ValueError("transition_recompute must be one of 'off' or 'update'")
        if self.edge_tail_backend not in {"off", "triton"}:
            raise ValueError("edge_tail_backend must be one of 'off' or 'triton'")
        if self.node_message_backend not in {"off", "triton"}:
            raise ValueError("node_message_backend must be one of 'off' or 'triton'")
        if (
            self.node_message_backend == "triton"
            and self.encoder_node_w1_recompute != "off"
        ):
            # The fused node message replays its whole chain in backward, so a
            # checkpoint around part of it would be dead configuration that silently
            # never engages.
            raise ValueError(
                "node_message_backend='triton' subsumes encoder_node_w1_recompute; "
                "set encoder_node_w1_recompute='off'"
            )
        if self.edge_tail_backend == "triton" and self.edge_w1_recompute != "off":
            # The fused tail already replays the whole edge update in backward, so a
            # second checkpoint around part of it would be dead configuration that
            # silently never engages.
            raise ValueError(
                "edge_tail_backend='triton' subsumes edge_w1_recompute; set "
                "edge_w1_recompute='off'"
            )


@dataclass(frozen=True)
class EncodedMPNN:
    """Reusable backbone-dependent state for parallel sequence scoring."""

    node_states: torch.Tensor
    edge_states: torch.Tensor
    neighbor_indices: torch.Tensor
    residue_mask: torch.Tensor
    _owner_id: int
    _module_training: bool
    _training_messages: bool
    edge_mask: torch.Tensor | None = None


class MPNNEncoder(nn.Module):
    """Stack of semantic node/edge encoder layers."""

    def __init__(self, config: ProteinMPNNConfig) -> None:
        super().__init__()
        self.encoder_node_w1_recompute = config.encoder_node_w1_recompute
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.node_width,
                    config.edge_width,
                    config.hidden_width,
                    config.dropout,
                    config.k_neighbors,
                    config.message_backend,
                    config.edge_mlp_backend,
                    config.edge_w1_recompute,
                    config.edge_norm_backend,
                    config.transition_recompute,
                    config.edge_dropout_backend,
                    config.edge_tail_backend,
                    config.node_message_backend,
                )
                for _ in range(config.encoder_depth)
            ]
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
        *,
        training_messages: bool,
        block_linear: bool,
        checkpoint_layers: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for index, layer in enumerate(self.layers):
            checkpoint_node_w1 = (
                training_messages
                and block_linear
                and not checkpoint_layers
                and self.encoder_node_w1_recompute == "checkpoint"
            )
            if checkpoint_node_w1 and index == 0:
                execution_path: EncoderExecutionPath = "zero_node_recompute"
            elif checkpoint_node_w1:
                execution_path = "node_recompute"
            elif training_messages and block_linear and index == 0:
                execution_path = "zero_node_training"
            elif training_messages and not block_linear:
                execution_path = "dense_training"
            else:
                execution_path = "block"

            if checkpoint_layers:
                node_states, edge_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    node_states,
                    edge_states,
                    neighbor_indices,
                    residue_mask,
                    neighbor_mask,
                    use_reentrant=False,
                    execution_path=execution_path,
                    allow_transition_recompute=False,
                )
            else:
                node_states, edge_states = layer(
                    node_states,
                    edge_states,
                    neighbor_indices,
                    residue_mask,
                    neighbor_mask,
                    execution_path=execution_path,
                )
        return node_states, edge_states


class MPNNDecoder(nn.Module):
    """Stack of teacher-forced decoder layers."""

    def __init__(self, config: ProteinMPNNConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config.node_width,
                    config.edge_width,
                    config.hidden_width,
                    config.dropout,
                    config.k_neighbors,
                    config.message_backend,
                    config.transition_recompute,
                    config.decoder_node_w1_recompute,
                )
                for _ in range(config.decoder_depth)
            ]
        )

    def forward(
        self,
        encoder_nodes: torch.Tensor,
        edge_states: torch.Tensor,
        sequence_features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor,
        future_mask: torch.Tensor,
        past_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        *,
        training_messages: bool,
        block_linear: bool,
        checkpoint_layers: bool,
    ) -> torch.Tensor:
        node_states = encoder_nodes
        if training_messages and not block_linear:
            edge_context = (future_mask + past_mask) * edge_states
            sequence_context = past_mask * gather_neighbors(
                sequence_features, neighbor_indices
            )
            encoder_context = future_mask * gather_neighbors(
                encoder_nodes, neighbor_indices
            )
            for layer in self.layers:
                if checkpoint_layers:
                    node_states = torch.utils.checkpoint.checkpoint(
                        layer.forward_dense_training,
                        node_states,
                        edge_context,
                        sequence_context,
                        encoder_context,
                        neighbor_indices,
                        past_mask,
                        residue_mask,
                        edge_mask,
                        use_reentrant=False,
                        allow_transition_recompute=False,
                    )
                else:
                    node_states = layer.forward_dense_training(
                        node_states,
                        edge_context,
                        sequence_context,
                        encoder_context,
                        neighbor_indices,
                        past_mask,
                        residue_mask,
                        edge_mask,
                    )
            return node_states

        # Every layer's packed projection feeds this same tensor to an autocast
        # ``F.linear``, which casts it to the autocast dtype. Autocast caches that
        # cast only for leaves, so three layers would each retain a distinct
        # edge-sized copy of the identical value. Casting once collapses them into
        # one saved tensor without changing any result.
        if torch.is_autocast_enabled("cuda") and edge_states.is_cuda:
            autocast_dtype = torch.get_autocast_dtype("cuda")
            if edge_states.dtype != autocast_dtype:
                edge_states = edge_states.to(autocast_dtype)

        for layer in self.layers:
            if checkpoint_layers:
                node_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    node_states,
                    edge_states,
                    sequence_features,
                    encoder_nodes,
                    neighbor_indices,
                    future_mask,
                    past_mask,
                    residue_mask,
                    edge_mask,
                    use_reentrant=False,
                    allow_transition_recompute=False,
                )
            else:
                node_states = layer(
                    node_states,
                    edge_states,
                    sequence_features,
                    encoder_nodes,
                    neighbor_indices,
                    future_mask,
                    past_mask,
                    residue_mask,
                    edge_mask,
                )
        return node_states


class ProteinMPNN(nn.Module):
    """Clean production MPNN preserving the frozen reference mathematics."""

    def __init__(self, config: ProteinMPNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProteinMPNNConfig()
        config = self.config

        self.backbone_features = BackboneFeatures(
            config.edge_width,
            k_neighbors=config.k_neighbors,
            coordinate_noise=config.coordinate_noise,
            feature_backend=config.feature_backend,
            relative_position_backend=config.relative_position_backend,
            edge_norm_backend=config.edge_norm_backend,
            knn_backend=config.knn_backend,
            knn_query_chunk=config.knn_query_chunk,
            knn_cutoff=config.knn_cutoff,
        )
        self.edge_input_projection = nn.Linear(
            config.edge_width, config.edge_width, bias=True
        )
        self.sequence_embedding = nn.Embedding(
            config.sequence_vocabulary, config.node_width
        )
        self.encoder = MPNNEncoder(config)
        self.decoder = MPNNDecoder(config)
        self.output_projection = nn.Linear(
            config.node_width, config.output_vocabulary, bias=True
        )

        lecun_normal_(self.edge_input_projection)
        lecun_normal_(self.sequence_embedding)
        nn.init.zeros_(self.edge_input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _uses_block_linear(self, edge_states: torch.Tensor) -> bool:
        total_edges = edge_states.shape[0] * edge_states.shape[1] * edge_states.shape[2]
        return total_edges >= self.config.block_linear_min_edges

    def encode_backbone(
        self,
        backbone: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        segment_lengths: torch.Tensor | None = None,
        *,
        checkpoint_layers: bool = False,
    ) -> EncodedMPNN:
        """Encode geometry once for reuse across parallel sequence scores."""
        batch, length = backbone.shape[:2]
        graph = self.backbone_features.build_graph(
            backbone,
            residue_mask,
            residue_index,
            chain_index,
            segment_lengths,
        )
        edge_states = self.edge_input_projection(graph.edge_features)
        edge_states = torch.where(
            graph.edge_mask.bool().unsqueeze(-1),
            edge_states,
            torch.zeros_like(edge_states),
        )
        node_states = edge_states.new_zeros((batch, length, self.config.node_width))
        training_messages = self.training and torch.is_grad_enabled()
        node_states, edge_states = self.encoder(
            node_states,
            edge_states,
            graph.neighbor_indices,
            residue_mask,
            graph.edge_mask,
            training_messages=training_messages,
            block_linear=self._uses_block_linear(edge_states),
            checkpoint_layers=checkpoint_layers,
        )
        return EncodedMPNN(
            node_states=node_states,
            edge_states=edge_states,
            neighbor_indices=graph.neighbor_indices,
            residue_mask=residue_mask,
            edge_mask=graph.edge_mask,
            _owner_id=id(self),
            _module_training=self.training,
            _training_messages=training_messages,
        )

    def score_sequence(
        self,
        encoded: EncodedMPNN,
        sequence: torch.Tensor,
        decoding_order: torch.Tensor,
        patch_index: torch.Tensor,
        *,
        fixed_decoding_order_length: int | torch.Tensor = 0,
        checkpoint_layers: bool = False,
        return_log_prob: bool = False,
    ) -> torch.Tensor:
        """Score every sequence position in parallel from an encoded backbone."""
        training_messages = self.training and torch.is_grad_enabled()
        if encoded._owner_id != id(self):
            raise ValueError("encoded backbone belongs to a different model instance")
        if (
            encoded._module_training != self.training
            or encoded._training_messages != training_messages
        ):
            raise ValueError("encoded backbone was created in a different model mode")

        edge_mask = encoded.edge_mask
        if edge_mask is None:
            edge_mask = gather_neighbors(
                encoded.residue_mask.unsqueeze(-1), encoded.neighbor_indices
            ).squeeze(-1)
            edge_mask = encoded.residue_mask.unsqueeze(-1) * edge_mask
        sequence_features = self.sequence_embedding(sequence)
        future_mask, past_mask = build_decoding_masks(
            encoded.neighbor_indices,
            encoded.residue_mask,
            decoding_order,
            patch_index,
            fixed_decoding_order_length,
            edge_mask,
        )
        node_states = self.decoder(
            encoded.node_states,
            encoded.edge_states,
            sequence_features,
            encoded.neighbor_indices,
            encoded.residue_mask,
            future_mask,
            past_mask,
            edge_mask,
            training_messages=training_messages,
            block_linear=self._uses_block_linear(encoded.edge_states),
            checkpoint_layers=checkpoint_layers,
        )
        logits = self.output_projection(node_states)
        if return_log_prob:
            return F.log_softmax(logits, dim=-1)
        return logits.permute(0, 2, 1)

    def forward(
        self,
        backbone: torch.Tensor,
        sequence: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        decoding_order: torch.Tensor,
        patch_index: torch.Tensor,
        segment_lengths: torch.Tensor | None = None,
        *,
        fixed_decoding_order_length: int | torch.Tensor = 0,
        checkpoint_layers: bool = False,
        return_log_prob: bool = False,
    ) -> torch.Tensor:
        encoded = self.encode_backbone(
            backbone,
            residue_mask,
            residue_index,
            chain_index,
            segment_lengths,
            checkpoint_layers=checkpoint_layers,
        )
        return self.score_sequence(
            encoded,
            sequence,
            decoding_order,
            patch_index,
            fixed_decoding_order_length=fixed_decoding_order_length,
            checkpoint_layers=checkpoint_layers,
            return_log_prob=return_log_prob,
        )
