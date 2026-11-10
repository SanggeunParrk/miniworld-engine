"""ProteinMPNN: backbone KNN features, a node/edge message-passing encoder, a teacher-forced decoder (with its decoding masks) and the
output projection.

Each module is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from miniworld_engine.modules.mpnn._functional import lecun_normal_
from miniworld_engine.modules.mpnn.features import BackboneFeatures
from miniworld_engine.modules.mpnn.layers import DecoderLayer, EncoderLayer
from miniworld_engine.modules.primitives import BackendModule, key_mask, random_input


def build_decoding_masks(
    neighbor_indices: torch.Tensor,
    residue_mask: torch.Tensor,
    decoding_order: torch.Tensor,
    patch_index: torch.Tensor,
    fixed_decoding_order_length: int | torch.Tensor = 0,
    edge_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return future/backbone and past/sequence masks for each graph edge."""
    batch, length = decoding_order.shape
    decode_rank = torch.empty_like(decoding_order)
    steps = torch.arange(length, device=decoding_order.device).expand(batch, -1)
    decode_rank.scatter_(1, decoding_order, steps)
    decode_group = torch.gather(patch_index, 1, decode_rank)
    neighbor_group = torch.gather(
        decode_group.unsqueeze(1).expand(-1, length, -1),
        2,
        neighbor_indices,
    )
    past = (decode_group.unsqueeze(2) > neighbor_group).unsqueeze(-1)
    if isinstance(fixed_decoding_order_length, torch.Tensor):
        fixed_lengths = fixed_decoding_order_length.to(
            device=decode_rank.device, dtype=decode_rank.dtype
        )
        if fixed_lengths.ndim == 0:
            fixed_lengths = fixed_lengths.expand(batch)
        if fixed_lengths.shape != (batch,):
            raise ValueError(
                "fixed_decoding_order_length must be a scalar or have shape [batch]"
            )
        motif = decode_rank < fixed_lengths.unsqueeze(1)
        has_fixed_positions = True
    else:
        motif = decode_rank < fixed_decoding_order_length
        has_fixed_positions = fixed_decoding_order_length != 0

    if has_fixed_positions:
        motif_neighbor = torch.gather(
            motif.unsqueeze(1).expand(-1, length, -1), 2, neighbor_indices
        )
        past = past | motif_neighbor.unsqueeze(-1)
        query_index = torch.arange(length, device=neighbor_indices.device).view(
            1, length, 1
        )
        motif_self_edge = (neighbor_indices == query_index) & motif.unsqueeze(-1)
        past = past & ~motif_self_edge.unsqueeze(-1)

    if edge_mask is None:
        valid_edge = residue_mask.reshape(-1, length, 1, 1)
    else:
        if edge_mask.shape != neighbor_indices.shape:
            raise ValueError("edge_mask must have the same shape as neighbor_indices")
        valid_edge = edge_mask.unsqueeze(-1)
    past_mask = valid_edge * past
    future_mask = valid_edge * (~past)
    return future_mask, past_mask


@dataclass(frozen=True)
class ProteinMPNNConfig:
    """Architecture of :class:`ProteinMPNN`."""

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
    #: The `"triton"` backend: the encoder layers run the fused node-message and edge-tail kernels (the unfused children otherwise).
    fused_encoder: bool = True
    #: The `"triton"` backend, training: kernels recompute their GEMM outputs in backward instead of saving them (less memory, more time).
    replay_in_backward: bool = False

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


@dataclass(frozen=True)
class EncodedMPNN:
    """Reusable backbone-dependent state for parallel sequence scoring."""

    node_states: torch.Tensor
    edge_states: torch.Tensor
    neighbor_indices: torch.Tensor
    residue_mask: torch.Tensor
    edge_mask: torch.Tensor
    _owner_id: int
    _module_training: bool
    _training_messages: bool


class MPNNEncoder(BackendModule):
    """Stack of node/edge encoder layers."""

    backends = ("pytorch",)

    def __init__(self, config: ProteinMPNNConfig, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.node_width,
                    config.edge_width,
                    config.hidden_width,
                    config.dropout,
                    config.k_neighbors,
                    fused=config.fused_encoder,
                    replay_in_backward=config.replay_in_backward,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
        *,
        training_messages: bool,
        checkpoint_layers: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for index, layer in enumerate(self.layers):
            # Layer 0 sees zero node states: in training its node message takes only the edge block of the projection.
            zero_nodes = training_messages and index == 0
            if checkpoint_layers:
                node_states, edge_states = torch.utils.checkpoint.checkpoint(
                    layer, node_states, edge_states, neighbor_indices, residue_mask, neighbor_mask,
                    use_reentrant=False, zero_nodes=zero_nodes,
                )
            else:
                node_states, edge_states = layer(
                    node_states, edge_states, neighbor_indices, residue_mask, neighbor_mask, zero_nodes=zero_nodes
                )
        return node_states, edge_states


class MPNNDecoder(BackendModule):
    """Stack of teacher-forced decoder layers."""

    backends = ("pytorch",)

    def __init__(self, config: ProteinMPNNConfig, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config.node_width,
                    config.edge_width,
                    config.hidden_width,
                    config.dropout,
                    config.k_neighbors,
                    replay_in_backward=config.replay_in_backward,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.set_backend(backend)

    def _forward_pytorch(
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
        checkpoint_layers: bool,
    ) -> torch.Tensor:
        node_states = encoder_nodes
        # Every layer's packed projection feeds this same tensor to an autocast ``F.linear``, which casts it to the autocast dtype.
        # Autocast caches that cast only for leaves, so three layers would each retain a distinct edge-sized copy of the identical
        # value. Casting once collapses them into one saved tensor without changing any result.
        if torch.is_autocast_enabled("cuda") and edge_states.is_cuda:
            autocast_dtype = torch.get_autocast_dtype("cuda")
            if edge_states.dtype != autocast_dtype:
                edge_states = edge_states.to(autocast_dtype)

        for layer in self.layers:
            args = (node_states, edge_states, sequence_features, encoder_nodes, neighbor_indices, future_mask, past_mask,
                    residue_mask, edge_mask)
            node_states = torch.utils.checkpoint.checkpoint(layer, *args, use_reentrant=False) if checkpoint_layers else layer(*args)
        return node_states


class ProteinMPNN(BackendModule):
    """ProteinMPNN preserving the frozen reference mathematics.

    No ``"anthropic"`` backend: the release's proteinmpnn/ kit binds no kernel for the model's math (it drives the stock dauparas package in
    fp32 under no_grad, neighbour scale 30 where this module divides by ``k_neighbors``; its one Triton kernel is the sampler's multinomial
    draw), and cuEquivariance has no message-passing op.

    ``"triton"``: the composition with every child that has a ``"triton"`` backend set to it. Features: the relative-position embedding
    (:class:`~miniworld_engine.modules.mpnn.features.RelativePositionEmbedding`, its bucket-reduction backward). Encoder layers
    (:class:`~miniworld_engine.modules.mpnn.layers.EncoderLayer`): the fused node-message and edge-tail kernels (``config.fused_encoder``,
    the default), or the unfused children -- the node message on the message-reduction kernels, the edge message on the edge-MLP kernels
    with the packed-mask edge dropout and the memory edge LayerNorm. Decoder layers: the node messages on the message-reduction kernels.
    ``config.replay_in_backward`` picks the recompute kernels over the saved-activation ones in training. Packed projections,
    transitions, the other norms and the output projection are the PyTorch statements. Each child states its range (bf16, widths 128,
    48 neighbours) and raises ``BackendUnsupported`` outside it; CUDA has no kernel of these families (``kernels/mpnn_*`` hold Triton
    only), so no ``"cuda"`` backend."""
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "triton")

    def __init__(self, config: ProteinMPNNConfig | None = None, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.config = config or ProteinMPNNConfig()
        config = self.config

        self.backbone_features = BackboneFeatures(
            config.edge_width, k_neighbors=config.k_neighbors, coordinate_noise=config.coordinate_noise
        )
        self.edge_input_projection = nn.Linear(config.edge_width, config.edge_width, bias=True)
        self.sequence_embedding = nn.Embedding(config.sequence_vocabulary, config.node_width)
        self.encoder = MPNNEncoder(config)
        self.decoder = MPNNDecoder(config)
        self.output_projection = nn.Linear(config.node_width, config.output_vocabulary, bias=True)

        lecun_normal_(self.edge_input_projection)
        lecun_normal_(self.sequence_embedding)
        nn.init.zeros_(self.edge_input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.set_backend(backend)

    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        for sub in self.modules():
            if sub is not self and isinstance(sub, BackendModule):
                sub.set_backend(backend if backend in sub.backends else "pytorch")
        return self

    def _supports_triton(self, backbone: torch.Tensor, *args, **kwargs) -> str | None:
        if backbone.device.type != "cuda":
            return "the Triton kernels run on CUDA tensors only"
        c = self.config
        if (c.node_width, c.edge_width, c.hidden_width) != (128, 128, 128):
            return "the message kernels are built for node / edge / hidden width 128"
        if c.k_neighbors != 48:
            return f"k_neighbors {c.k_neighbors}: the message kernels are built for 48 neighbours"
        return None

    def _forward_triton(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_pytorch(*args, **kwargs)   # the composition; the message children run their "triton" backends

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
        graph = self.backbone_features(backbone, residue_mask, residue_index, chain_index, segment_lengths)
        edge_states = self.edge_input_projection(graph.edge_features)
        edge_states = torch.where(graph.edge_mask.bool().unsqueeze(-1), edge_states, torch.zeros_like(edge_states))
        node_states = edge_states.new_zeros((batch, length, self.config.node_width))
        training_messages = self.training and torch.is_grad_enabled()
        node_states, edge_states = self.encoder(
            node_states, edge_states, graph.neighbor_indices, residue_mask, graph.edge_mask,
            training_messages=training_messages, checkpoint_layers=checkpoint_layers,
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
        if encoded._module_training != self.training or encoded._training_messages != training_messages:
            raise ValueError("encoded backbone was created in a different model mode")

        edge_mask = encoded.edge_mask
        sequence_features = self.sequence_embedding(sequence)
        future_mask, past_mask = build_decoding_masks(
            encoded.neighbor_indices, encoded.residue_mask, decoding_order, patch_index, fixed_decoding_order_length, edge_mask
        )
        node_states = self.decoder(
            encoded.node_states, encoded.edge_states, sequence_features, encoded.neighbor_indices, encoded.residue_mask,
            future_mask, past_mask, edge_mask, checkpoint_layers=checkpoint_layers,
        )
        logits = self.output_projection(node_states)
        if return_log_prob:
            return F.log_softmax(logits, dim=-1)
        return logits.permute(0, 2, 1)

    def _forward_pytorch(
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
        """Logits ``[B, output_vocabulary, L]`` (or log-probabilities ``[B, L, output_vocabulary]`` with ``return_log_prob``)."""
        encoded = self.encode_backbone(
            backbone, residue_mask, residue_index, chain_index, segment_lengths, checkpoint_layers=checkpoint_layers
        )
        return self.score_sequence(
            encoded, sequence, decoding_order, patch_index,
            fixed_decoding_order_length=fixed_decoding_order_length, checkpoint_layers=checkpoint_layers,
            return_log_prob=return_log_prob,
        )

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` chains-of-two backbones of ``length`` residues (a random walk of 3.8 A CA steps, N / C / O within ~1 A of their CA;
        real residues first), a random sequence and decoding order, one decoding patch per position."""
        step = random_input((samples, length, 3), dtype=torch.float32, device=device, generator=generator)
        ca = torch.cumsum(3.8 * step / step.norm(dim=-1, keepdim=True), dim=1)
        offsets = random_input((samples, length, 4, 3), dtype=torch.float32, device=device, generator=generator)
        offsets[:, :, 1] = 0.0
        index = torch.arange(length, device=device).expand(samples, length).contiguous()
        return {"backbone": (ca[:, :, None] + offsets).to(dtype),
                "sequence": torch.randint(0, self.config.sequence_vocabulary, (samples, length), device=device, generator=generator),
                "residue_mask": key_mask(samples, length, device=device), "residue_index": index,
                "chain_index": (index >= length // 2).long(),
                "decoding_order": torch.argsort(torch.rand((samples, length), device=device, generator=generator), dim=-1),
                "patch_index": index}
