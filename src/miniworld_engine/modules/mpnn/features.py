"""Backbone geometry and relative-position features of ProteinMPNN: the exact ``cdist`` KNN graph and its edge features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_engine.modules.mpnn._functional import MPNNLayerNorm, gather_neighbors, mpnn_sm80_unsupported, projection_input
from miniworld_engine.modules.primitives import BackendModule, wide


@dataclass(frozen=True)
class NeighborGraph:
    """Backbone-derived fixed-width neighbor graph.

    ``edge_features`` has shape ``[B, L, K, C]`` while ``neighbor_indices`` and ``edge_mask`` have shape ``[B, L, K]``. A zero in
    ``edge_mask`` marks a padded or otherwise invalid edge whose feature is guaranteed to be zero.
    """

    edge_features: torch.Tensor
    neighbor_indices: torch.Tensor
    edge_mask: torch.Tensor


def _virtual_cb(backbone: torch.Tensor) -> torch.Tensor:
    nitrogen, alpha_carbon, carbon = backbone[:, :, 0], backbone[:, :, 1], backbone[:, :, 2]
    n_to_ca = alpha_carbon - nitrogen
    ca_to_c = carbon - alpha_carbon
    cross = torch.cross(n_to_ca, ca_to_c, dim=-1)
    return alpha_carbon - 0.58273431 * cross + 0.56802827 * n_to_ca - 0.54067466 * ca_to_c


class RelativePositionEmbedding(BackendModule):
    """Embedding form of the source one-hot relative-position projection: one table row per edge plus the shared bias
    (``embedding(bucket) + bias``, the one-hot GEMM of the source written as a lookup).

    ``"triton"``: the same lookup, its backward's table / bias reduction on the Triton bucket reduction
    (``kernels/mpnn_relative_position``) instead of ATen's sorted ``embedding_dense_backward``. Range: a CUDA table of at most 512 rows
    and 128 channels, fp32 or bf16, with a bias of its dtype.
    """

    backends = ("pytorch", "triton", "cuda")

    #: The reduction privatises a whole table per program: the real table is 66 x 16.
    _MAX_BUCKETS = 512
    _MAX_WIDTH = 128

    def __init__(self, width: int, max_relative_offset: int = 32, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.max_relative_offset = max_relative_offset
        self.num_buckets = 2 * max_relative_offset + 2
        self.embedding = nn.Embedding(self.num_buckets, width)
        self.bias = nn.Parameter(torch.empty(width))
        self.reset_parameters()
        self.set_backend(backend)

    def reset_parameters(self) -> None:
        # Match nn.Linear(num_buckets, width), then store its transpose as a contiguous lookup table [bucket, channel].
        linear_layout = torch.empty(
            self.embedding.embedding_dim, self.num_buckets, device=self.embedding.weight.device, dtype=self.embedding.weight.dtype
        )
        nn.init.kaiming_uniform_(linear_layout, a=math.sqrt(5.0))
        bound = 1.0 / math.sqrt(self.num_buckets)
        with torch.no_grad():
            self.embedding.weight.copy_(linear_layout.T)
            self.bias.uniform_(-bound, bound)

    def _bucket(self, relative_offset: torch.Tensor, same_chain: torch.Tensor) -> torch.Tensor:
        return (
            torch.clamp(relative_offset + self.max_relative_offset, 0, 2 * self.max_relative_offset) * same_chain
            + (1 - same_chain) * (2 * self.max_relative_offset + 1)
        ).long()

    def _forward_pytorch(self, relative_offset: torch.Tensor, same_chain: torch.Tensor) -> torch.Tensor:
        return F.embedding(self._bucket(relative_offset, same_chain), self.embedding.weight) + self.bias

    def _supports_triton(self, relative_offset: torch.Tensor, same_chain: torch.Tensor) -> str | None:
        table, bias = self.embedding.weight, self.bias
        if relative_offset.device.type != "cuda" or table.device != relative_offset.device or bias.device != relative_offset.device:
            return "the Triton kernels run on CUDA tensors only"
        if relative_offset.numel() == 0:
            return "the bucket reduction needs a non-empty index"
        buckets, width = table.shape
        if not 0 < buckets <= self._MAX_BUCKETS or not 0 < width <= self._MAX_WIDTH:
            return f"table {tuple(table.shape)}: the reduction is built for at most {self._MAX_BUCKETS} rows and {self._MAX_WIDTH} channels"
        if table.dtype != bias.dtype or table.dtype not in {torch.float32, torch.bfloat16}:
            return f"table {table.dtype}, bias {bias.dtype}: the reduction takes an fp32 or bf16 table with a bias of its dtype"
        if not (table.is_contiguous() and bias.is_contiguous()):
            return "the table and the bias must be contiguous"
        return None

    def _forward_triton(self, relative_offset: torch.Tensor, same_chain: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.mpnn_relative_position.triton import relative_position_embed_op

        bucket = self._bucket(relative_offset, same_chain).contiguous()
        return relative_position_embed_op(bucket, self.embedding.weight, self.bias, "triton")

    def _supports_cuda(self, relative_offset: torch.Tensor, same_chain: torch.Tensor) -> str | None:
        """A100: the same lookup, its backward's bucket reduction on the sm_80 one-hot matmul (``kernels/integrations/mpnn_msg_sm80.relative_position_embed``):
        a 16-channel table of at most 79 buckets."""
        from miniworld_engine.kernels.integrations import mpnn_msg_sm80

        why = mpnn_sm80_unsupported(relative_offset, mpnn_msg_sm80._relpos_loads) or self._supports_triton(relative_offset, same_chain)
        if why is not None:
            return why
        buckets, width = self.embedding.weight.shape
        if width != 16 or buckets > 79:
            return f"table {tuple(self.embedding.weight.shape)}: the sm_80 reduction is built for 16 channels and at most 79 buckets"
        return None

    def _forward_cuda(self, relative_offset: torch.Tensor, same_chain: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.integrations import mpnn_msg_sm80

        bucket = self._bucket(relative_offset, same_chain).contiguous()
        return mpnn_msg_sm80.relative_position_embed(bucket, self.embedding.weight, self.bias)


class BackboneFeatures(BackendModule):
    """KNN-first geometric features with O(B*L*K) atom-pair work: the backbone -> the neighbour graph (:class:`NeighborGraph`)."""

    backends = ("pytorch",)

    _pair_a: torch.Tensor
    _pair_b: torch.Tensor

    _PAIR_A = (1, 0, 2, 3, 4, 1, 1, 1, 1, 0, 0, 0, 4, 4, 3, 0, 2, 3, 4, 2, 3, 4, 2, 3, 2)
    _PAIR_B = (1, 0, 2, 3, 4, 0, 2, 3, 4, 2, 3, 4, 2, 3, 2, 1, 1, 1, 1, 0, 0, 0, 4, 4, 3)

    def __init__(
        self,
        edge_width: int,
        position_width: int = 16,
        num_rbf: int = 16,
        k_neighbors: int = 30,
        coordinate_noise: float = 0.0,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.edge_width = edge_width
        self.position_width = position_width
        self.num_rbf = num_rbf
        self.k_neighbors = k_neighbors
        self.coordinate_noise = coordinate_noise
        self.relative_position = RelativePositionEmbedding(position_width)
        self.edge_projection = nn.Linear(position_width + num_rbf * 25, edge_width, bias=False)
        self.edge_norm = MPNNLayerNorm(edge_width)
        self.register_buffer("_pair_a", torch.tensor(self._PAIR_A, dtype=torch.long), persistent=False)
        self.register_buffer("_pair_b", torch.tensor(self._PAIR_B, dtype=torch.long), persistent=False)
        self.set_backend(backend)

    def _segment_ids(self, alpha_carbon: torch.Tensor, segment_lengths: torch.Tensor | None) -> torch.Tensor | None:
        if segment_lengths is None:
            return None
        if alpha_carbon.shape[0] != 1:
            raise ValueError(
                "segment_lengths is only supported for physical batch size 1; use a padded batch and residue_mask for multiple graphs"
            )
        lengths = segment_lengths.to(alpha_carbon.device)
        return torch.arange(lengths.numel(), device=alpha_carbon.device).repeat_interleave(lengths, output_size=alpha_carbon.shape[1])

    def _nearest_neighbors(
        self,
        alpha_carbon: torch.Tensor,
        residue_mask: torch.Tensor,
        segment_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact k nearest CA neighbours (masked and cross-segment pairs ranked last)."""
        length = alpha_carbon.shape[1]
        pair_mask = (residue_mask[:, None] * residue_mask[:, :, None]).to(alpha_carbon.dtype)  # a bool key mask is as good as a float one
        segment = self._segment_ids(alpha_carbon, segment_lengths)
        if segment is not None:
            pair_mask = pair_mask * (segment[:, None] == segment[None, :])
        distances = torch.cdist(alpha_carbon, alpha_carbon) * pair_mask
        row_max = distances.max(dim=-1, keepdim=True).values
        adjusted = distances + (1.0 - pair_mask) * (row_max + 100.0)
        distances, neighbor_indices = torch.topk(adjusted, min(self.k_neighbors, length), dim=-1, largest=False)
        edge_mask = torch.gather(pair_mask, 2, neighbor_indices)
        return distances, neighbor_indices, edge_mask

    def _radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(2.0, 22.0, self.num_rbf, device=distances.device, dtype=wide(distances)).view(1, 1, 1, 1, -1)
        sigma = 20.0 / self.num_rbf
        return torch.exp(-(((distances.unsqueeze(-1) - centers) / sigma) ** 2))

    def _forward_pytorch(
        self,
        backbone: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        segment_lengths: torch.Tensor | None = None,
    ) -> NeighborGraph:
        """Build edge features and validity metadata for the backbone KNN graph."""
        if self.training and self.coordinate_noise > 0:
            backbone = backbone + self.coordinate_noise * torch.randn_like(backbone)
        backbone = torch.where(residue_mask.bool().unsqueeze(-1).unsqueeze(-1), backbone, torch.zeros_like(backbone))

        virtual_cb = _virtual_cb(backbone)
        atoms = torch.cat((backbone, virtual_cb.unsqueeze(2)), dim=2)
        ca_distances, neighbor_indices, edge_mask = self._nearest_neighbors(atoms[:, :, 1], residue_mask, segment_lengths)

        batch, length = atoms.shape[:2]
        neighbor_atoms = gather_neighbors(atoms.reshape(batch, length, 15), neighbor_indices).reshape(
            batch, length, neighbor_indices.shape[2], 5, 3
        )
        source = atoms[:, :, None, self._pair_a, :]
        target = neighbor_atoms[:, :, :, self._pair_b, :]
        pair_distances = torch.linalg.vector_norm(source - target, dim=-1)
        pair_distances = torch.cat((ca_distances.unsqueeze(-1), pair_distances[..., 1:]), dim=-1)
        neighbor_residue_index = gather_neighbors(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        neighbor_chain_index = gather_neighbors(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        relative_offset = residue_index.unsqueeze(-1) - neighbor_residue_index
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(relative_offset, same_chain)

        weight = self.edge_projection.weight
        radial_features = self._radial_basis(pair_distances).flatten(start_dim=-2)
        if position_features.requires_grad and not pair_distances.requires_grad:
            # Fixed geometry: the radial block is a constant operand, so the two blocks project apart.
            edge_features = F.linear(position_features, weight[:, : self.position_width], self.edge_projection.bias)
            edge_features = edge_features + F.linear(projection_input(radial_features, weight), weight[:, self.position_width :])
        else:
            edge_features = self.edge_projection(projection_input(torch.cat((position_features, radial_features), dim=-1), weight))
        edge_features = self.edge_norm(edge_features)
        edge_features = torch.where(edge_mask.bool().unsqueeze(-1), edge_features, torch.zeros_like(edge_features))
        return NeighborGraph(edge_features=edge_features, neighbor_indices=neighbor_indices, edge_mask=edge_mask)
