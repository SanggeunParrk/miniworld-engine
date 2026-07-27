"""Backbone geometry and relative-position features for production MPNN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from miniworld_kernels.kernels.mpnn_edge_layernorm import (
    EdgeNormBackend,
    edge_layer_norm,
)
from miniworld_kernels.kernels.mpnn_relative_position import (
    RelativePositionBackend,
    relative_position_embed,
)

from ._functional import gather_neighbors


FeatureBackend = Literal["auto", "pytorch", "recompute"]
KNNBackend = Literal["cdist", "chunked", "grid_cutoff"]


@dataclass(frozen=True)
class NeighborGraph:
    """Backbone-derived fixed-width neighbor graph.

    ``edge_features`` has shape ``[B, L, K, C]`` while
    ``neighbor_indices`` and ``edge_mask`` have shape ``[B, L, K]``.  A zero
    in ``edge_mask`` marks a padded or otherwise invalid edge whose feature is
    guaranteed to be zero.
    """

    edge_features: torch.Tensor
    neighbor_indices: torch.Tensor
    edge_mask: torch.Tensor


def _distance_index_key(
    distances: torch.Tensor,
    indices: torch.Tensor,
    inside: torch.Tensor,
) -> torch.Tensor:
    """Rank by (distance, index) with no ties and without perturbing distances.

    Non-negative IEEE-754 floats compare in the same order as their bit patterns,
    so the high word orders by distance and the low word breaks every remaining
    tie by candidate index. ``topk`` on the composite key is therefore a total
    order, unlike ``topk`` on the distances alone, whose tie behaviour is not
    specified and changes with the candidate axis.
    """
    # Distances are non-negative, so their patterns stay below 2**31 and the shift
    # cannot reach the sign bit. The sentinel has to respect that too: 0xFFFFFFFF
    # would overflow into a negative key and make excluded candidates rank first.
    bits = distances.to(torch.float32).contiguous().view(torch.int32).to(torch.int64)
    excluded = torch.full_like(bits, 0x7FFFFFFF)
    bits = torch.where(inside, bits.clamp_min(0), excluded)
    return (bits << 32) | indices.clamp_min(0)


def _virtual_cb(backbone: torch.Tensor) -> torch.Tensor:
    nitrogen, alpha_carbon, carbon = (
        backbone[:, :, 0],
        backbone[:, :, 1],
        backbone[:, :, 2],
    )
    n_to_ca = alpha_carbon - nitrogen
    ca_to_c = carbon - alpha_carbon
    cross = torch.cross(n_to_ca, ca_to_c, dim=-1)
    return (
        alpha_carbon - 0.58273431 * cross + 0.56802827 * n_to_ca - 0.54067466 * ca_to_c
    )


class RelativePositionEmbedding(nn.Module):
    """Embedding form of the source one-hot relative-position projection."""

    def __init__(
        self,
        width: int,
        max_relative_offset: int = 32,
        backend: RelativePositionBackend = "off",
    ) -> None:
        super().__init__()
        self.max_relative_offset = max_relative_offset
        self.num_buckets = 2 * max_relative_offset + 2
        self.embedding = nn.Embedding(self.num_buckets, width)
        self.bias = nn.Parameter(torch.empty(width))
        self.backend = backend
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.Linear(num_buckets, width), then store its transpose as a
        # contiguous lookup table [bucket, channel].
        linear_layout = torch.empty(
            self.embedding.embedding_dim,
            self.num_buckets,
            device=self.embedding.weight.device,
            dtype=self.embedding.weight.dtype,
        )
        nn.init.kaiming_uniform_(linear_layout, a=math.sqrt(5.0))
        bound = 1.0 / math.sqrt(self.num_buckets)
        with torch.no_grad():
            self.embedding.weight.copy_(linear_layout.T)
            self.bias.uniform_(-bound, bound)

    def forward(
        self, relative_offset: torch.Tensor, same_chain: torch.Tensor
    ) -> torch.Tensor:
        bucket = (
            torch.clamp(
                relative_offset + self.max_relative_offset,
                0,
                2 * self.max_relative_offset,
            )
            * same_chain
            + (1 - same_chain) * (2 * self.max_relative_offset + 1)
        ).long()
        # One index per *edge* into a table of a few dozen rows, so the backward is a
        # 6-million-into-66 reduction whose cost is set by how unevenly the rows land.
        # The clamp puts every long-range contact in the two end buckets -- a third of
        # all edges at T=8192 -- and the compiler's reduction for that shape measured
        # 30.8 ms per call, 16% of a B=16 step. The boundary lets a better one run.
        return relative_position_embed(
            bucket, self.embedding.weight, self.bias, backend=self.backend
        )


class BackboneFeatures(nn.Module):
    """KNN-first geometric features with O(B*L*K) atom-pair work."""

    _PAIR_A = (
        1,
        0,
        2,
        3,
        4,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        4,
        4,
        3,
        0,
        2,
        3,
        4,
        2,
        3,
        4,
        2,
        3,
        2,
    )
    _PAIR_B = (
        1,
        0,
        2,
        3,
        4,
        0,
        2,
        3,
        4,
        2,
        3,
        4,
        2,
        3,
        2,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        4,
        4,
        3,
    )

    def __init__(
        self,
        edge_width: int,
        position_width: int = 16,
        num_rbf: int = 16,
        k_neighbors: int = 30,
        coordinate_noise: float = 0.0,
        feature_backend: FeatureBackend = "auto",
        relative_position_backend: RelativePositionBackend = "off",
        knn_backend: KNNBackend = "cdist",
        knn_query_chunk: int = 2048,
        knn_cutoff: float = 16.0,
        edge_norm_backend: EdgeNormBackend = "auto",
    ) -> None:
        super().__init__()
        self.edge_width = edge_width
        self.position_width = position_width
        self.num_rbf = num_rbf
        self.k_neighbors = k_neighbors
        self.coordinate_noise = coordinate_noise
        if feature_backend not in {"auto", "pytorch", "recompute"}:
            raise ValueError(f"unknown MPNN feature backend: {feature_backend!r}")
        self.feature_backend = feature_backend
        if knn_backend not in {"cdist", "chunked", "grid_cutoff"}:
            raise ValueError(f"unknown MPNN knn backend: {knn_backend!r}")
        if knn_query_chunk <= 0:
            raise ValueError("knn_query_chunk must be positive")
        if knn_cutoff <= 0:
            raise ValueError("knn_cutoff must be positive")
        self.knn_backend = knn_backend
        self.knn_query_chunk = knn_query_chunk
        self.knn_cutoff = knn_cutoff
        if edge_norm_backend not in {"auto", "pytorch", "memory"}:
            raise ValueError(f"unknown MPNN edge norm backend: {edge_norm_backend!r}")
        self.edge_norm_backend = edge_norm_backend
        self.relative_position = RelativePositionEmbedding(
            position_width, backend=relative_position_backend
        )
        self.edge_projection = nn.Linear(
            position_width + num_rbf * 25,
            edge_width,
            bias=False,
        )
        self.edge_norm = nn.LayerNorm(edge_width)
        self.register_buffer(
            "_pair_a", torch.tensor(self._PAIR_A, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_pair_b", torch.tensor(self._PAIR_B, dtype=torch.long), persistent=False
        )

    def _segment_ids(
        self,
        alpha_carbon: torch.Tensor,
        segment_lengths: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if segment_lengths is None:
            return None
        if alpha_carbon.shape[0] != 1:
            raise ValueError(
                "segment_lengths is only supported for physical batch size 1; "
                "use a padded batch and residue_mask for multiple graphs"
            )
        lengths = segment_lengths.to(alpha_carbon.device)
        return torch.arange(
            lengths.numel(), device=alpha_carbon.device
        ).repeat_interleave(lengths, output_size=alpha_carbon.shape[1])

    def _nearest_neighbors(
        self,
        alpha_carbon: torch.Tensor,
        residue_mask: torch.Tensor,
        segment_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.knn_backend == "grid_cutoff":
            return self._nearest_neighbors_grid(
                alpha_carbon, residue_mask, segment_lengths
            )
        if self.knn_backend == "chunked" and not (
            torch.is_grad_enabled() and alpha_carbon.requires_grad
        ):
            # Chunking only bounds the peak when nothing is retained for backward.
            # `cdist` saves its own output, so under autograd every chunk's matrix
            # would stay live and the peak would be unchanged; fall through to the
            # single-shot path instead of paying the launch overhead for nothing.
            with torch.no_grad():
                return self._nearest_neighbors_chunked(
                    alpha_carbon, residue_mask, segment_lengths
                )
        return self._nearest_neighbors_cdist(
            alpha_carbon, residue_mask, segment_lengths
        )

    def _nearest_neighbors_cdist(
        self,
        alpha_carbon: torch.Tensor,
        residue_mask: torch.Tensor,
        segment_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = alpha_carbon.shape[1]
        pair_mask = residue_mask[:, None] * residue_mask[:, :, None]
        segment = self._segment_ids(alpha_carbon, segment_lengths)
        if segment is not None:
            pair_mask = pair_mask * (segment[:, None] == segment[None, :])
        distances = torch.cdist(alpha_carbon, alpha_carbon) * pair_mask
        row_max = distances.max(dim=-1, keepdim=True).values
        adjusted = distances + (1.0 - pair_mask) * (row_max + 100.0)
        distances, neighbor_indices = torch.topk(
            adjusted,
            min(self.k_neighbors, length),
            dim=-1,
            largest=False,
        )
        edge_mask = torch.gather(pair_mask, 2, neighbor_indices)
        return distances, neighbor_indices, edge_mask

    def _nearest_neighbors_chunked(
        self,
        alpha_carbon: torch.Tensor,
        residue_mask: torch.Tensor,
        segment_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same exact KNN, one query block at a time.

        Every row of the pair matrix is complete inside its block, so the
        row-maximum shift and the selection are unchanged; only the peak size of
        the intermediate falls from ``length x length`` to ``chunk x length``.
        ``cdist`` is pinned to its direct form because it otherwise switches to a
        matmul expansion above a size threshold, which would make the retained
        CA distance depend on the chunk size rather than on the geometry.
        """
        batch, length = alpha_carbon.shape[:2]
        segment = self._segment_ids(alpha_carbon, segment_lengths)
        neighbors = min(self.k_neighbors, length)
        distance_blocks = []
        index_blocks = []
        mask_blocks = []
        for start in range(0, length, self.knn_query_chunk):
            stop = min(start + self.knn_query_chunk, length)
            block_mask = residue_mask[:, start:stop, None] * residue_mask[:, None, :]
            if segment is not None:
                block_mask = block_mask * (
                    segment[start:stop, None] == segment[None, :]
                )
            block = torch.cdist(
                alpha_carbon[:, start:stop],
                alpha_carbon,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            block = block * block_mask
            row_max = block.max(dim=-1, keepdim=True).values
            adjusted = block + (1.0 - block_mask) * (row_max + 100.0)
            values, indices = torch.topk(adjusted, neighbors, dim=-1, largest=False)
            distance_blocks.append(values)
            index_blocks.append(indices)
            mask_blocks.append(torch.gather(block_mask, 2, indices))
        del batch
        return (
            torch.cat(distance_blocks, dim=1),
            torch.cat(index_blocks, dim=1),
            torch.cat(mask_blocks, dim=1),
        )

    def _nearest_neighbors_grid(
        self,
        alpha_carbon: torch.Tensor,
        residue_mask: torch.Tensor,
        segment_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """At most ``k`` neighbours, all strictly within ``knn_cutoff``.

        A uniform grid of cell size ``knn_cutoff`` makes this exact rather than
        approximate: every point within ``knn_cutoff`` of a query lies in the
        query's own cell or one of the 26 adjacent ones, because a Euclidean
        distance of at most one cell width can move each axis by at most one cell.
        So the 3x3x3 block is a superset of the answer and no fallback can ever be
        required -- which is only true because the rule is a cutoff. Plain k-NN
        would need the cell to exceed the k-th neighbour distance, and on real
        structures that reaches 100 angstrom in disordered or detached regions.

        Ordering is defined here rather than inherited: candidates are ranked by
        (distance, index) through a composite key, so equal distances resolve the
        same way on every device and for every candidate ordering.
        """
        if alpha_carbon.shape[0] != 1:
            raise ValueError(
                "the grid_cutoff KNN backend expects physical batch size 1; "
                "pack the graphs into one row and pass segment_lengths"
            )
        device = alpha_carbon.device
        length = alpha_carbon.shape[1]
        neighbors = min(self.k_neighbors, length)
        points = alpha_carbon[0]
        valid = residue_mask[0] > 0
        segment = self._segment_ids(alpha_carbon, segment_lengths)
        if segment is None:
            segment = torch.zeros(length, dtype=torch.long, device=device)

        positions = valid.nonzero(as_tuple=True)[0]
        live = points[positions]
        cells = torch.floor(live / self.knn_cutoff).to(torch.long)
        cells = cells - cells.amin(dim=0, keepdim=True)
        extent = cells.amax(dim=0) + 1
        strides = torch.stack(
            (extent[1] * extent[2], extent[2], torch.ones_like(extent[2]))
        )
        keys = segment[positions] * int(extent.prod()) + (cells * strides).sum(-1)

        order = torch.argsort(keys, stable=True)
        sorted_keys = keys[order]
        unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        occupancy = int(counts.max())

        table = torch.full(
            (unique_keys.numel(), occupancy), -1, dtype=torch.long, device=device
        )
        slot = torch.arange(
            sorted_keys.numel(), device=device
        ) - starts.repeat_interleave(counts)
        cell_of = torch.repeat_interleave(
            torch.arange(unique_keys.numel(), device=device), counts
        )
        table[cell_of, slot] = positions[order]

        span = torch.arange(-1, 2, device=device)
        offsets = torch.cartesian_prod(span, span, span)
        # Bound each axis before packing. Adding an offset directly to the packed
        # key would let one axis borrow from the next -- stepping -1 in z from
        # z=0 lands on the previous y row rather than outside the grid -- which
        # both admits wrong cells and lets one cell appear twice among the 27.
        neighbour_cells = cells[:, None, :] + offsets[None, :, :]
        in_range = ((neighbour_cells >= 0) & (neighbour_cells < extent)).all(dim=-1)
        candidate_keys = segment[positions][:, None] * int(extent.prod()) + (
            neighbour_cells * strides
        ).sum(-1)
        found = torch.searchsorted(unique_keys, candidate_keys.clamp_min(0)).clamp_max(
            unique_keys.numel() - 1
        )
        matched = in_range & (unique_keys[found] == candidate_keys)
        candidates = table[torch.where(matched, found, torch.zeros_like(found))]
        candidates = torch.where(
            matched[:, :, None], candidates, torch.full_like(candidates, -1)
        ).reshape(positions.numel(), -1)

        present = candidates >= 0
        offset = points[candidates.clamp_min(0)] - live[:, None, :]
        candidate_distances = torch.linalg.vector_norm(offset, dim=-1)
        inside = present & (candidate_distances <= self.knn_cutoff)
        ranked = _distance_index_key(candidate_distances, candidates, inside)
        keys_taken, slots = torch.topk(
            ranked, min(neighbors, ranked.shape[-1]), dim=-1, largest=False
        )
        taken = torch.gather(candidates, 1, slots)
        taken_valid = torch.gather(inside, 1, slots)
        taken_distances = torch.gather(candidate_distances, 1, slots)
        del keys_taken

        # Unused slots point at the query itself so every downstream gather stays
        # in bounds; the mask is what removes them, exactly as for padded rows.
        self_index = positions[:, None].expand_as(taken)
        taken = torch.where(taken_valid, taken, self_index)
        taken_distances = torch.where(
            taken_valid, taken_distances, torch.zeros_like(taken_distances)
        )

        distances = torch.zeros(1, length, neighbors, device=device, dtype=points.dtype)
        indices = (
            torch.arange(length, device=device)[None, :, None]
            .expand(1, length, neighbors)
            .clone()
        )
        edge_mask = torch.zeros(
            1, length, neighbors, device=device, dtype=residue_mask.dtype
        )
        pad = neighbors - taken.shape[1]
        if pad > 0:
            taken = torch.cat((taken, self_index[:, :pad]), dim=1)
            taken_distances = torch.nn.functional.pad(taken_distances, (0, pad))
            taken_valid = torch.nn.functional.pad(taken_valid, (0, pad))
        distances[0, positions] = taken_distances
        indices[0, positions] = taken
        edge_mask[0, positions] = taken_valid.to(residue_mask.dtype)
        return distances, indices, edge_mask

    def _radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(2.0, 22.0, self.num_rbf, device=distances.device).view(
            1, 1, 1, 1, -1
        )
        sigma = 20.0 / self.num_rbf
        return torch.exp(-(((distances.unsqueeze(-1) - centers) / sigma) ** 2))

    def _project_radial(
        self,
        pair_distances: torch.Tensor,
        radial_weight: torch.Tensor,
    ) -> torch.Tensor:
        radial_features = self._radial_basis(pair_distances).flatten(start_dim=-2)
        return F.linear(radial_features, radial_weight)

    def _project_combined(
        self,
        position_features: torch.Tensor,
        pair_distances: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        radial_features = self._radial_basis(pair_distances).flatten(start_dim=-2)
        return F.linear(torch.cat((position_features, radial_features), dim=-1), weight)

    def build_graph(
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
        backbone = torch.where(
            residue_mask.bool().unsqueeze(-1).unsqueeze(-1),
            backbone,
            torch.zeros_like(backbone),
        )

        virtual_cb = _virtual_cb(backbone)
        atoms = torch.cat((backbone, virtual_cb.unsqueeze(2)), dim=2)
        ca_distances, neighbor_indices, edge_mask = self._nearest_neighbors(
            atoms[:, :, 1], residue_mask, segment_lengths
        )

        batch, length = atoms.shape[:2]
        neighbor_atoms = gather_neighbors(
            atoms.reshape(batch, length, 15), neighbor_indices
        ).reshape(batch, length, neighbor_indices.shape[2], 5, 3)
        source = atoms[:, :, None, self._pair_a, :]
        target = neighbor_atoms[:, :, :, self._pair_b, :]
        pair_distances = torch.linalg.vector_norm(source - target, dim=-1)
        pair_distances = torch.cat(
            (ca_distances.unsqueeze(-1), pair_distances[..., 1:]), dim=-1
        )
        neighbor_residue_index = gather_neighbors(
            residue_index.unsqueeze(-1), neighbor_indices
        ).squeeze(-1)
        neighbor_chain_index = gather_neighbors(
            chain_index.unsqueeze(-1), neighbor_indices
        ).squeeze(-1)
        relative_offset = residue_index.unsqueeze(-1) - neighbor_residue_index
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(relative_offset, same_chain)

        weight = self.edge_projection.weight
        split_projection = (
            position_features.requires_grad and not pair_distances.requires_grad
        )
        recompute = self.feature_backend == "recompute" and torch.is_grad_enabled()
        if recompute and split_projection:
            edge_features = F.linear(
                position_features,
                weight[:, : self.position_width],
                self.edge_projection.bias,
            )
            radial_projection = torch.utils.checkpoint.checkpoint(
                self._project_radial,
                pair_distances,
                weight[:, self.position_width :],
                use_reentrant=False,
                preserve_rng_state=False,
            )
            edge_features = edge_features + radial_projection
        elif recompute:
            edge_features = torch.utils.checkpoint.checkpoint(
                self._project_combined,
                position_features,
                pair_distances,
                weight,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            radial_features = self._radial_basis(pair_distances).flatten(start_dim=-2)
            if split_projection:
                edge_features = F.linear(
                    position_features,
                    weight[:, : self.position_width],
                    self.edge_projection.bias,
                )
                edge_features = edge_features + F.linear(
                    radial_features, weight[:, self.position_width :]
                )
            else:
                edge_features = self.edge_projection(
                    torch.cat((position_features, radial_features), dim=-1)
                )
        if self.edge_norm_backend == "memory":
            # Autocast promotes layer_norm to FP32, so the ordinary module retains a
            # full-width FP32 edge tensor here -- twice the size of everything else
            # on the tape. The memory boundary keeps the same native forward and
            # stores a BF16 copy for backward instead.
            edge_features = edge_layer_norm(
                edge_features,
                self.edge_norm.weight,
                self.edge_norm.bias,
                self.edge_norm.eps,
                backend="memory",
            )
        else:
            edge_features = self.edge_norm(edge_features)
        edge_features = torch.where(
            edge_mask.bool().unsqueeze(-1),
            edge_features,
            torch.zeros_like(edge_features),
        )
        return NeighborGraph(
            edge_features=edge_features,
            neighbor_indices=neighbor_indices,
            edge_mask=edge_mask,
        )

    def forward(
        self,
        backbone: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        segment_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the historical edge-feature pair for compatibility."""
        graph = self.build_graph(
            backbone,
            residue_mask,
            residue_index,
            chain_index,
            segment_lengths,
        )
        return graph.edge_features, graph.neighbor_indices
