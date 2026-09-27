"""Naive PyTorch reference for the current CSSB ProteinMPNN.

This module preserves the parameter layout, tensor materialization, operation
ordering, masks, and initialization used by ``ProteinMPNN_CSSB`` at
``origin/dev`` commit ``4870bca``.  It is deliberately *not* an optimized or
cleaned-up implementation: future MPNN kernels are compared against this oracle.

Only the model's parallel training/scoring forward is included here.  The much
larger sampling policy (patch selection, symmetry, and side-chain orchestration)
will live outside the numerical core and be added as a separate reference path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

NUM_AMINO_ACID_TYPES = 21


def _init_lecun_normal(module: nn.Module, scale: float = 1.0) -> nn.Module:
    """Preserve the source repository's truncated LeCun initialization."""

    def truncated_normal(
        uniform: torch.Tensor,
        mu: float = 0.0,
        sigma: float = 1.0,
        a: float = -2,
        b: float = 2,
    ) -> torch.Tensor:
        normal = torch.distributions.normal.Normal(0, 1)
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        alpha_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_cdf + (normal.cdf(torch.tensor(beta)) - alpha_cdf) * uniform
        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        return torch.clamp(x, a, b)

    def sample_truncated_normal(shape: torch.Size, scale: float = 1.0) -> torch.Tensor:
        stddev = np.sqrt(scale / shape[-1]) / 0.87962566103423978
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = nn.Parameter(sample_truncated_normal(module.weight.shape))
    return module


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, N, N, C]`` edges at ``[B, N, K]`` neighbor indices."""
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, N, C]`` nodes at ``[B, N, K]`` neighbor indices."""
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


def cat_neighbors_nodes(
    h_nodes: torch.Tensor,
    h_neighbors: torch.Tensor,
    edge_idx: torch.Tensor,
) -> torch.Tensor:
    """Append gathered neighbor node features to existing edge features."""
    return torch.cat([h_neighbors, gather_nodes(h_nodes, edge_idx)], dim=-1)


def _get_cb(xyz: torch.Tensor) -> torch.Tensor:
    """Construct the virtual C-beta coordinates used by the source model."""
    n = xyz[:, :, 0]
    ca = xyz[:, :, 1]
    c = xyz[:, :, 2]
    b = ca - n
    c_from_ca = c - ca
    a = torch.cross(b, c_from_ca, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_from_ca + ca


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = torch.clip(
            offset + self.max_relative_feature,
            0,
            2 * self.max_relative_feature,
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 2)
        return self.linear(d_onehot.float())


class ProteinFeatures(nn.Module):
    """Naive pairwise-distance and geometric edge featurizer."""

    def __init__(
        self,
        edge_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int | None = 30,
        augment_trans: float = 0.0,
        augment_rot: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.augment_trans = augment_trans
        self.augment_rot = augment_rot

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(
        self,
        xyz: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
        len_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del eps  # Kept in the signature for source compatibility.
        _, length = xyz.shape[:2]
        mask_2d = mask[:, None] * mask[:, :, None]
        if len_tensor is not None:
            segment = torch.arange(
                len(len_tensor), device=len_tensor.device
            ).repeat_interleave(len_tensor, output_size=length)
            mask_2d *= segment[:, None] == segment[None, :]

        dist = torch.cdist(xyz, xyz) * mask_2d
        d_max, _ = torch.max(dist, -1, keepdim=True)
        d_adjust = dist + (1.0 - mask_2d) * (d_max + 100.0)
        if self.top_k is not None:
            return torch.topk(
                d_adjust,
                np.minimum(self.top_k, length),
                dim=-1,
                largest=False,
            )
        return d_adjust, None

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        d_min, d_max, d_count = 2.0, 22.0, self.num_rbf
        d_mu = torch.linspace(d_min, d_max, d_count, device=distances.device).view(
            1, 1, 1, -1
        )
        d_sigma = (d_max - d_min) / d_count
        return torch.exp(-(((distances.unsqueeze(-1) - d_mu) / d_sigma) ** 2))

    def _get_rbf(
        self,
        xyz_a: torch.Tensor,
        xyz_b: torch.Tensor,
        edge_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        distances = torch.cdist(xyz_a, xyz_b)
        if edge_idx is not None:
            distances = gather_edges(distances[:, :, :, None], edge_idx)[..., 0]
        return self._rbf(distances)

    def forward(
        self,
        xyz: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_idx: torch.Tensor,
        len_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.training and (self.augment_trans > 0 or self.augment_rot > 0):
            # The source returns immediately after additive coordinate noise;
            # augment_rot is therefore intentionally unused.
            xyz = xyz + self.augment_trans * torch.randn_like(xyz)

        cb = _get_cb(xyz)
        n, ca, c, o = (xyz[:, :, atom, :] for atom in range(4))
        d_neighbors, edge_idx = self._dist(ca, mask, len_tensor=len_tensor)

        rbf_all = [self._rbf(d_neighbors)]
        atom_pairs = (
            (n, n),
            (c, c),
            (o, o),
            (cb, cb),
            (ca, n),
            (ca, c),
            (ca, o),
            (ca, cb),
            (n, c),
            (n, o),
            (n, cb),
            (cb, c),
            (cb, o),
            (o, c),
            (n, ca),
            (c, ca),
            (o, ca),
            (cb, ca),
            (c, n),
            (o, n),
            (cb, n),
            (c, cb),
            (o, cb),
            (c, o),
        )
        rbf_all.extend(self._get_rbf(a, b, edge_idx) for a, b in atom_pairs)
        rbf_features = torch.cat(tuple(rbf_all), dim=-1)

        seqsep = residue_idx[:, :, None] - residue_idx[:, None, :]
        d_chains = ((chain_idx[:, :, None] - chain_idx[:, None, :]) == 0).long()
        if edge_idx is not None:
            seqsep = gather_edges(seqsep[:, :, :, None], edge_idx)[..., 0]
            d_chains = gather_edges(d_chains[:, :, :, None], edge_idx)[..., 0]

        positional = self.embeddings(seqsep.long(), d_chains)
        edges = torch.cat((positional, rbf_features), dim=-1)
        edges = self.edge_embedding(edges)
        return self.norm_edges(edges), edge_idx


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int) -> None:
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()
        self.reset_parameter()

    def reset_parameter(self) -> None:
        nn.init.kaiming_normal_(self.W_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.W_out(self.act(self.W_in(node_features)))


class EncLayer(nn.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_hidden: int,
        dropout: float = 0.1,
        scale: int = 30,
    ) -> None:
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hidden = d_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_node)
        self.norm2 = nn.LayerNorm(d_node)
        self.norm3 = nn.LayerNorm(d_edge)
        self.W1 = nn.Linear(2 * d_node + d_edge, d_hidden, bias=True)
        self.W2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W3 = nn.Linear(d_hidden, d_node, bias=True)
        self.W11 = nn.Linear(2 * d_node + d_edge, d_hidden, bias=True)
        self.W12 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W13 = nn.Linear(d_hidden, d_edge, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(d_node, d_node * 4)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        for layer in (self.W1, self.W2, self.W3, self.W11, self.W12, self.W13):
            _init_lecun_normal(layer)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        h_v: torch.Tensor,
        h_e: torch.Tensor,
        edge_idx: torch.Tensor,
        mask_v: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = h_e.shape[2]
        h_ev = cat_neighbors_nodes(h_v, h_e, edge_idx)
        h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, top_k, -1)
        h_ev = torch.cat([h_v_expand, h_ev], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_v = self.norm1(h_v + self.dropout1(dh))
        h_v = self.norm2(h_v + self.dropout2(self.dense(h_v)))
        if mask_v is not None:
            h_v = mask_v.unsqueeze(-1) * h_v

        h_ev = cat_neighbors_nodes(h_v, h_e, edge_idx)
        h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_ev], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_ev)))))
        h_e = self.norm3(h_e + self.dropout3(h_message))
        return h_v, h_e


class DecLayer(nn.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_hidden: int,
        dropout: float = 0.1,
        num_heads: int | None = None,
        scale: int = 30,
    ) -> None:
        super().__init__()
        del num_heads
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hidden = d_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_node)
        self.norm2 = nn.LayerNorm(d_node)
        self.W1 = nn.Linear(3 * d_node + d_edge, d_hidden, bias=True)
        self.W2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W3 = nn.Linear(d_hidden, d_node, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(d_node, d_node * 4)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        for layer in (self.W1, self.W2, self.W3):
            _init_lecun_normal(layer)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        h_v: torch.Tensor,
        h_e: torch.Tensor,
        mask_v: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_e.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_e], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_v = self.norm1(h_v + self.dropout1(dh))
        h_v = self.norm2(h_v + self.dropout2(self.dense(h_v)))
        if mask_v is not None:
            h_v = mask_v.unsqueeze(-1) * h_v
        return h_v


class NaiveProteinMPNN(nn.Module):
    """Numerical reference for the current CSSB ProteinMPNN training forward."""

    def __init__(
        self,
        num_letters: int = NUM_AMINO_ACID_TYPES,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = NUM_AMINO_ACID_TYPES,
        k_neighbors: int = 32,
        augment_trans: float = 0.5,
        augment_rot: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.features = ProteinFeatures(
            edge_features,
            top_k=k_neighbors,
            augment_trans=augment_trans,
            augment_rot=augment_rot,
        )
        self.W_e = nn.Linear(edge_features, edge_features, bias=True)
        self.W_s = nn.Embedding(vocab, node_features)
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(
                    node_features,
                    edge_features,
                    hidden_dim,
                    dropout=dropout,
                    scale=k_neighbors,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(
                    node_features,
                    edge_features,
                    hidden_dim,
                    dropout=dropout,
                    scale=k_neighbors,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.W_out = nn.Linear(node_features, num_letters, bias=True)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        self.W_e = _init_lecun_normal(self.W_e)
        self.W_s = _init_lecun_normal(self.W_s)
        nn.init.zeros_(self.W_e.bias)
        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

    @staticmethod
    def get_decoding_masks(
        edge_idx: torch.Tensor,
        mask: torch.Tensor,
        decoding_order: torch.Tensor,
        patch_index_batch: torch.Tensor,
        fixed_decoding_order_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = decoding_order.shape
        arange_l = (
            torch.arange(length, device=decoding_order.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        residue_to_order = torch.empty_like(decoding_order)
        residue_to_order.scatter_(1, decoding_order, arange_l)
        residue_to_patch = torch.gather(patch_index_batch, 1, residue_to_order)
        patch_self = residue_to_patch.unsqueeze(2)
        patch_neighbor = torch.gather(
            residue_to_patch.unsqueeze(1).expand(-1, length, -1),
            2,
            index=edge_idx,
        )
        mask_attend = (patch_self > patch_neighbor).float().unsqueeze(-1)
        if fixed_decoding_order_len > 0:
            motif_self = residue_to_order < fixed_decoding_order_len
            motif_neighbor = torch.gather(
                motif_self.unsqueeze(1).expand(-1, length, -1), 2, edge_idx
            )
            mask_attend[motif_neighbor] = 1.0
            mask_attend[:, :, 0, :][motif_self] = 0.0

        mask_1d = mask.view([-1, length, 1, 1])
        mask_bw = mask_1d * mask_attend
        mask_fw = mask_1d * (1.0 - mask_attend)
        return mask_fw, mask_bw

    def forward(
        self,
        xyz: torch.Tensor,
        seq: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_idx: torch.Tensor,
        decoding_order: torch.Tensor,
        patch_index: torch.Tensor,
        loss_mask: torch.Tensor,
        len_tensor: torch.Tensor | None,
        use_checkpoint: bool = False,
        return_log_prob: bool = False,
    ) -> torch.Tensor:
        del loss_mask  # Present but unused in the source implementation.
        batch, length = xyz.shape[:2]
        edges, edge_idx = self.features(xyz, mask, residue_idx, chain_idx, len_tensor)
        if edge_idx is None:
            raise ValueError("NaiveProteinMPNN requires a finite k_neighbors")
        h_e = self.W_e(edges)
        h_v = torch.zeros((batch, length, self.node_features), device=xyz.device)

        mask_attend = gather_nodes(mask.unsqueeze(-1), edge_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            if use_checkpoint:
                h_v, h_e = torch.utils.checkpoint.checkpoint(
                    layer, h_v, h_e, edge_idx, mask, mask_attend
                )
            else:
                h_v, h_e = layer(h_v, h_e, edge_idx, mask, mask_attend)

        h_s = self.W_s(seq)
        h_es = cat_neighbors_nodes(h_s, h_e, edge_idx)
        h_ex = cat_neighbors_nodes(torch.zeros_like(h_s), h_e, edge_idx)
        h_exv = cat_neighbors_nodes(h_v, h_ex, edge_idx)
        mask_fw, mask_bw = self.get_decoding_masks(
            edge_idx, mask, decoding_order, patch_index, 0
        )
        h_exv_fw = mask_fw * h_exv
        for layer in self.decoder_layers:
            h_esv = cat_neighbors_nodes(h_v, h_es, edge_idx)
            h_esv = mask_bw * h_esv + h_exv_fw
            if use_checkpoint:
                h_v = torch.utils.checkpoint.checkpoint(layer, h_v, h_esv, mask)
            else:
                h_v = layer(h_v, h_esv, mask)

        logits = self.W_out(h_v)
        if return_log_prob:
            return nn.functional.log_softmax(logits, dim=-1)
        return logits.permute(0, 2, 1)
