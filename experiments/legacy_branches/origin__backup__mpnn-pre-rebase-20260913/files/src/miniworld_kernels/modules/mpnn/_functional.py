"""Internal tensor primitives shared by the production MPNN modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.weak import WeakTensorKeyDictionary


def lecun_normal_(module: nn.Module, scale: float = 1.0) -> None:
    """Initialize ``module.weight`` with the truncated LeCun distribution."""
    normal = torch.distributions.normal.Normal(0, 1)
    lower_cdf = normal.cdf(torch.tensor(-2.0))
    upper_cdf = normal.cdf(torch.tensor(2.0))
    probabilities = lower_cdf + (upper_cdf - lower_cdf) * torch.rand_like(module.weight)
    values = math.sqrt(2.0) * torch.erfinv(
        torch.clamp(2.0 * probabilities - 1.0, -1.0 + 1e-8, 1.0 - 1e-8)
    )
    values.clamp_(-2.0, 2.0)
    stddev = math.sqrt(scale / module.weight.shape[-1]) / 0.87962566103423978
    with torch.no_grad():
        module.weight.copy_(stddev * values)


_FLAT_NEIGHBOR_CACHE: WeakTensorKeyDictionary = WeakTensorKeyDictionary()


def _flat_neighbor_indices(
    neighbor_indices: torch.Tensor, batch: int, length: int
) -> torch.Tensor:
    """Return ``neighbor_indices`` shifted into a flattened ``[B*L]`` node axis.

    Every projection in every layer gathers with the same index tensor, so the
    shifted copy is built once per distinct index tensor rather than once per
    call. At crop 2048 with K=48 one copy is 25 MiB at B=32, and ``F.embedding``
    keeps each one alive for backward. Inductor already eliminates the duplicate
    addition inside a compiled graph, so the memo only covers eager execution
    and is skipped while tracing, where a Python-level cache would be an
    invisible guard.
    """
    batch_offsets = (torch.arange(batch, device=neighbor_indices.device) * length).view(
        batch, 1, 1
    )
    if torch.compiler.is_compiling():
        return neighbor_indices + batch_offsets
    cached = _FLAT_NEIGHBOR_CACHE.get(neighbor_indices)
    if cached is not None:
        version, flat = cached
        if (
            version == neighbor_indices._version
            and flat.shape == neighbor_indices.shape
        ):
            return flat
    flat = neighbor_indices + batch_offsets
    _FLAT_NEIGHBOR_CACHE[neighbor_indices] = (neighbor_indices._version, flat)
    return flat


def gather_neighbors(
    values: torch.Tensor, neighbor_indices: torch.Tensor
) -> torch.Tensor:
    """Gather node values at ``neighbor_indices`` with a fusible backward."""
    batch, length, channels = values.shape
    if values.is_floating_point():
        if batch == 1:
            return F.embedding(neighbor_indices, values[0])
        flattened = values.reshape(batch * length, channels)
        return F.embedding(
            _flat_neighbor_indices(neighbor_indices, batch, length), flattened
        )

    flattened_indices = neighbor_indices.reshape(batch, -1)
    flattened_indices = flattened_indices.unsqueeze(-1).expand(-1, -1, channels)
    gathered = torch.gather(values, 1, flattened_indices)
    return gathered.reshape(*neighbor_indices.shape, channels)


def concatenate_neighbor_features(
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """Concatenate edge features with their destination-node features."""
    return torch.cat(
        (edge_features, gather_neighbors(node_features, neighbor_indices)), dim=-1
    )


def project_packed_neighbor_inputs(
    query_nodes: torch.Tensor,
    edge_features: torch.Tensor,
    neighbor_nodes: torch.Tensor,
    neighbor_indices: torch.Tensor,
    projection: nn.Linear,
    node_width: int,
    edge_width: int,
) -> torch.Tensor:
    """Apply a packed ``[query, edge, neighbor]`` projection by blocks."""
    weight = projection.weight
    output = F.linear(query_nodes, weight[:, :node_width], projection.bias).unsqueeze(2)
    output = output + F.linear(
        edge_features, weight[:, node_width : node_width + edge_width]
    )
    neighbor_projection = F.linear(neighbor_nodes, weight[:, node_width + edge_width :])
    return output + gather_neighbors(neighbor_projection, neighbor_indices)
