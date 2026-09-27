from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.mpnn_node_message import (
    node_message_reduce,
    node_message_reduce_pytorch,
    node_message_supported,
)


_WIDTH = 128
_GRADIENT_NAMES = (
    "edge_states",
    "node_states",
    "packed_weight",
    "packed_bias",
    "hidden_weight",
    "hidden_bias",
)


def _leaves(batch: int, length: int, neighbors: int, *, double: bool):
    generator = torch.Generator(device="cuda").manual_seed(17)

    def normal(*shape, scale: float):
        return torch.randn(*shape, device="cuda", generator=generator) * scale

    edge = normal(batch, length, neighbors, _WIDTH, scale=0.5)
    # The kernel consumes a BF16 edge tensor, so the exact evaluation starts from the
    # same rounded values instead of charging the kernel for input rounding.
    values = [
        edge.to(torch.bfloat16).float(),
        normal(batch, length, _WIDTH, scale=0.5),
        normal(_WIDTH, 3 * _WIDTH, scale=0.05),
        normal(_WIDTH, scale=0.05),
        normal(_WIDTH, _WIDTH, scale=0.08),
        normal(_WIDTH, scale=0.05),
    ]
    return [
        (value.double() if double else value.clone()).requires_grad_(True)
        for value in values
    ]


def _evaluate(made, indices, edge_mask, neighbors, *, fused: bool):
    edge, node, packed, packed_bias, hidden_weight, hidden_bias = made
    query = F.linear(node, packed[:, :_WIDTH], packed_bias)
    neighbor = F.linear(node, packed[:, 2 * _WIDTH :])
    edge_weight = packed[:, _WIDTH : 2 * _WIDTH]
    if fused:
        return node_message_reduce(
            edge.to(torch.bfloat16),
            query,
            neighbor,
            indices,
            edge_weight,
            hidden_weight,
            hidden_bias,
            edge_mask.to(torch.bfloat16),
            neighbors,
        )
    return node_message_reduce_pytorch(
        edge,
        query,
        neighbor,
        indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask.double() if edge.dtype == torch.float64 else edge_mask,
        neighbors,
    )


def _relative_error(value: torch.Tensor, exact: torch.Tensor) -> float:
    scale = exact.float().abs().mean().clamp_min(1e-12)
    return ((value.float() - exact.float()).abs().mean() / scale).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_node_message_is_no_less_accurate_than_the_pytorch_chain() -> None:
    """One fused pass must not lose accuracy against four separate operations.

    Both paths are compared against FP64, so this asserts a property of the kernel
    rather than agreement between two equally approximate BF16 evaluations.
    """
    batch, length, neighbors = 2, 256, 48
    torch.manual_seed(17)
    indices = torch.randint(
        0, batch * length, (batch, length, neighbors), device="cuda"
    )
    edge_mask = (
        torch.rand(batch, length, neighbors, device="cuda") > 0.1
    ).float()

    fused_leaves = _leaves(batch, length, neighbors, double=False)
    torch_leaves = _leaves(batch, length, neighbors, double=False)
    exact_leaves = _leaves(batch, length, neighbors, double=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused = _evaluate(fused_leaves, indices, edge_mask, neighbors, fused=True)
        reference = _evaluate(
            torch_leaves, indices, edge_mask, neighbors, fused=False
        )
    exact = _evaluate(exact_leaves, indices, edge_mask, neighbors, fused=False)

    upstream = torch.randn(fused.shape, device="cuda")
    fused.float().mul(upstream).sum().backward()
    reference.float().mul(upstream).sum().backward()
    exact.mul(upstream.double()).sum().backward()

    errors = {
        "reduced": (_relative_error(fused, exact), _relative_error(reference, exact))
    }
    for name, made, other, truth in zip(
        _GRADIENT_NAMES, fused_leaves, torch_leaves, exact_leaves
    ):
        errors[name] = (
            _relative_error(made.grad, truth.grad),
            _relative_error(other.grad, truth.grad),
        )
    failures = {
        name: pair
        for name, pair in errors.items()
        if pair[0] > max(1.5 * pair[1], 5e-3)
    }
    assert not failures, f"fused path is less accurate: {failures} (all: {errors})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_node_message_rejects_contracts_it_cannot_honor() -> None:
    batch, length, neighbors = 1, 64, 48
    made = _leaves(batch, length, neighbors, double=False)
    indices = torch.randint(0, length, (batch, length, neighbors), device="cuda")
    edge_mask = torch.ones(batch, length, neighbors, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        query = F.linear(made[1], made[2][:, :_WIDTH], made[3])
        neighbor = F.linear(made[1], made[2][:, 2 * _WIDTH :])
    arguments = (
        query,
        neighbor,
        indices,
        made[2][:, _WIDTH : 2 * _WIDTH],
        made[4],
        made[5],
        edge_mask,
    )
    # FP32 edge states: the fused path takes a BF16 edge tensor by contract.
    assert not node_message_supported(made[0], *arguments)
    with pytest.raises(ValueError, match="requires contiguous CUDA BF16"):
        node_message_reduce(made[0], *arguments, neighbors)
    # A non-contiguous edge tensor would make the flat row indexing wrong.
    assert not node_message_supported(
        made[0].to(torch.bfloat16).transpose(1, 2), *arguments
    )
