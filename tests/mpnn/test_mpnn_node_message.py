from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# Every test here launches a kernel, so it needs a card. Without the marker pytest runs it
# on a login node and reports "Found no NVIDIA driver" as a FAILURE -- eleven red lines whose
# content is the absence of hardware, which is what conftest's skip exists to prevent.
pytestmark = pytest.mark.gpu
from miniworld_engine.kernels.mpnn_node_message import (
    NodeMessageBackend,
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


def _evaluate(made, indices, edge_mask, neighbors, *, fused: bool, backend: NodeMessageBackend = "triton"):
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
            backend=backend,
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
@pytest.mark.parametrize("backend", ["triton", "triton_compute"])
def test_mpnn_node_message_is_no_less_accurate_than_the_pytorch_chain(backend: NodeMessageBackend) -> None:
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
        fused = _evaluate(fused_leaves, indices, edge_mask, neighbors, fused=True, backend=backend)
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
        _GRADIENT_NAMES, fused_leaves, torch_leaves, exact_leaves, strict=False
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



@pytest.mark.parametrize("backend", ["triton", "triton_compute"])
@pytest.mark.parametrize("neighbors", [1, 48, 128])
@pytest.mark.parametrize("chunk_groups", [5, 262144])
def test_node_message_neighbor_reduction_handles_collisions_and_chunks(
    monkeypatch: pytest.MonkeyPatch, neighbors: int, chunk_groups: int, backend: NodeMessageBackend,
) -> None:
    """Masked queries, repeated neighbor IDs and a short final chunk keep all gradients."""
    import triton

    from miniworld_engine.kernels.mpnn_node_message.triton import main

    monkeypatch.setattr(main, "_WEIGHT_CHUNK_ROWS", chunk_groups * neighbors)
    from miniworld_engine.kernels.mpnn_message.triton import main as message
    monkeypatch.setattr(message, "_DX_CHUNK_ROWS", chunk_groups * neighbors)
    for name in ("_node_message_fwd_kernel", "_node_message_replay_kernel", "_node_message_dx_kernel"):
        tuner = getattr(main, name)
        monkeypatch.setattr(tuner, "configs", [
            triton.Config({"GROUPS": 1}, num_warps=4, num_stages=1),
        ])
        monkeypatch.setattr(tuner, "cache", {})
    torch.manual_seed(913)
    batch, nodes = 2, 17
    edge = torch.randn(batch, nodes, neighbors, 128, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(batch, nodes, 128, device="cuda", dtype=torch.bfloat16)
    neighbor = torch.randn_like(query)
    packed = torch.randn(128, 384, device="cuda") / 128**0.5
    weight = torch.randn(128, 128, device="cuda") / 128**0.5
    bias = torch.randn(128, device="cuda") * 0.1
    leaves = [edge, query, neighbor, packed, weight, bias]
    for leaf in leaves:
        leaf.requires_grad_(True)
    index = torch.randint(batch * nodes, (batch, nodes, neighbors), device="cuda")
    index[..., ::2] = 0
    mask = (torch.rand(batch, nodes, neighbors, device="cuda") > 0.2).float()
    mask[:, 0] = 0
    upstream = torch.randn(batch, nodes, 128, device="cuda") * 0.01
    arguments = (edge, query, neighbor, index, packed[:, 128:256], weight, bias, mask, neighbors)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = node_message_reduce(*arguments, backend=backend)
        expected = node_message_reduce_pytorch(*arguments)
    got = torch.autograd.grad(actual, leaves, upstream)
    want = torch.autograd.grad(expected, leaves, upstream)
    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]), rtol=0, atol=0)
    for a, b in zip((actual, *got), (expected, *want), strict=True):
        error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
        assert error.item() < 0.015



def test_node_message_build_driver_reaches_both_save_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary registry driver builds inference/memory and compute-forward keys."""
    import triton

    from miniworld_engine.kernels.drivers import mpnn_node_message as driver
    from miniworld_engine.kernels.mpnn_node_message.triton import main

    graph = driver._graph
    monkeypatch.setattr(driver, "_graph", lambda *, grad: graph(nodes=17, grad=grad))
    monkeypatch.setattr(driver, "_nodes", lambda: 17)
    tuner = main._node_message_fwd_kernel
    monkeypatch.setattr(tuner, "configs", [triton.Config({"GROUPS": 1}, num_warps=4, num_stages=1)])
    monkeypatch.setattr(tuner, "cache", {})
    run = tuner.run
    seen = set()

    def witnessed(*args, **kwargs):
        seen.add(kwargs["SAVE_PREACT"])
        return run(*args, **kwargs)

    monkeypatch.setattr(tuner, "run", witnessed)
    with torch.no_grad():
        driver.mpnn_node_message_fwd_gemm_triton()
    assert seen == {False, True}


def test_node_message_compute_inference_does_not_save_projections(monkeypatch: pytest.MonkeyPatch) -> None:
    from miniworld_engine.kernels.mpnn_node_message.triton import main

    def forbidden(*args, **kwargs):
        raise AssertionError("inference must not allocate saved projections")

    monkeypatch.setattr(main, "_compute_forward_op", forbidden)
    values = _leaves(1, 17, 48, double=False)
    index = torch.randint(17, (1, 17, 48), device="cuda")
    mask = torch.ones(1, 17, 48, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = _evaluate(values, index, mask, 48, fused=True, backend="triton_compute")
        reference = _evaluate(values, index, mask, 48, fused=False)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
    assert not actual.requires_grad


def test_node_message_compute_fullgraph_preserves_gradients() -> None:
    """AOTAutograd must preserve every gradient across the opaque forward boundary."""
    values = _leaves(1, 17, 48, double=False)
    index = torch.randint(17, (1, 17, 48), device="cuda")
    mask = (torch.rand(1, 17, 48, device="cuda") > 0.2).float()

    def forward(*leaves):
        return _evaluate(leaves, index, mask, 48, fused=True, backend="triton_compute")

    compiled = torch.compile(forward, fullgraph=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = forward(*values)
        actual = compiled(*values)
    upstream = torch.randn_like(expected) * 0.01
    expected_grads = torch.autograd.grad(expected, values, upstream)
    actual_grads = torch.autograd.grad(actual, values, upstream)
    for got, want in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)
        assert error.item() < 0.01
