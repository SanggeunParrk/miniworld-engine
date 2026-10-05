"""ProteinMPNN message-side kernels on A100 (integrations/mpnn_msg_sm80.py): the hidden-message reduction, the fused encoder node message and the relative-position backward,
against an fp64 evaluation. Errors are held to the bf16 PyTorch path's own error in the same regime (ratio <= 1.1 plus a small absolute floor); the CUDA paths are deterministic (bit-for-bit repeatable),
equal under ``torch.compile(fullgraph=True)`` and under CUDA-graph capture, and switched off by ``MINIWORLD_MPNN_MSG_SM80=0``."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.integrations import mpnn_msg_sm80 as sm80
from miniworld_engine.kernels.mpnn_message import (
    message_hidden_reduce,
    message_hidden_reduce_pytorch,
)
from miniworld_engine.kernels.mpnn_node_message import (
    node_message_reduce,
    node_message_reduce_pytorch,
)
from miniworld_engine.kernels.mpnn_relative_position import (
    relative_position_embed,
    relative_position_embed_pytorch,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.detach().double() - b.detach().double()).norm() / b.detach().double().norm().clamp_min(1e-30))


def _within_bf16_error(error, bf16_error, floor=2e-4):
    return error <= 1.1 * bf16_error + floor


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- relative position
def _rp_inputs(rows, dtype, buckets=66, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bucket = torch.randint(0, buckets, (rows,), device="cuda", generator=g)
    heavy = torch.rand(rows, device="cuda", generator=g) < 0.33              # a third of the edges in the two clamp buckets: the skew of the real graph
    bucket = torch.where(heavy, torch.where(torch.rand(rows, device="cuda", generator=g) < 0.5, 0, buckets - 2), bucket)
    table = torch.randn(buckets, 16, device="cuda", generator=g).to(dtype)
    bias = torch.randn(16, device="cuda", generator=g).to(dtype)
    grad = torch.randn(rows, 16, device="cuda", generator=g).to(dtype)
    return bucket, table, bias, grad


@pytest.mark.parametrize("dtype", [torch.float32, BF16])
@pytest.mark.parametrize("rows", [1, 15, 17, 5000, 300_000])
def test_relative_position_matches_fp64_and_repeats(rows, dtype):
    bucket, table, bias, grad = _rp_inputs(rows, dtype)
    assert sm80.serves_relpos(bucket, table, bias)
    results = []
    for _ in range(2):
        t, b = table.detach().clone().requires_grad_(), bias.detach().clone().requires_grad_()
        out = relative_position_embed(bucket, t, b, backend="triton")
        out.backward(grad)
        results.append((out.detach(), t.grad, b.grad))
    assert results[0][1].dtype == dtype
    assert results[0][2].dtype == dtype
    assert torch.equal(results[0][1], results[1][1])                                                                # bit-reproducible
    assert torch.equal(results[0][2], results[1][2])
    assert torch.equal(results[0][0], relative_position_embed_pytorch(bucket, table, bias))                          # the forward is the plain lookup
    exact_table = torch.zeros(66, 16, device="cuda", dtype=torch.float64).index_add_(0, bucket, grad.double())
    exact_bias = grad.double().sum(0)
    # the gradient leaves in the table's dtype: bf16 rounds it once (2^-9), fp32 keeps ~1e-6
    tolerance = 1e-2 if dtype == BF16 else 1e-5
    assert _rel(results[0][1], exact_table) < tolerance
    assert _rel(results[0][2], exact_bias) < tolerance


def test_relative_position_unused_buckets_are_exact_zero_and_gate():
    bucket = torch.randint(0, 40, (5000,), device="cuda")
    table = torch.randn(66, 16, device="cuda", requires_grad=True)
    bias = torch.randn(16, device="cuda", requires_grad=True)
    relative_position_embed(bucket, table, bias, backend="triton").backward(torch.randn(5000, 16, device="cuda"))
    assert torch.equal(table.grad[40:], torch.zeros_like(table.grad[40:]))
    # the gate: a 16-channel table of at most 79 buckets; everything else keeps the Triton reduction
    assert sm80.serves_relpos(bucket, table, bias)
    assert not sm80.serves_relpos(bucket, torch.randn(66, 32, device="cuda"), torch.randn(32, device="cuda"))
    assert not sm80.serves_relpos(bucket, torch.randn(80, 16, device="cuda"), bias)
    assert not sm80.serves_relpos(bucket.cpu(), table.cpu(), bias.cpu())


def test_relative_position_is_a_compiled_and_graphed_op():
    bucket, table, bias, grad = _rp_inputs(20_000, BF16)

    def step(t, b):
        out = relative_position_embed(bucket, t, b, backend="triton")
        out.backward(grad)
        return t.grad, b.grad

    t, b = table.clone().requires_grad_(), bias.clone().requires_grad_()
    eager = step(t, b)
    t2, b2 = table.clone().requires_grad_(), bias.clone().requires_grad_()

    def compiled_step(t, b):
        out = torch.compile(lambda t, b: relative_position_embed(bucket, t, b, backend="triton"), fullgraph=True)(t, b)
        out.backward(grad)
        return t.grad, b.grad

    compiled = compiled_step(t2, b2)
    assert torch.equal(eager[0], compiled[0])
    assert torch.equal(eager[1], compiled[1])
    # CUDA graph capture + replay of the whole forward + backward
    t3, b3 = table.clone().requires_grad_(), bias.clone().requires_grad_()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step(t3, b3)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    t3.grad = b3.grad = None
    with torch.cuda.graph(graph):
        out = relative_position_embed(bucket, t3, b3, backend="triton")
        grads = torch.autograd.grad(out, (t3, b3), grad)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(grads[0], eager[0])
    assert torch.equal(grads[1], eager[1])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- hidden message
def _msg_inputs(groups, mixed, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    p = torch.randn(groups, 48, 128, device="cuda", generator=g).to(BF16)
    w = torch.randn(128, 128, device="cuda", generator=g) / 128 ** 0.5
    b = torch.randn(128, device="cuda", generator=g) * 0.1
    if not mixed:
        w, b = w.to(BF16), b.to(BF16)
    mask = (torch.rand(groups, 48, device="cuda", generator=g) > 0.2).float()
    mask[0].zero_()
    up = torch.randn(groups, 128, device="cuda", generator=g)
    return p, w, b, mask, up


def _msg_run(p, w, b, mask, up, backend, mixed, fn=None):
    leaves = [t.detach().clone().requires_grad_() for t in (p, w, b)]
    with torch.autocast("cuda", dtype=BF16, enabled=mixed):
        out = (fn or message_hidden_reduce)(*leaves, mask, 48, backend=backend)
    grads = torch.autograd.grad(out, leaves, up)
    return out.detach(), grads


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("backend", ["auto", "triton_compute", "triton_memory"])
@pytest.mark.parametrize("groups", [1, 3, 17, 130])
def test_message_matches_fp64_as_well_as_the_bf16_path(groups, backend, mixed):
    p, w, b, mask, up = _msg_inputs(groups, mixed)
    assert sm80.serves_message(p, backend)
    out, grads = _msg_run(p, w, b, mask, up, backend, mixed)
    leaves64 = [t.detach().double().requires_grad_() for t in (p, w, b)]
    out64 = message_hidden_reduce_pytorch(*leaves64, mask.double(), 48)
    grads64 = torch.autograd.grad(out64, leaves64, up.double())
    leaves_b = [t.detach().clone().requires_grad_() for t in (p, w, b)]
    with torch.autocast("cuda", dtype=BF16, enabled=mixed):
        out_b = message_hidden_reduce_pytorch(*leaves_b, mask, 48)
    grads_b = torch.autograd.grad(out_b, leaves_b, up)
    assert out.dtype == torch.float32
    assert [g.dtype for g in grads] == [p.dtype, w.dtype, b.dtype]
    assert _within_bf16_error(_rel(out, out64), _rel(out_b, out64))
    for name, got, base, want in zip("pwb", grads, grads_b, grads64, strict=True):
        assert _within_bf16_error(_rel(got, want), _rel(base, want), floor=1e-3), name


def test_message_repeats_bit_for_bit_and_compiled_graphed_equal_eager():
    p, w, b, mask, up = _msg_inputs(200, True)
    first = _msg_run(p, w, b, mask, up, "triton_compute", True)
    again = _msg_run(p, w, b, mask, up, "triton_compute", True)
    assert torch.equal(first[0], again[0])
    assert all(torch.equal(x, y) for x, y in zip(first[1], again[1], strict=True))
    compiled_fn = torch.compile(lambda *a, **k: message_hidden_reduce(*a, **k), fullgraph=True)
    compiled = _msg_run(p, w, b, mask, up, "triton_compute", True, fn=compiled_fn)
    assert torch.equal(first[0], compiled[0])
    assert all(torch.equal(x, y) for x, y in zip(first[1], compiled[1], strict=True))
    # CUDA-graph capture + replay of forward + backward
    leaves = [t.detach().clone().requires_grad_() for t in (p, w, b)]

    def step():
        with torch.autocast("cuda", dtype=BF16):
            out = message_hidden_reduce(*leaves, mask, 48, backend="triton_compute")
        return out, torch.autograd.grad(out, leaves, up)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, grads = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first[0])
    assert all(torch.equal(x, y) for x, y in zip(grads, first[1], strict=True))


def test_message_fused_and_split_backward_agree(monkeypatch):
    """The two-step form (kernel + cuBLAS dW, ``MINIWORLD_MPNN_MSG_SM80_BWD=split``) stays reachable: same dP bit for bit, dW / db the same fp32 sums in another order."""
    from miniworld_engine.kernels.mpnn_message.cuda import sm80 as kernels

    p, w, b, mask, up = _msg_inputs(130, False)
    flat_p, flat_mask = p.reshape(-1, 128), mask.reshape(-1)
    fused = kernels.backward(flat_p, w, b, flat_mask, up, 48, fused=True)
    split = kernels.backward(flat_p, w, b, flat_mask, up, 48, fused=False)
    assert torch.equal(fused[0], split[0])
    assert _rel(fused[1], split[1]) < 1e-5
    assert _rel(fused[2], split[2]) < 1e-5
    monkeypatch.setenv("MINIWORLD_MPNN_MSG_SM80_BWD", "split")
    switched = kernels.backward(flat_p, w, b, flat_mask, up, 48)
    assert all(torch.equal(x, y) for x, y in zip(switched, split, strict=True))


def test_message_inference_runs_the_single_kernel_and_the_switch_keeps_triton(monkeypatch):
    p, w, b, mask, _ = _msg_inputs(64, False)
    with torch.no_grad():
        out = message_hidden_reduce(p, w, b, mask, 48, backend="auto")
        out64 = message_hidden_reduce_pytorch(p.double(), w.double(), b.double(), mask.double(), 48)
        assert _rel(out, out64) < 3e-3
        monkeypatch.setenv("MINIWORLD_MPNN_MSG_SM80", "0")
        assert not sm80.serves_message(p, "auto")
        assert _rel(message_hidden_reduce(p, w, b, mask, 48, backend="auto"), out64) < 3e-3          # the Triton kernel still serves
        monkeypatch.delenv("MINIWORLD_MPNN_MSG_SM80")
        assert sm80.serves_message(p, "auto")
        assert not sm80.serves_message(p, "pytorch")
        previous = settings.configure(engine_backend="triton")
        try:
            assert not sm80.serves_message(p, "auto")                                                  # a forced Triton engine backend keeps Triton
        finally:
            settings.configure(engine_backend=previous.engine_backend)
    assert not sm80.serves_message(p.cpu(), "auto")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- node message
def _node_inputs(batch, length, neighbors, mixed, mask_dtype=torch.float32, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=g) * scale

    edge = r(batch, length, neighbors, 128).to(BF16)
    query = r(batch, length, 128).to(BF16)
    nb = r(batch, length, 128).to(BF16)
    index = torch.randint(0, batch * length, (batch, length, neighbors), device="cuda", generator=g)
    index[..., ::2] = index[..., :1].clone()                                  # repeated neighbours: the scatter has collisions
    packed = r(128, 384, scale=128 ** -0.5)
    hidden = r(128, 128, scale=128 ** -0.5)
    bias = r(128, scale=0.1)
    if not mixed:
        packed, hidden, bias = packed.to(BF16), hidden.to(BF16), bias.to(BF16)
    mask = (torch.rand(batch, length, neighbors, device="cuda", generator=g) > 0.2).to(mask_dtype)
    mask[:, 0] = 0
    up = r(batch, length, 128, scale=0.05)
    return edge, query, nb, index, packed, hidden, bias, mask, up


def _node_run(edge, query, nb, index, packed, hidden, bias, mask, up, backend, mixed, scale=48, fn=None):
    leaves = [t.detach().clone().requires_grad_() for t in (edge, query, nb, packed, hidden, bias)]
    e, q, n, pk, hw, hb = leaves
    with torch.autocast("cuda", dtype=BF16, enabled=mixed):
        out = (fn or node_message_reduce)(e, q, n, index, pk[:, 128:256], hw, hb, mask, scale, backend=backend)
    grads = torch.autograd.grad(out, leaves, up)
    return out.detach(), grads


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("backend", ["triton", "triton_compute"])
@pytest.mark.parametrize("shape", [(1, 17, 48), (2, 64, 48), (1, 40, 16), (1, 21, 17), (1, 33, 1), (1, 20, 128)])
def test_node_message_matches_fp64_as_well_as_the_bf16_path(shape, backend, mixed):
    batch, length, neighbors = shape
    inputs = _node_inputs(batch, length, neighbors, mixed, mask_dtype=BF16 if batch == 2 else torch.float32)
    edge, query, nb, index, packed, hidden, bias, mask, up = inputs
    out, grads = _node_run(*inputs, backend, mixed)
    # the fp64 evaluation and the bf16 module path of the same operands
    leaves64 = [t.detach().double().requires_grad_() for t in (edge, query, nb, packed, hidden, bias)]
    e6, q6, n6, pk6, hw6, hb6 = leaves64
    out64 = node_message_reduce_pytorch(e6, q6, n6, index, pk6[:, 128:256], hw6, hb6, mask.double(), 48)
    grads64 = torch.autograd.grad(out64, leaves64, up.double())
    leaves_b = [t.detach().clone().requires_grad_() for t in (edge, query, nb, packed, hidden, bias)]
    eb, qb, nbb, pkb, hwb, hbb = leaves_b
    with torch.autocast("cuda", dtype=BF16, enabled=mixed):
        out_b = node_message_reduce_pytorch(eb, qb, nbb, index, pkb[:, 128:256], hwb, hbb, mask.float(), 48)
    grads_b = torch.autograd.grad(out_b, leaves_b, up)
    assert sm80.serves_node(edge, backend)
    assert _within_bf16_error(_rel(out, out64), _rel(out_b, out64))
    for name, got, base, want in zip(("edge", "query", "neighbor", "packed", "hidden", "bias"), grads, grads_b, grads64, strict=True):
        assert got.dtype == base.dtype, name
        assert _within_bf16_error(_rel(got, want), _rel(base, want), floor=2e-3), name


def test_node_message_repeats_bit_for_bit_and_compiled_graphed_equal_eager(monkeypatch):
    inputs = _node_inputs(1, 200, 48, True)
    first = _node_run(*inputs, "triton_compute", True)
    again = _node_run(*inputs, "triton_compute", True)
    assert torch.equal(first[0], again[0])
    assert all(torch.equal(x, y) for x, y in zip(first[1], again[1], strict=True))
    compiled_fn = torch.compile(lambda *a, **k: node_message_reduce(*a, **k), fullgraph=True)
    compiled = _node_run(*inputs, "triton_compute", True, fn=compiled_fn)
    assert torch.equal(first[0], compiled[0])
    assert all(torch.equal(x, y) for x, y in zip(first[1], compiled[1], strict=True))
    edge, query, nb, index, packed, hidden, bias, mask, up = inputs
    leaves = [t.detach().clone().requires_grad_() for t in (edge, query, nb, packed, hidden, bias)]

    def step():
        e, q, n, pk, hw, hb = leaves
        with torch.autocast("cuda", dtype=BF16):
            out = node_message_reduce(e, q, n, index, pk[:, 128:256], hw, hb, mask, 48, backend="triton_compute")
        return out, torch.autograd.grad(out, leaves, up)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, grads = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first[0])
    assert all(torch.equal(x, y) for x, y in zip(grads, first[1], strict=True))
    monkeypatch.setenv("MINIWORLD_MPNN_MSG_SM80", "0")
    assert not sm80.serves_node(edge, "triton")


def test_node_message_inference_matches_the_reference():
    edge, query, nb, index, packed, hidden, bias, mask, _ = _node_inputs(1, 130, 48, False)
    with torch.no_grad():
        out = node_message_reduce(edge, query, nb, index, packed[:, 128:256], hidden, bias, mask, 48, backend="triton_compute")
        out64 = node_message_reduce_pytorch(edge.double(), query.double(), nb.double(), index, packed[:, 128:256].double(), hidden.double(), bias.double(), mask.double(), 48)
    assert out.dtype == torch.float32
    assert _rel(out, out64) < 3e-3


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- the model
def test_proteinmpnn_with_the_cuda_message_side_matches_the_pytorch_model():
    from miniworld_engine.modules.mpnn import ProteinMPNN, ProteinMPNNConfig

    common = {"node_width": 128, "edge_width": 128, "hidden_width": 128, "encoder_depth": 2, "decoder_depth": 2, "k_neighbors": 48, "coordinate_noise": 0.0, "dropout": 0.0,
              "block_linear_min_edges": 0, "edge_mlp_backend": "pytorch", "edge_norm_backend": "pytorch", "edge_dropout_backend": "pytorch"}
    torch.manual_seed(19)
    reference = ProteinMPNN(ProteinMPNNConfig(**common, message_backend="pytorch")).cuda()
    nn.init.normal_(reference.output_projection.weight, std=128 ** -0.5)
    candidate = ProteinMPNN(ProteinMPNNConfig(**common, message_backend="triton_compute", node_message_backend="triton_compute", relative_position_backend="triton")).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference.train()
    candidate.train()
    batch, length = 2, 96
    backbone = torch.randn(batch, length, 4, 3, device="cuda") * 3.0
    sequence = torch.randint(0, 21, (batch, length), device="cuda")
    residue_mask = torch.ones(batch, length, device="cuda")
    residue_index = torch.arange(length, device="cuda").expand(batch, -1)
    chain_index = torch.zeros(batch, length, dtype=torch.long, device="cuda")
    decoding_order = torch.stack([torch.randperm(length, device="cuda") for _ in range(batch)])
    patch_index = (torch.arange(length, device="cuda") // 8).expand(batch, -1)

    def run(model, coordinates):
        with torch.autocast("cuda", dtype=BF16):
            return model(coordinates, sequence, residue_mask, residue_index, chain_index, decoding_order, patch_index)

    expected_in = backbone.clone().requires_grad_()
    actual_in = backbone.clone().requires_grad_()
    expected, actual = run(reference, expected_in), run(candidate, actual_in)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    flat = lambda m: torch.cat([p.grad.detach().float().flatten() for p in m.parameters()])
    relative = (flat(candidate) - flat(reference)).norm() / flat(reference).norm().clamp_min(1e-12)
    cosine = F.cosine_similarity(flat(candidate), flat(reference), dim=0)
    assert relative < 0.02
    assert cosine > 0.999
