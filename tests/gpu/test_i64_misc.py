"""64-bit address arithmetic in the MPNN / pair-embedding / smooth-lDDT Triton kernels (I64 audit, part 2).

The fixes are scalar: each program's (or each tile's) first row is formed in int64 and folded into the pointers, the in-tile offsets stay
int32, so there is no WIDE switch to force. No served case reaches 2**31 elements in these families (ProteinMPNN: 8192 residues x 48
neighbours x 128 = 5e7; the pair embeddings / token-pair init: 5 x 384 x 384 x 256 = 1.9e8), so their module accuracy is covered at
small sizes by ``test_accuracy_smoke.py``. Here the MPNN kernels whose addressing changed run directly at a size past 2**31 elements and
the rows at the high offsets are checked against a float32 reference computed on those rows only.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

W, K = 128, 48
G = 352_000  # groups of 48 edges x 128 channels: 2.16e9 elements > 2**31
assert G * K * W > 2**31


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


requires_card = pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")


def _close(out: torch.Tensor, ref: torch.Tensor, what: str, rel: float = 2e-2) -> None:
    err = float((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6))
    assert err <= rel, f"{what}: max error {err:.3e} of the reference's max"


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _weights(gen: torch.Generator, n: int):
    return [torch.randn(W, W, device="cuda", generator=gen) / W**0.5 for _ in range(n)], [
        0.2 * torch.randn(W, device="cuda", generator=gen) for _ in range(n)
    ]


@requires_card
def test_edge_dropout_pack_and_backward_past_2_31() -> None:
    from miniworld_engine.kernels.mpnn_edge_dropout.triton.main import _backward_op, _pack_op

    n = 2**31 + 4096
    gen = torch.Generator(device="cuda").manual_seed(0)
    mask = torch.rand(n, device="cuda", generator=gen) > 0.25
    grad = torch.randn(n, device="cuda", generator=gen).to(torch.bfloat16)
    packed = _pack_op(mask)
    out = _backward_op(grad, packed, 4.0 / 3.0)
    for s in (slice(0, 8192), slice(n - 8192, n)):
        ref = torch.where(mask[s], grad[s].float() * (4.0 / 3.0), 0.0)
        _close(out[s], ref, f"dropout backward {s}", 1e-2)


@requires_card
def test_edge_mlp_forward_past_2_31() -> None:
    from miniworld_engine.kernels.mpnn_edge_mlp.triton.main import _forward_impl

    rows = G * K
    gen = torch.Generator(device="cuda").manual_seed(1)
    (wh, wo), (bh, bo) = _weights(gen, 2)
    x = torch.randn(rows, W, device="cuda", generator=gen).to(torch.bfloat16)
    out = _forward_impl(x, wh.to(torch.bfloat16), bh.to(torch.bfloat16), wo.to(torch.bfloat16), bo.to(torch.bfloat16))
    for s in (slice(0, 256), slice(rows - 256, rows)):
        h1 = _bf(F.gelu(x[s].float()))
        h2 = _bf(F.gelu(_bf(h1 @ _bf(wh).t() + _bf(bh))))
        _close(out[s], h2 @ _bf(wo).t() + _bf(bo), f"edge MLP rows {s}")


@requires_card
def test_message_inference_past_2_31() -> None:
    from miniworld_engine.kernels.mpnn_message.triton.inference import triton_message_hidden_reduce_inference

    gen = torch.Generator(device="cuda").manual_seed(2)
    (wt,), (b,) = _weights(gen, 1)
    x = torch.randn(G, K, W, device="cuda", generator=gen).to(torch.bfloat16)
    mask = (torch.rand(G, K, device="cuda", generator=gen) > 0.2).to(torch.bfloat16)
    out = triton_message_hidden_reduce_inference(x, wt.to(torch.bfloat16), b.to(torch.bfloat16), mask, 30)
    for s in (slice(0, 16), slice(G - 16, G)):
        hidden = _bf(F.gelu(_bf(_bf(F.gelu(x[s].float())) @ _bf(wt).t() + _bf(b))))
        _close(out[s], (hidden * mask[s].float()[..., None]).sum(1) / 30, f"message groups {s}")


def _node_inputs(gen: torch.Generator):
    nodes = G
    edge = torch.randn(G, K, W, device="cuda", generator=gen).to(torch.bfloat16)
    query = torch.randn(G, W, device="cuda", generator=gen).to(torch.bfloat16)
    neighbor = torch.randn(nodes, W, device="cuda", generator=gen).to(torch.bfloat16)
    index = torch.randint(0, nodes, (G * K,), device="cuda", generator=gen)
    mask = (torch.rand(G * K, device="cuda", generator=gen) > 0.2).to(torch.bfloat16)
    return edge, query, neighbor, index, mask


def _preactivation(edge, query, neighbor, index, we):
    """bf16(bf16(query + bf16(edge We^T)) + neighbor[index]) for edge [g, K, W] (the kernels' rounding boundaries)."""
    projected = _bf(edge.float() @ _bf(we).t())
    return _bf(_bf(query.float()[:, None, :] + projected) + neighbor[index].float().view(edge.shape))


@requires_card
def test_node_message_forward_past_2_31() -> None:
    from miniworld_engine.kernels.mpnn_node_message.triton.main import _forward_op

    gen = torch.Generator(device="cuda").manual_seed(3)
    edge, query, neighbor, index, mask = _node_inputs(gen)
    (we, wh), (_, bh) = _weights(gen, 2)
    w16 = [t.to(torch.bfloat16) for t in (we, wh, bh)]
    out = _forward_op(edge, query, neighbor, index, w16[0], w16[1], w16[2], mask, 30)
    for s in (slice(0, 16), slice(G - 16, G)):
        rows = torch.arange(s.start * K, s.stop * K, device="cuda")
        pre = _preactivation(edge[s], query[s], neighbor, index[rows], we)
        hidden = _bf(_bf(F.gelu(pre)) @ _bf(wh).t() + _bf(bh))
        gated = _bf(F.gelu(hidden))
        _close(out[s], (gated * mask[rows].float().view(-1, K)[..., None]).sum(1) / 30, f"node message groups {s}")


@requires_card
def test_edge_tail_forward_past_2_31() -> None:
    from miniworld_engine.kernels.mpnn_edge_tail.triton.main import _forward_op

    gen = torch.Generator(device="cuda").manual_seed(4)
    edge, query, neighbor, index, _ = _node_inputs(gen)
    (we, wh, wo), (_, bh, bo) = _weights(gen, 3)
    nw = 1 + 0.1 * torch.randn(W, device="cuda", generator=gen)
    nb = 0.1 * torch.randn(W, device="cuda", generator=gen)
    w16 = [t.to(torch.bfloat16) for t in (we, wh, bh, wo, bo)]
    seed = torch.zeros(1, device="cuda", dtype=torch.int64)
    out = _forward_op(edge, query, neighbor, index, *w16, nw, nb, seed, 1e-5, 0.0)  # pyright: ignore[reportCallIssue]
    for s in (slice(0, 16), slice(G - 16, G)):
        rows = torch.arange(s.start * K, s.stop * K, device="cuda")
        pre = _preactivation(edge[s], query[s], neighbor, index[rows], we)
        hidden = _bf(_bf(F.gelu(pre)) @ _bf(wh).t() + _bf(bh))
        update = _bf(_bf(F.gelu(hidden)) @ _bf(wo).t() + _bf(bo))
        values = _bf(edge[s].float() + update)
        ref = F.layer_norm(values, (W,), eps=1e-5) * nw + nb
        _close(out[s], ref, f"edge tail groups {s}")


@requires_card
def test_relative_position_reduce_past_2_31() -> None:
    from miniworld_engine.kernels.mpnn_relative_position.triton.main import triton_bucket_reduce

    width, buckets = 16, 66
    rows = 2**31 // width + 4096
    gen = torch.Generator(device="cuda").manual_seed(5)
    grad = torch.randn(rows, width, device="cuda", generator=gen).to(torch.bfloat16)
    bucket = torch.randint(0, buckets, (rows,), device="cuda", generator=gen)
    table, bias = triton_bucket_reduce(grad, bucket, buckets)
    ref_table = torch.zeros(buckets, width, device="cuda", dtype=torch.float64)
    ref_bias = torch.zeros(width, device="cuda", dtype=torch.float64)
    step = 2**24
    for r0 in range(0, rows, step):
        g = grad[r0 : r0 + step].double()
        ref_table.index_add_(0, bucket[r0 : r0 + step], g)
        ref_bias += g.sum(0)
    _close(table, ref_table, "relative-position table", 1e-3)
    _close(bias, ref_bias, "relative-position bias", 1e-3)
