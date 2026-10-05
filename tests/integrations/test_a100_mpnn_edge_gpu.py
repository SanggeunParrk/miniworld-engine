"""The ProteinMPNN edge side on A100 (``integrations/mpnn_edge_sm80.py``): the hand-CUDA edge tail, edge MLP, edge LayerNorm backward and edge dropout mask against an FP64
evaluation and the bf16 PyTorch chain, eager and compiled, replayed from a CUDA graph.  Errors are held to the bf16 PyTorch chain's own error in the same regime."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.integrations import mpnn_edge_sm80 as sm80
from miniworld_engine.kernels.mpnn_edge_dropout import edge_dropout
from miniworld_engine.kernels.mpnn_edge_layernorm import edge_layer_norm
from miniworld_engine.kernels.mpnn_edge_layernorm.reference import (
    edge_layer_norm_pytorch,
)
from miniworld_engine.kernels.mpnn_edge_mlp import (
    edge_mlp_update,
    edge_mlp_update_pytorch,
)
from miniworld_engine.kernels.mpnn_edge_tail import (
    edge_tail_update,
    edge_tail_update_pytorch,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

W = 128


@pytest.fixture(autouse=True)
def ampere(monkeypatch):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")
    monkeypatch.setenv(sm80.ENV, "1")


def _rel(a, b):
    return float((a.detach().double() - b.detach().double()).norm() / b.detach().double().norm().clamp_min(1e-30))


def _mean_rel(a, b):
    scale = b.detach().float().abs().mean().clamp_min(1e-12)
    return ((a.detach().float() - b.detach().float()).abs().mean() / scale).item()


def _keep_mask(seed, rows, p):
    return sm80._ext().dropout_mask(seed, rows, p)


# =========================================================================================================================================== edge tail
def _tail_inputs(batch, length, k, *, fp32_params, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def n(*shape, scale):
        return torch.randn(*shape, device="cuda", generator=g) * scale

    pdt = torch.float32 if fp32_params else torch.bfloat16
    edge = n(batch, length, k, W, scale=0.6).to(torch.bfloat16)
    node = n(batch, length, W, scale=0.5)
    packed = n(W, 3 * W, scale=0.05)
    packed_bias = n(W, scale=0.05)
    vals = {
        "edge": edge,
        "node": node.to(torch.bfloat16),
        "packed": packed.to(pdt),
        "packed_bias": packed_bias.to(pdt),
        "hidden_weight": n(W, W, scale=0.08).to(pdt),
        "hidden_bias": n(W, scale=0.05).to(pdt),
        "output_weight": n(W, W, scale=0.08).to(pdt),
        "output_bias": n(W, scale=0.05).to(pdt),
        "norm_weight": (torch.rand(W, device="cuda", generator=g) + 0.5),
        "norm_bias": n(W, scale=0.05),
    }
    index = torch.randint(0, batch * length, (batch, length, k), device="cuda", generator=g)
    return vals, index


def _tail_apply(v, index, seed, p, impl, *, autocast):
    """impl: 'cuda' | 'torch' (the bf16 reference chain, mask given) | 'exact' (fp64)."""
    node, packed, pb = v["node"], v["packed"], v["packed_bias"]
    query = F.linear(node, packed[:, :W], pb)
    neighbor = F.linear(node, packed[:, 2 * W:])
    edge_weight = packed[:, W:2 * W]
    if impl == "cuda":
        return edge_tail_update(v["edge"], query, neighbor, index, edge_weight, v["hidden_weight"], v["hidden_bias"], v["output_weight"], v["output_bias"],
                                v["norm_weight"], v["norm_bias"], seed, 1e-5, p, "cuda")
    rows = index.numel()
    keep = _keep_mask(seed, rows, p).reshape(*index.shape, W) if p > 0 else None
    return edge_tail_update_pytorch(v["edge"], query, neighbor, index, edge_weight, v["hidden_weight"], v["hidden_bias"], v["output_weight"], v["output_bias"],
                                    v["norm_weight"], v["norm_bias"], keep, 1e-5, p)


def _leaves(vals, dtype=None):
    return {n: (t.detach().clone() if dtype is None else t.detach().to(dtype)).requires_grad_(True) for n, t in vals.items()}


GRAD_NAMES = ("edge", "node", "packed", "packed_bias", "hidden_weight", "hidden_bias", "output_weight", "output_bias", "norm_weight", "norm_bias")


def _tail_errors(batch, length, k, p, fp32_params):
    vals, index = _tail_inputs(batch, length, k, fp32_params=fp32_params)
    seed = torch.randint(0, 2**31 - 1, (1,), device="cuda", dtype=torch.int64)
    ours_l = _leaves({n: vals[n] for n in GRAD_NAMES})
    torch_l = _leaves({n: vals[n] for n in GRAD_NAMES})
    exact_l = _leaves({n: vals[n] for n in GRAD_NAMES}, torch.float64)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if fp32_params else torch.autocast("cuda", enabled=False)
    with ctx:
        ours = _tail_apply(ours_l, index, seed, p, "cuda", autocast=fp32_params)
        ref = _tail_apply(torch_l, index, seed, p, "torch", autocast=fp32_params)
    exact = _tail_apply(exact_l, index, seed, p, "torch", autocast=False)
    up = torch.randn(ours.shape, device="cuda", dtype=torch.float32)
    ours.float().mul(up).sum().backward()
    ref.float().mul(up).sum().backward()
    exact.mul(up.double()).sum().backward()
    errors = {"output": (_mean_rel(ours, exact), _mean_rel(ref, exact))}
    for n in GRAD_NAMES:
        errors[n] = (_mean_rel(ours_l[n].grad, exact_l[n].grad), _mean_rel(torch_l[n].grad, exact_l[n].grad))
    return errors, ours, ref


@pytest.mark.parametrize("fp32_params", [False, True])
@pytest.mark.parametrize("shape", [(1, 37, 5), (2, 256, 48), (1, 130, 32)])
@pytest.mark.parametrize("p", [0.0, 0.1])
def test_tail_forward_and_gradients_are_as_accurate_as_the_bf16_chain(shape, p, fp32_params):
    """Against FP64: the CUDA tail's output and ten gradients (the dropout mask replayed from the kernel's own draw) stay within the bf16 chain's error
    (2x and an absolute 5e-3 floor, the band of the Triton path's test; the output and the edge-state gradient carry the bf16 rounding of the LayerNorm output the fp32 reference
    does not have)."""
    errors, _ours, _ref = _tail_errors(*shape, p, fp32_params)
    failures = {n: e for n, e in errors.items() if e[0] > max(2.0 * e[1], 5e-3)}
    assert not failures, f"{failures} (all: {errors})"


def test_tail_dropout_masks_are_bernoulli_and_independent():
    """The draw is a counter hash: the kept fraction matches 1 - p, neighbouring channels, rows and two seeds are uncorrelated."""
    rows = 1 << 16
    for p in (0.1, 0.25, 0.5):
        seed = torch.tensor([20260929], device="cuda", dtype=torch.int64)
        m = _keep_mask(seed, rows, p).float()
        assert abs(m.mean().item() - (1 - p)) < 2e-3
        x = m - m.mean()
        for shifted in (x.roll(1, dims=1), x.roll(2, dims=1), x.roll(8, dims=1), x.roll(1, dims=0), x.roll(48, dims=0)):
            assert abs((x * shifted).mean().item() / x.var().item()) < 5e-3
        other = _keep_mask(seed + 1, rows, p).float()
        assert abs(((other - other.mean()) * x).mean().item() / x.var().item()) < 5e-3


def test_tail_dropout_replays_the_same_mask_in_backward_and_is_deterministic():
    """The decisions are saved bit-packed and the backward reads them: the forward is bitwise reproducible for a seed, differs for another, and drops about the requested share."""
    vals, index = _tail_inputs(2, 256, 48, fp32_params=False)
    seed = torch.tensor([7], device="cuda", dtype=torch.int64)

    def run(sd):
        leaves = _leaves({n: vals[n] for n in GRAD_NAMES})
        out = _tail_apply(leaves, index, sd, 0.25, "cuda", autocast=False)
        out.float().square().sum().backward()
        return out.detach(), leaves["edge"].grad, leaves["hidden_weight"].grad

    a, b = run(seed), run(seed)
    assert torch.equal(a[0], b[0])
    torch.testing.assert_close(a[1].float(), b[1].float(), atol=0, rtol=0)         # the dX path has no reduction whose order varies
    c = run(seed + 1)
    assert not torch.equal(a[0], c[0])


def test_tail_gate_and_switch():
    """The CUDA path serves the A100 call; ``MINIWORLD_MPNN_EDGE_SM80=0`` hands it to the Triton kernel (both agree to bf16 accuracy); an explicit ``cuda`` refuses what it does not serve."""
    vals, index = _tail_inputs(1, 128, 48, fp32_params=False)
    seed = torch.zeros(1, device="cuda", dtype=torch.int64)
    node, packed, pb = vals["node"], vals["packed"], vals["packed_bias"]
    query, neighbor = F.linear(node, packed[:, :W], pb), F.linear(node, packed[:, 2 * W:])
    args = (vals["edge"], query, neighbor, index, packed[:, W:2 * W], vals["hidden_weight"], vals["hidden_bias"], vals["output_weight"], vals["output_bias"], vals["norm_weight"],
            vals["norm_bias"], seed, 1e-5, 0.0)
    assert sm80.tail_serves(vals["edge"], packed[:, W:2 * W], vals["hidden_weight"], vals["output_weight"])
    with torch.no_grad():
        ours = edge_tail_update(*args, "triton_compute")
        os.environ[sm80.ENV] = "0"
        try:
            assert not sm80.tail_serves(vals["edge"], packed[:, W:2 * W], vals["hidden_weight"], vals["output_weight"])
            triton = edge_tail_update(*args, "triton_compute")
            with pytest.raises(ValueError, match="sm_80 CUDA MPNN edge tail"):
                edge_tail_update(*args, "cuda")
        finally:
            os.environ[sm80.ENV] = "1"
    assert _rel(ours, triton) < 1.5e-2


def test_tail_fp32_activations_are_not_served():
    vals, index = _tail_inputs(1, 64, 8, fp32_params=False)
    node, packed, pb = vals["node"], vals["packed"], vals["packed_bias"]
    query, neighbor = F.linear(node, packed[:, :W], pb), F.linear(node, packed[:, 2 * W:])
    seed = torch.zeros(1, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="requires contiguous CUDA BF16"):
        edge_tail_update(vals["edge"].float(), query, neighbor, index, packed[:, W:2 * W], vals["hidden_weight"], vals["hidden_bias"], vals["output_weight"], vals["output_bias"],
                         vals["norm_weight"], vals["norm_bias"], seed, 1e-5, 0.0, "cuda")


@pytest.mark.parametrize("p", [0.0, 0.1])
def test_tail_compiled_matches_eager(p):
    """``torch.compile(fullgraph=True)`` of forward + backward is bit-identical to eager for the forward and agrees on every gradient."""
    torch._dynamo.reset()
    vals, index = _tail_inputs(1, 256, 48, fp32_params=False)
    seed = torch.tensor([11], device="cuda", dtype=torch.int64)

    def forward(*ts):
        return _tail_apply(dict(zip(GRAD_NAMES, ts, strict=True)), index, seed, p, "cuda", autocast=False)

    def run(fn):
        leaves = _leaves({n: vals[n] for n in GRAD_NAMES})
        out = fn(*[leaves[n] for n in GRAD_NAMES])
        out.float().square().sum().backward()
        return out.detach(), [leaves[n].grad for n in GRAD_NAMES]

    out_e, grads_e = run(forward)
    out_c, grads_c = run(torch.compile(forward, fullgraph=True))
    assert torch.equal(out_e, out_c)
    for n, ge, gc in zip(GRAD_NAMES, grads_e, grads_c, strict=True):
        assert _rel(gc, ge) < 2e-3, n


def test_tail_cuda_graph_replay():
    """A captured forward + backward replays: forward bit-identical, gradients equal to eager (the neighbour scatter sums in the order the CSR fill hands out)."""
    vals, index = _tail_inputs(1, 256, 48, fp32_params=False)
    seed = torch.tensor([5], device="cuda", dtype=torch.int64)
    leaves = _leaves({n: vals[n] for n in GRAD_NAMES})
    up = torch.randn(1, 256, 48, W, device="cuda", dtype=torch.bfloat16)

    def step():
        for t in leaves.values():
            t.grad = None
        out = _tail_apply(leaves, index, seed, 0.1, "cuda", autocast=False)
        out.backward(up)
        return out

    want = step().detach().clone()
    want_g = {n: t.grad.detach().clone() for n, t in leaves.items()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    for n, t in leaves.items():
        assert _rel(t.grad, want_g[n]) < 2e-3, n


def _tail_policy_run(vals, index, seed, p, backend):
    """forward + backward of the public tail under one policy name; (output, gradients by name, bytes the forward left allocated beyond its inputs, peak bytes of the whole step beyond them)."""
    leaves = _leaves({n: vals[n] for n in GRAD_NAMES})
    node, packed, pb = leaves["node"], leaves["packed"], leaves["packed_bias"]
    query, neighbor = F.linear(node, packed[:, :W], pb), F.linear(node, packed[:, 2 * W:])
    up = torch.randn(*index.shape, W, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(41))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = edge_tail_update(leaves["edge"], query, neighbor, index, packed[:, W:2 * W], leaves["hidden_weight"], leaves["hidden_bias"], leaves["output_weight"],
                           leaves["output_bias"], leaves["norm_weight"], leaves["norm_bias"], seed, 1e-5, p, backend)
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - before
    out.backward(up)
    torch.cuda.synchronize()
    return out.detach(), {n: t.grad for n, t in leaves.items()}, kept, torch.cuda.max_memory_allocated() - before


@pytest.mark.parametrize("p", [0.0, 0.25])
def test_tail_recompute_policy_replays_the_forward_and_keeps_less(p):
    """``triton`` (the recompute policy) runs the same kernels on an A100 and keeps only its inputs: the output and every gradient equal the saved-activation policy's
    (the replayed forward is the same kernel with the same seed; the neighbour scatter sums in the order the CSR fill hands out), and the forward leaves a fraction of the memory behind."""
    vals, index = _tail_inputs(2, 256, 48, fp32_params=False)
    seed = torch.tensor([13], device="cuda", dtype=torch.int64)
    out_s, grads_s, kept_s, _ = _tail_policy_run(vals, index, seed, p, "triton_compute")
    out_r, grads_r, kept_r, _ = _tail_policy_run(vals, index, seed, p, "triton")
    assert torch.equal(out_s, out_r)
    assert torch.equal(grads_s["edge"], grads_r["edge"])
    for n in GRAD_NAMES:
        assert _rel(grads_r[n], grads_s[n]) < 2e-3, n
    activation = out_s.numel() * out_s.element_size()
    assert kept_s > 4 * activation, (kept_s / activation, kept_r / activation)
    assert kept_r <= 1.5 * activation, (kept_s / activation, kept_r / activation)


@pytest.mark.parametrize("p", [0.0, 0.25])
@pytest.mark.parametrize("fp32_params", [False, True])
def test_tail_recompute_in_slices_matches_the_saved_activations(monkeypatch, p, fp32_params):
    """The recompute policy replays the forward in slices of whole nodes (here 16 nodes = 768 rows of 24,576): the slice at row r0 draws the dropout decisions of rows r0 ..., so the gradients equal the
    saved-activation policy's (fp32 sums of the slices, rounded once); the neighbour scatter and the parameter gradients agree to the order of the sums."""
    monkeypatch.setattr(sm80, "_RECOMPUTE_CHUNK_ROWS", 16 * 48)
    vals, index = _tail_inputs(2, 256, 48, fp32_params=fp32_params)
    seed = torch.tensor([17], device="cuda", dtype=torch.int64)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if fp32_params else torch.autocast("cuda", enabled=False)
    with ctx:
        out_s, grads_s, _, _ = _tail_policy_run(vals, index, seed, p, "triton_compute")
        out_r, grads_r, _, _ = _tail_policy_run(vals, index, seed, p, "triton")
    assert torch.equal(out_s, out_r)
    assert torch.equal(grads_s["edge"], grads_r["edge"])
    for n in GRAD_NAMES:
        assert _rel(grads_r[n], grads_s[n]) < 5e-3, n


def test_tail_recompute_in_slices_bounds_the_peak_memory(monkeypatch):
    """With slices of 4,096 rows the step's peak beyond its inputs is a fraction of the saved-activation policy's, which holds 1.3 KB per edge row from the forward on."""
    monkeypatch.setattr(sm80, "_RECOMPUTE_CHUNK_ROWS", 1 << 12)
    vals, index = _tail_inputs(1, 4096, 48, fp32_params=False)
    seed = torch.tensor([19], device="cuda", dtype=torch.int64)
    _, _, kept_s, peak_s = _tail_policy_run(vals, index, seed, 0.1, "triton_compute")
    _, _, kept_r, peak_r = _tail_policy_run(vals, index, seed, 0.1, "triton")
    assert kept_s > 4 * kept_r
    assert peak_r < 0.5 * peak_s, (peak_r / 2**20, peak_s / 2**20)


# =========================================================================================================================================== edge MLP
def _mlp_inputs(rows, fp32_params):
    torch.manual_seed(29)
    x = torch.randn(rows, W, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pdt = torch.float32 if fp32_params else torch.bfloat16
    ps = [(torch.randn(*s, device="cuda") / W**0.5).to(pdt).requires_grad_(True) for s in ((W, W), (W,), (W, W), (W,))]
    return x, ps, torch.randn_like(x)


@pytest.mark.parametrize("backend", ["cuda", "triton_compute", "triton_memory"])
@pytest.mark.parametrize("rows", [1, 17, 257, 2048 * 48])
@pytest.mark.parametrize("fp32_params", [False, True])
def test_mlp_matches_the_bf16_chain(rows, backend, fp32_params):
    x, ps, up = _mlp_inputs(rows, fp32_params)
    ref_in = [t.detach().clone().requires_grad_(True) for t in (x, *ps)]
    exact_in = [t.detach().double().requires_grad_(True) for t in (x, *ps)]
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if fp32_params else torch.autocast("cuda", enabled=False)
    with ctx:
        out = edge_mlp_update(x, *ps, backend=backend)
        ref = edge_mlp_update_pytorch(*ref_in)
    exact = edge_mlp_update_pytorch(*exact_in)
    grads = torch.autograd.grad(out, [x, *ps], up)
    ref_grads = torch.autograd.grad(ref, ref_in, up)
    exact_grads = torch.autograd.grad(exact, exact_in, up.double())
    # a single row (128 numbers) is a noisy sample of the error: the band is wider there
    factor, floor = (1.1, 5e-4) if rows > 17 else (1.4, 1.5e-3)
    assert _rel(out, exact) <= factor * _rel(ref, exact) + 1e-4
    for n, a, b, e in zip(["x", "Wh", "bh", "Wo", "bo"], grads, ref_grads, exact_grads, strict=True):
        assert _rel(a, e) <= factor * _rel(b, e) + floor, n


def test_mlp_auto_serves_the_a100_above_the_size_floor_and_keeps_pytorch_below():
    x, ps, _ = _mlp_inputs(2048 * 48, False)
    small = x[:1000].detach().contiguous()
    assert sm80.mlp_serves(x, ps[0], ps[2])
    with torch.no_grad():
        big = edge_mlp_update(x.detach(), *[p.detach() for p in ps], backend="auto")
        ref = edge_mlp_update_pytorch(x.detach(), *[p.detach() for p in ps])
        tiny = edge_mlp_update(small, *[p.detach() for p in ps], backend="auto")
    assert _rel(big, ref) < 1e-2
    assert torch.equal(tiny, edge_mlp_update_pytorch(small, *[p.detach() for p in ps]))


def test_mlp_follows_the_triton_paths_rounding_points():
    """The CUDA edge MLP rounds the same intermediates as the Triton one (the same call with the switch off; both keep the bf16 autograd chain's roundings): the output and the input and bias gradients
    agree far below bf16 noise.  The weight gradients differ by the noise of cuBLAS's bf16 split-K reduction, which the Triton path's GEMMs carry and the fixed-order fp32 sums of the CUDA kernel do not
    (2.5e-3 at 98 K rows against the fp64 value, where the CUDA kernel is the more accurate one: ``test_mlp_matches_the_bf16_chain``)."""
    x, ps, up = _mlp_inputs(2048 * 4, False)
    out = edge_mlp_update(x, *ps, backend="triton_compute")
    grads = torch.autograd.grad(out, [x, *ps], up)
    os.environ[sm80.ENV] = "0"
    try:
        leaves = [t.detach().clone().requires_grad_(True) for t in (x, *ps)]
        ref = edge_mlp_update(*leaves, backend="triton_compute")
        ref_grads = torch.autograd.grad(ref, leaves, up)
    finally:
        os.environ[sm80.ENV] = "1"
    assert _rel(out, ref) < 1e-4
    for n, a, b in zip(["x", "Wh", "bh", "Wo", "bo"], grads, ref_grads, strict=True):
        assert _rel(a, b) < (5e-3 if n in ("Wh", "Wo") else 1e-3), n


def test_mlp_rank_one_backward_preserves_bias_shapes():
    x, ps, up = _mlp_inputs(1, False)
    out = edge_mlp_update(x[0], *ps, backend="cuda")
    grads = torch.autograd.grad(out, [x, *ps], up[0], allow_unused=True)
    assert out.shape == (W,)
    assert grads[2].shape == (W,)
    assert grads[4].shape == (W,)


def test_mlp_compiled_matches_eager():
    torch._dynamo.reset()
    x, ps, up = _mlp_inputs(2048, False)

    def forward(*ts):
        return edge_mlp_update(ts[0], *ts[1:], backend="cuda")

    def run(fn):
        out = fn(x, *ps)
        return out, torch.autograd.grad(out, [x, *ps], up)

    out_e, grads_e = run(forward)
    out_c, grads_c = run(torch.compile(forward, fullgraph=True))
    assert torch.equal(out_e, out_c)
    for a, b in zip(grads_c, grads_e, strict=True):
        assert _rel(a, b) < 2e-3


# =========================================================================================================================================== edge LayerNorm
@pytest.mark.parametrize("rows", [1, 3, 100, 777, 2048 * 48])
@pytest.mark.parametrize("autocast", [False, True])
def test_layernorm_memory_backward_matches_torch(rows, autocast):
    torch.manual_seed(6)
    x = (torch.randn(rows, W, device="cuda") * 2 + 0.3).to(torch.bfloat16).requires_grad_(True)
    w = (torch.rand(W, device="cuda") + 0.5).requires_grad_(True)
    b = torch.randn(W, device="cuda").requires_grad_(True)
    up = torch.randn(rows, W, device="cuda")
    ref_in = [t.detach().clone().requires_grad_(True) for t in (x, w, b)]
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)
    with ctx:
        out = edge_layer_norm(x, w, b, 1e-5, backend="cuda")
        ref = edge_layer_norm_pytorch(*ref_in, 1e-5)
    grads = torch.autograd.grad(out, [x, w, b], up.to(out.dtype))
    ref_grads = torch.autograd.grad(ref, ref_in, up.to(ref.dtype))
    assert torch.equal(out, ref)
    for n, a, c in zip("xwb", grads, ref_grads, strict=True):
        assert _rel(a, c) < 2e-4, n


def test_layernorm_compiled_matches_eager():
    torch._dynamo.reset()
    x = torch.randn(4096, W, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.rand(W, device="cuda") + 0.5).requires_grad_(True)
    b = torch.randn(W, device="cuda").requires_grad_(True)
    up = torch.randn(4096, W, device="cuda", dtype=torch.bfloat16)

    def forward(a, g, c):
        return edge_layer_norm(a, g, c, 1e-5, backend="cuda")

    def run(fn):
        out = fn(x, w, b)
        return out, torch.autograd.grad(out, [x, w, b], up)

    out_e, grads_e = run(forward)
    out_c, grads_c = run(torch.compile(forward, fullgraph=True))
    # the forward is the native LayerNorm (Inductor fuses its own): within one bf16 ulp; the backward is the CUDA kernel in both
    torch.testing.assert_close(out_c.float(), out_e.float(), atol=2e-2, rtol=2e-2)
    for a, c in zip(grads_c, grads_e, strict=True):
        assert _rel(a, c) < 1e-2


# =========================================================================================================================================== edge dropout
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [13, 1000, 4099, 2048 * 48 * W])
def test_dropout_bitpack_is_bitwise_native(n, dtype):
    """Native forward (the same Philox draw as F.dropout) and a backward off the packed bits equal to native_dropout_backward, to the bit."""
    x = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
    torch.manual_seed(11)
    out = edge_dropout(x, 0.25, training=True, backend="cuda")
    torch.manual_seed(11)
    ref_in = x.detach().clone().requires_grad_(True)
    ref = F.dropout(ref_in, p=0.25, training=True)
    up = torch.randn(n, device="cuda", dtype=dtype)
    assert torch.equal(out, ref)
    assert torch.equal(torch.autograd.grad(out, x, up)[0], torch.autograd.grad(ref, ref_in, up)[0])


def test_dropout_compiled_matches_compiled_native():
    """Compiled, the packed-mask dropout still consumes the CUDA RNG exactly as the (compiled) native dropout does and returns its values and gradient."""
    torch._dynamo.reset()
    x = torch.randn(1 << 20, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    up = torch.randn_like(x)
    native = torch.compile(lambda t: F.dropout(t, p=0.25, training=True), fullgraph=True)
    candidate = torch.compile(lambda t: edge_dropout(t, 0.25, training=True, backend="cuda"), fullgraph=True)
    torch.cuda.manual_seed_all(109)
    expected = native(x)
    (expected_grad,) = torch.autograd.grad(expected, x, up)
    torch.cuda.manual_seed_all(109)
    actual = candidate(x2)
    (actual_grad,) = torch.autograd.grad(actual, x2, up)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_grad, expected_grad)


def _capture(step):
    """Warm ``step`` up on a side stream, capture one call and return the graph and what the captured call returned."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = step()
    return graph, result


def test_mlp_and_layernorm_cuda_graph_replay():
    """A captured edge MLP (saved projection) + compressed-save LayerNorm, forward and backward, replays: output and every gradient equal to the eager call (nothing in these paths depends on an atomic order)."""
    torch.manual_seed(31)
    rows = 4096
    names = ("x", "wh", "bh", "wo", "bo", "gamma", "beta")
    shapes = ((rows, W), (W, W), (W,), (W, W), (W,), (W,), (W,))
    leaves = {n: (torch.randn(*s, device="cuda") * (0.5 if n == "x" else 0.08)).to(torch.float32 if n in ("gamma", "beta") else torch.bfloat16).requires_grad_(True)
              for n, s in zip(names, shapes, strict=True)}
    up = torch.randn(rows, W, device="cuda", dtype=torch.bfloat16)

    def step():
        for t in leaves.values():
            t.grad = None
        hidden = edge_mlp_update(leaves["x"], leaves["wh"], leaves["bh"], leaves["wo"], leaves["bo"], backend="cuda")
        out = edge_layer_norm(hidden, leaves["gamma"], leaves["beta"], 1e-5, backend="cuda")
        out.backward(up)
        return out

    want = step().detach().clone()
    want_grads = {n: t.grad.detach().clone() for n, t in leaves.items()}
    graph, out = _capture(step)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    for n, t in leaves.items():
        assert _rel(t.grad, want_grads[n]) < 1e-6, n


def test_dropout_cuda_graph_replay():
    """A captured bit-packed dropout (forward and backward) replays with a fresh draw whose backward is consistent with its own forward: the gradient is ``grad / (1 - p)`` exactly where the output is kept."""
    p = 0.25
    x = (torch.randn(1 << 18, device="cuda", dtype=torch.bfloat16).abs() + 1.0).requires_grad_(True)       # >= 1: a kept element is non-zero in the output
    up = torch.randn_like(x)

    def step():
        x.grad = None
        out = edge_dropout(x, p, training=True, backend="cuda")
        out.backward(up)
        return out

    graph, out = _capture(step)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        kept = out != 0
        assert 0.7 < kept.float().mean().item() < 0.8
        expected = torch.where(kept, (up.float() * (1.0 / (1.0 - p))).to(torch.bfloat16), torch.zeros_like(up))
        assert torch.equal(x.grad, expected)


# =========================================================================================================================================== the model
def test_encoder_with_the_fused_tail_matches_the_separate_operations():
    """A ProteinMPNN with the CUDA tail policy against the same model on the separate-operation path (dropout 0): outputs and parameter gradients agree to bf16 noise."""
    from miniworld_engine.modules.mpnn import ProteinMPNN, ProteinMPNNConfig

    def build(backend):
        torch.manual_seed(5)
        model = ProteinMPNN(ProteinMPNNConfig(encoder_depth=2, decoder_depth=2, k_neighbors=48, coordinate_noise=0.0, dropout=0.0, message_backend="pytorch",
                                              edge_mlp_backend="pytorch", feature_backend="pytorch", edge_tail_backend=backend)).cuda().train()
        with torch.no_grad():
            model.output_projection.weight.normal_(0.0, 0.05)
        return model.bfloat16()

    length = 256
    torch.manual_seed(7)
    inputs = (torch.randn(1, length, 4, 3, device="cuda") * 20.0, torch.randint(0, 21, (1, length), device="cuda"), torch.ones(1, length, device="cuda"),
              torch.arange(length, device="cuda").unsqueeze(0), torch.zeros(1, length, dtype=torch.long, device="cuda"), torch.randperm(length, device="cuda").unsqueeze(0),
              (torch.arange(length, device="cuda") // 8).unsqueeze(0).contiguous())
    outputs, grads = {}, {}
    for backend in ("off", "triton_compute"):
        model = build(backend)
        logits = model(*inputs)
        logits.float().square().mean().backward()
        outputs[backend] = logits.float()
        grads[backend] = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    assert _mean_rel(outputs["triton_compute"], outputs["off"]) < 5e-2
    comparable = [n for n in grads["off"] if grads["off"][n].abs().mean() > 0]
    assert len(comparable) > 10
    worst = max(_mean_rel(grads["triton_compute"][n], grads["off"][n]) for n in comparable)
    assert worst < 0.25
