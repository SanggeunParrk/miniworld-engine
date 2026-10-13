"""Qualify the explicit Hopper front/core/tail candidate; never exercise a fallback."""

import copy

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_triattn as native
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def relative(a, b):
    return float(
        (a.detach().float() - b.detach().float()).norm()
        / b.detach().float().norm().clamp_min(1e-12)
    )


def make(width, starting=True, dropout=0.0, master=False):
    m = (
        TriangleAttention(
            width,
            width // 32,
            starting=starting,
            p_drop=dropout,
            implementation=ImplementationType.PYTORCH,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    m.ln_pair.float()
    return m.float() if master else m


def reference(m, x, mask, drop=None):
    """PyTorch module math in bounded outer-row chunks, with activation-typed weights."""
    B, L, _, C = x.shape
    H = C // 32
    xx = x if m.starting else x.transpose(1, 2).contiguous()
    z = torch.nn.functional.layer_norm(
        xx.float(),
        (C,),
        m.ln_pair.weight.to(torch.float32),
        m.ln_pair.bias.to(torch.float32),
        m.ln_pair.eps,
    ).to(x.dtype)
    q, k, v, g = (
        torch.nn.functional.linear(z, w.weight.to(x.dtype))
        for w in (m.to_query, m.to_key, m.to_value, m.to_gate)
    )
    bias = torch.nn.functional.linear(z, m.to_bias.weight.to(x.dtype)).permute(
        0, 3, 1, 2
    )
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(x.dtype).min)
    q, k, v = (t.view(B, L, L, H, 32).permute(0, 3, 1, 2, 4) for t in (q, k, v))
    oo = []
    for b in range(B):
        chunks = []
        for r in range(0, L, 16):
            qq, kk, vv = (t[b, :, r : r + 16] for t in (q, k, v))
            score = (qq * (32**-0.5)) @ kk.transpose(-2, -1) + bias[b, :, None]
            chunks.append(score.softmax(-1) @ vv)
        oo.append(torch.cat(chunks, 1))
    o = torch.stack(oo).permute(0, 2, 3, 1, 4).reshape(B, L, L, C)
    y = torch.nn.functional.linear(g.sigmoid() * o, m.to_out.weight.to(x.dtype))
    if not m.starting:
        y = y.transpose(1, 2).contiguous()
    if drop is not None:
        y = y * drop.to(x.dtype)
    return x + y


@pytest.fixture(autouse=True)
def h100():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    torch.manual_seed(321)


def qualify(
    width,
    length,
    *,
    batch=1,
    starting=True,
    dropout=0.0,
    master=False,
    masking="mixed",
    training=True,
):
    ours = make(width, starting, dropout, master)
    ref16, ref32 = copy.deepcopy(ours), copy.deepcopy(ours).float()
    x = torch.randn(batch, length, length, width, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x) * 0.1
    mask = None if masking == "none" else torch.rand(batch, length, device="cuda") > 0.2
    if masking == "all":
        assert mask is not None
        mask.fill_(False)
    elif masking == "one":
        assert mask is not None
        mask.fill_(False)
        mask[:, 0] = True
    drop = None
    if dropout:
        torch.manual_seed(123)
        drop = ours._make_drop_scale(x, dropout)
    outcomes, grads = [], []
    for m, xi in [
        (ours, x),
        (ref16, x),
        (ref32, x.float()),
    ]:
        xi = xi.clone().requires_grad_(training)
        torch.manual_seed(123)
        with torch.set_grad_enabled(training):
            y = (
                native.forward(m, xi, mask)
                if m is ours
                else reference(m, xi, mask, drop)
            )
            if training:
                y.backward(dy.to(y.dtype))
        outcomes.append(y.detach())
        if training:
            grads.append([xi.grad, *(p.grad for p in m.parameters())])
    assert torch.isfinite(outcomes[0]).all()
    update = outcomes[2] - x.float()
    assert update.norm() > 0
    err = relative(outcomes[0].float() - x.float(), update)
    err16 = relative(outcomes[1].float() - x.float(), update)
    assert err <= 1.25 * err16 + 1e-3, (err, err16)
    if training:
        for n, a, b, c in zip(
            ["x", *(n for n, _ in ours.named_parameters())], *grads, strict=True
        ):
            assert not master or n == "x" or a.dtype == torch.float32
            err, err16 = relative(a, c), relative(b, c)
            # Bias projection gradient is exactly zero when all keys are masked.
            assert err <= 1.25 * err16 + 2e-3, (n, err, err16)
    return err


@pytest.mark.parametrize("width", [64, 128, 256, 384, 512])
@pytest.mark.parametrize("length", [128, 256, 384, 512, 640, 768])
def test_full_module_backward(width, length):
    qualify(width, length)


@pytest.mark.parametrize("width", [64, 128, 256, 384, 512])
@pytest.mark.parametrize("length", [128, 256, 384, 512, 640, 768])
def test_native_dropout_matches_torch_philox(width, length):
    m = make(width, dropout=0.25)
    x = torch.empty(1, length, length, width, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(987)
    want = m._make_drop_scale(x, 0.25).reshape(1, length, width)
    want_state = torch.cuda.get_rng_state()
    torch.manual_seed(987)
    got = native.extension("pack").drop_scale(x, 0.25)
    assert torch.equal(got, want)
    assert torch.equal(torch.cuda.get_rng_state(), want_state)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        native.extension("pack").drop_scale(x, 0.25)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = native.extension("pack").drop_scale(x, 0.25)
    torch.cuda.current_stream().wait_stream(stream)
    torch.manual_seed(987)
    graph.replay()
    torch.testing.assert_close(replayed, want, atol=0, rtol=0)
    first = replayed.clone()
    torch.manual_seed(987)
    graph.replay()
    assert torch.equal(replayed, first)
    torch.manual_seed(988)
    graph.replay()
    assert not torch.equal(replayed, first)


@pytest.mark.parametrize("length", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize("width", [64, 128, 256, 384, 512])
def test_bucket_forward(width, length):
    qualify(width, length, training=False)


@pytest.mark.parametrize(
    "options",
    [
        {"batch": 2},
        {"starting": False},
        {"dropout": 0.25},
        {"starting": False, "dropout": 0.25},
        {"master": True},
        {"masking": "none"},
        {"masking": "all"},
        {"masking": "one"},
    ],
)
def test_edges(options):
    qualify(128, 128, **options)


def test_graph_changes_inputs_masks_and_master_weights():
    m = make(256, master=True)
    x = torch.randn(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, 128, device="cuda") > 0.2
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            native.forward(m, x, mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        y = native.forward(m, x, mask)
    for _ in range(2):
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
            mask.copy_(torch.rand_like(mask, dtype=torch.float32) > 0.3)
            m.to_value.weight.add_(0.015)
            graph.replay()
            want = native.forward(m, x, mask)
        torch.testing.assert_close(y, want, atol=0, rtol=0)


def test_cuda_dispatch_has_no_triton():
    m = make(256).eval()
    x = torch.randn(
        1, 128, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dy = torch.randn_like(x)
    for _ in range(3):
        native.forward(m, x).backward(dy)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        native.forward(m, x).backward(dy)
        torch.cuda.synchronize()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    for kernel in [
        "training_fwd_stream",
        "dq_tma",
        "grouped_dkdv",
    ]:
        assert any(kernel in n for n in names), (kernel, names)
    assert any("ln_rows" in n for n in names), names
    assert not any("triton" in n.lower() or n.startswith("_attn_fwd") for n in names), (
        names
    )


def test_opt_in_module_dispatch():
    from miniworld_engine.integrations import triattn_h100
    from miniworld_engine.modules.dispatch import KernelBackend

    m = make(256).eval()
    m._backend = KernelBackend.TRITON
    x = torch.randn(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    assert not triattn_h100.serves(m, x)
    m._h100_b200style_cuda = True
    assert triattn_h100.serves(m, x)
    with torch.no_grad():
        torch.testing.assert_close(m(x), native.forward(m, x), atol=0, rtol=0)


def test_training_graph_changes_inputs_dy_masks_and_weights():
    m = make(256, master=True)
    x = torch.randn(
        1, 128, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dy = torch.randn_like(x) * 0.1
    mask = torch.rand(1, 128, device="cuda") > 0.2

    def step():
        m.zero_grad(set_to_none=False)
        if x.grad is not None:
            x.grad.zero_()
        y = native.forward(m, x, mask)
        y.backward(dy)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = step()
    for _ in range(2):
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
            dy.copy_(torch.randn_like(dy) * 0.1)
            mask.copy_(torch.rand_like(mask, dtype=torch.float32) > 0.3)
            m.to_query.weight.add_(0.01)
            graph.replay()
            assert x.grad is not None
            actual = [
                y.clone(),
                x.grad.clone(),
                *(p.grad.clone() for p in m.parameters()),
            ]
        expected = step()
        for a, b in zip(
            actual, [expected, x.grad, *(p.grad for p in m.parameters())], strict=True
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_dropout_graph_redraws():
    m = make(128, dropout=0.25)
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            native.forward(m, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        y = native.forward(m, x)
    graph.replay()
    a = y.clone()
    graph.replay()
    b = y.clone()
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    assert not torch.equal(a, b)
    assert (a == x).any()
    assert (b == x).any()


def test_zero_update_preserves_residual_and_gradient():
    m = make(256)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim == 2:
                p.zero_()
    x = torch.randn(
        1, 128, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dy = torch.randn_like(x)
    y = native.forward(m, x)
    y.backward(dy)
    torch.testing.assert_close(y, x, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, dy, atol=0, rtol=0)
    for p in m.parameters():
        assert torch.count_nonzero(p.grad) == 0
