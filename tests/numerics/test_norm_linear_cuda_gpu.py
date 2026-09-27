"""Native LN/WMMA fusion: full gradients, tails, views, graph, and fallback."""

import pytest
import torch
from miniworld_engine.kernels.norm_cuda.linear import fused_layernorm_linear, extension
from miniworld_engine.kernels.norm_cuda import extension as norm_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def preload_extensions():
    # Load all CUDA modules before autograd worker threads and graph capture.
    # This also avoids late module registration under Compute Sanitizer.
    torch.cuda.init()
    norm_extension()
    from miniworld_engine.kernels.norm_cuda.wide import extension as wide_extension

    wide_extension()
    extension()


def reference(x, g, b, w, bias):
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    xn = torch.nn.functional.layer_norm(
        x.to(acc), (x.shape[-1],), g.to(acc), b.to(acc), 1e-5
    ).to(x.dtype)
    return torch.nn.functional.linear(xn, w, bias)


def compare(x, g, b, w, bias, fn=fused_layernorm_linear):
    args = x, g, b, w, bias
    params = tuple(v for v in args if v is not None)
    y, yr = fn(*args), reference(*args)
    dy = torch.randn_like(y)
    grads = torch.autograd.grad(y, params, dy)
    expected = torch.autograd.grad(yr, params, dy)
    tolerance = {
        torch.bfloat16: 0.012,
        torch.float16: 0.0025,
        torch.float32: 2e-5,
        torch.float64: 1e-10,
    }[x.dtype]
    for value, target in zip((y, *grads), (yr, *expected)):
        assert torch.isfinite(value).all()
        relative = (
            value.double() - target.double()
        ).norm() / target.double().norm().clamp_min(1e-12)
        assert float(relative) < tolerance


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "d,n",
    [(64, 16), (128, 64), (256, 128), (384, 32), (512, 64), (33, 17), (1024, 256)],
)
def test_linear(dtype, d, n):
    torch.manual_seed(64)
    x = torch.randn(1, 37, d, device="cuda", dtype=dtype, requires_grad=True)
    acc = torch.float64 if dtype == torch.float64 else torch.float32
    g = torch.randn(d, device="cuda", dtype=acc, requires_grad=True)
    b = torch.randn_like(g, requires_grad=True)
    w = (torch.randn(n, d, device="cuda", dtype=dtype) / d**0.5).requires_grad_()
    bias = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
    compare(x, g, b, w, bias)


@pytest.mark.parametrize("layout", ["strided", "offset", "empty"])
def test_views(layout):
    dtype = torch.bfloat16
    d = 128
    n = 32
    m = 0 if layout == "empty" else 33
    base = torch.randn(max(1, m * d * 2 + 1), device="cuda", dtype=dtype)
    x = (
        base[1 : m * d + 1].view(m, d)
        if layout != "strided"
        else base[: m * d * 2 : 2].view(m, d)
    ).requires_grad_()
    w = (
        torch.randn(n * d + 1, device="cuda", dtype=dtype)[1:]
        .view(n, d)
        .requires_grad_()
    )
    g = torch.randn(d, device="cuda", requires_grad=True)
    b = torch.randn_like(g, requires_grad=True)
    compare(x, g, b, w, None)


def test_graph_and_compile():
    extension()
    norm_extension()
    torch.manual_seed(985)
    x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(128, device="cuda", requires_grad=True)
    b = torch.randn_like(g, requires_grad=True)
    w = torch.randn(32, 128, device="cuda", dtype=x.dtype, requires_grad=True)
    args = x, g, b, w
    compiled = torch.compile(fused_layernorm_linear, fullgraph=True)
    compare(*args, None, fn=compiled)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        out = fused_layernorm_linear(*args)
        return out, torch.autograd.grad(out, args, torch.ones_like(out))

    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    for _ in range(2):
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
            w.copy_(torch.randn_like(w))
        graph.replay()
        expected = run()
        for val, ref in zip((captured[0], *captured[1]), (expected[0], *expected[1])):
            torch.testing.assert_close(val, ref)


def test_inference_without_saved_activation():
    from miniworld_engine.kernels.norm_cuda.linear import _forward

    extension()
    x = torch.randn(35, 128, device="cuda", dtype=torch.float16)
    g = torch.randn(128, device="cuda")
    b = torch.randn_like(g)
    w = torch.randn(16, 128, device="cuda", dtype=x.dtype)
    with torch.no_grad():
        y, xn, _, _ = _forward(x, g, b, w, None, 1e-5, False)
        assert xn.numel() == 0
        torch.testing.assert_close(
            y, reference(x, g, b, w, None), atol=0.025, rtol=0.003
        )
