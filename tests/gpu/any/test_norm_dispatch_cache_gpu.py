import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.layernorm import dispatch
from miniworld_engine.kernels.layernorm.compile_native import _resolve_bwd_path
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def restore_settings():
    previous = settings.current()
    yield
    settings.configure(**vars(previous))


def test_dtype_scoped_lookup(monkeypatch):
    entries = {
        "384|131072": {"path": "persistent"},
        "384|131072|torch.bfloat16|torch.float32": {"path": "atomic"},
    }
    monkeypatch.setattr(dispatch, "_load", lambda idx: entries)
    dev = torch.device("cuda", 0)
    assert dispatch.lookup(dev, 384, 131072) == "persistent"
    assert (
        dispatch.lookup(dev, 384, 131072, regime="torch.bfloat16|torch.float32")
        == "atomic"
    )
    assert (
        dispatch.lookup(dev, 384, 131072, regime="torch.float32|torch.float32") is None
    )


def test_triton_override_and_off(monkeypatch):
    x = torch.empty(1, 384, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(384, device="cuda")
    s = torch.empty(1, device="cuda")
    monkeypatch.setattr(dispatch, "lookup", lambda *args, **kw: "atomic")
    try:
        settings.configure(
            engine_backend="triton", layernorm_dispatch="auto", layernorm_bwd_path=None
        )
        assert _resolve_bwd_path(147456, 384, x, x, w, s, s) == "atomic"
        settings.configure(layernorm_bwd_path="persistent")
        assert _resolve_bwd_path(147456, 384, x, x, w, s, s) == "persistent"
        settings.configure(layernorm_bwd_path=None, layernorm_dispatch="off")
        assert _resolve_bwd_path(147456, 384, x, x, w, s, s) == "persistent"
    finally:
        settings.reset()


def test_rms_cache_graph():
    settings.configure(engine_backend="triton", autotune_miss_cap=24)
    x = torch.randn(
        1, 8192, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w = torch.randn(1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)

    def call():
        y = triton_rmsnorm(x, w, 1e-5)
        return y, *torch.autograd.grad(y, (x, w), dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = call()
    with torch.no_grad():
        x.copy_(torch.randn_like(x))
        w.copy_(torch.randn_like(w))
        dy.copy_(torch.randn_like(dy))
    graph.replay()
    torch.cuda.synchronize()
    xf = x.float()
    xh = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
    expected = (xh * w).to(x.dtype)
    grads = torch.autograd.grad(expected, (x, w), dy)
    for actual, ref in zip(out, (expected, *grads), strict=False):
        rel = (actual.double() - ref.double()).norm() / ref.double().norm().clamp_min(
            1e-10
        )
        assert rel < 0.004, rel.item()


def test_portable_linear_stats(monkeypatch):
    from miniworld_engine.kernels.layernorm_linear import layernorm_linear

    x = torch.randn(1, 35, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(128, device="cuda")
    b = torch.randn_like(g)
    w = torch.randn(16, 128, device="cuda", dtype=x.dtype)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 0))
    y, mean, inv = layernorm_linear(x, g, b, w, None, save_stats=True)
    xf = x.float()
    ref = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(xf, (128,), g, b, 1e-5).to(x.dtype), w
    )
    assert y.shape == (1, 35, 16)
    assert mean.dtype == inv.dtype == torch.float32
    assert mean.shape == inv.shape == (35,)
    assert (y.float() - ref.float()).norm() / ref.float().norm() < 0.004
    torch.testing.assert_close(mean, xf.mean(-1).flatten(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        inv,
        torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5).flatten(),
        rtol=1e-5,
        atol=1e-6,
    )
