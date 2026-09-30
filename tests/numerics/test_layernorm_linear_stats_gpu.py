import os

import pytest
import torch
import triton

from miniworld_engine import settings
from miniworld_engine.kernels.layernorm_linear.autograd import (
    layernorm_linear_triton_fn,
)
from miniworld_engine.kernels.layernorm_linear.triton.fused import (
    _lnl_fwd_kernel,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def norm_settings():
    previous = settings.current()
    settings.configure(engine_backend="triton", autotune_miss_cap=24)
    yield
    settings.configure(**vars(previous))


def reference(x, g, b, w, bias):
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    xn = torch.nn.functional.layer_norm(
        x.to(acc), (x.shape[-1],), g.to(acc), b.to(acc), 1e-5
    )
    return torch.nn.functional.linear(xn.to(x.dtype), w, bias)


def check(a, b, dt):
    tol = {
        torch.bfloat16: 0.005,
        torch.float16: 0.001,
        torch.float32: 3e-5,
        torch.float64: 1e-10,
    }[dt]
    assert torch.isfinite(a).all()
    err = float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-10))
    assert err < tol, (a.shape, err, tol)


def inputs(k, n, dt, layout="2d", bias=True):
    torch.manual_seed(932)
    x = torch.randn(2, 17, k * 2 if layout == "strided" else k, device="cuda", dtype=dt)
    if layout == "2d":
        x = x.reshape(-1, k)
    if layout == "4d":
        x = x.reshape(1, 2, 17, k)
    if layout == "strided":
        x = x[..., ::2]
    x = x.detach().requires_grad_()
    acc = torch.float64 if dt == torch.float64 else torch.float32
    g = torch.randn(k, device="cuda", dtype=acc, requires_grad=True)
    b = torch.randn(k, device="cuda", dtype=acc, requires_grad=True)
    w = (torch.randn(n, k, device="cuda", dtype=dt) / k**0.5).requires_grad_()
    z = torch.randn(n, device="cuda", dtype=dt, requires_grad=True) if bias else None
    return x, g, b, w, z


_dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
if "NORM_CHECK_PART" in os.environ:
    _dtypes = [_dtypes[int(os.environ["NORM_CHECK_PART"])]]


@pytest.mark.parametrize("dt", _dtypes)
@pytest.mark.parametrize(
    ("k", "n", "layout"),
    [
        (64, 64, "2d"),
        (128, 16, "3d"),
        (137, 17, "strided"),
        (384, 512, "4d"),
        (1024, 33, "2d"),
        (1025, 17, "3d"),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_output_all_gradients(dt, k, n, layout, bias):
    args = inputs(k, n, dt, layout, bias)
    params = tuple(a for a in args if a is not None)
    y = layernorm_linear_triton_fn(*args)
    want = reference(*args)
    dy = torch.randn_like(y)
    gs = torch.autograd.grad(y, params, dy)
    ws = torch.autograd.grad(want, params, dy)
    for a, b in zip((y, *gs), (want, *ws), strict=False):
        check(a, b, dt)


@pytest.mark.parametrize("bk", [64, 256])
def test_saved_stats_tail_and_offset(bk):
    m, k, n = 35, 137, 17
    x = (torch.randn(m, k, device="cuda") + 1000).contiguous()
    g = torch.randn(k, device="cuda")
    b = torch.randn_like(g)
    w = torch.randn(n, k, device="cuda")
    y = torch.empty(m, n, device="cuda")
    mean = torch.empty(m, device="cuda")
    inv = torch.empty_like(mean)
    _lnl_fwd_kernel.fn[(triton.cdiv(m, 8),)](
        x,
        w,
        x,
        g,
        b,
        y,
        m,
        n,
        k,
        1e-5,
        k,
        1,
        k,
        1,
        n,
        1,
        False,
        BLOCK_M1=8,
        BLOCK_N=32,
        BLOCK_K=bk,
        shape_key=0,
        Mean=mean,
        Rstd=inv,
        SAVE_STATS=True,
        num_warps=4,
        num_stages=1,
    )
    # FP64 oracle: FP32 Welford itself loses precision at a large offset.
    torch.testing.assert_close(mean, x.double().mean(-1).float(), rtol=1e-6, atol=2e-4)
    torch.testing.assert_close(
        inv,
        torch.rsqrt(x.double().var(-1, unbiased=False) + 1e-5).float(),
        rtol=3e-6,
        atol=3e-6,
    )


def test_graph_changed_inputs_and_compile():
    args = inputs(128, 16, torch.bfloat16)
    params = tuple(args)
    dy = torch.randn(34, 16, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(layernorm_linear_triton_fn, fullgraph=True)

    def call():
        y = compiled(*args)
        return y, *torch.autograd.grad(y, params, dy)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        out = call()
    with torch.no_grad():
        for v in args:
            v.copy_(torch.randn_like(v))
        dy.copy_(torch.randn_like(dy))
    graph.replay()
    torch.cuda.synchronize()
    ref = reference(*args)
    gr = torch.autograd.grad(ref, params, dy)
    for a, b in zip(out, (ref, *gr), strict=False):
        check(a, b, torch.bfloat16)


def test_empty():
    x, g, b, w, z = inputs(64, 16, torch.bfloat16)
    x = x[:0].detach().requires_grad_()
    args = (x, g, b, w, z)
    y = layernorm_linear_triton_fn(*args)
    assert y.shape == (0, 16)
    for grad in torch.autograd.grad(y, args, torch.empty_like(y)):
        assert torch.count_nonzero(grad) == 0
