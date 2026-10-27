"""Pipelined LayerNorm / RMSNorm (kernels/layernorm/cuda/pipe) on H100: forward, input / weight / bias gradients against an fp64 reference at
widths that fill the lanes exactly and widths that pad them, all dtypes, weight dtype mixes, absent affine, row counts that leave a ragged last
pass, repeated backward on one graph, CUDA graph capture, torch.compile, and the routing of ``layernorm_kernel`` / ``rmsnorm_cuda`` to it."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.layernorm.cuda import pipe
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.rmsnorm.cuda import rmsnorm_cuda

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="needs an H100"),
]

# bytes per row of bf16 / fp32 = 2n / 4n: 144 (9 chunks, padded half warp), 256, 512, 1000-ish pads, 1536, 2048
WIDTHS = [72, 80, 128, 152, 256, 384, 768, 1024]
TOL = {torch.bfloat16: 1e-2, torch.float16: 2e-3, torch.float32: 2e-5}
EPS = 1e-5


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("rms", [False, True], ids=["ln", "rms"])
def test_forward_and_backward_match_fp64(n, dtype, rms):
    if not pipe.supported(torch.empty(1, n, device="cuda", dtype=dtype)):
        pytest.skip("width outside the pipelined builds")
    for m, affine in ((1, True), (37, False), (1003, True), (6151, True)):
        g = torch.Generator(device="cuda").manual_seed(n * 7 + m)
        x = (torch.randn(m, n, device="cuda", generator=g) * 3 + 0.5).to(dtype)
        w = (1 + 0.2 * torch.randn(n, device="cuda", generator=g)).to(dtype) if affine else None
        b = (0.1 * torch.randn(n, device="cuda", generator=g)).to(dtype) if affine and not rms else None
        dy = torch.randn(m, n, device="cuda", generator=g).to(dtype)
        xs = x.double().requires_grad_()
        ws = w.double().requires_grad_() if w is not None else None
        bs = b.double().requires_grad_() if b is not None else None
        y_ref = F.rms_norm(xs, (n,), ws, EPS) if rms else F.layer_norm(xs, (n,), ws, bs, EPS)
        y_ref.backward(dy.double())
        xi = x.clone().requires_grad_()
        wi = w.clone().requires_grad_() if w is not None else None
        bi = b.clone().requires_grad_() if b is not None else None
        y = pipe.rmsnorm_pipe(xi, wi, EPS) if rms else pipe.layernorm_pipe(xi, wi, bi, EPS)
        y.backward(dy)
        tol = TOL[dtype]
        assert y.dtype == dtype
        assert rel(y, y_ref) < tol, (n, m, "y")
        assert rel(xi.grad, xs.grad) < 4 * tol, (n, m, "dx")
        if wi is not None:
            assert wi.grad.dtype == dtype
            assert rel(wi.grad, ws.grad) < 8 * tol, (n, m, "dw")
        if bi is not None:
            assert bi.grad.dtype == dtype
            assert rel(bi.grad, bs.grad) < 8 * tol, (n, m, "db")


def test_fp32_weights_under_bf16_activations_and_partial_requires_grad():
    n = 256
    x = torch.randn(500, n, device="cuda", dtype=torch.bfloat16)
    w = (torch.rand(n, device="cuda") + 0.5).requires_grad_()  # fp32 affine, bf16 x
    b = torch.randn(n, device="cuda").requires_grad_()
    dy = torch.randn_like(x)
    y = pipe.layernorm_pipe(x, w, b, EPS)
    y.backward(dy)
    assert w.grad.dtype == torch.float32 and b.grad.dtype == torch.float32 and x.grad is None
    xs, ws, bs = x.double(), w.detach().double().requires_grad_(), b.detach().double().requires_grad_()
    F.layer_norm(xs, (n,), ws, bs, EPS).backward(dy.double())
    assert rel(w.grad, ws.grad) < 5e-2 and rel(b.grad, bs.grad) < 5e-2
    # only x requires grad: no weight gradient work, none returned
    xi = x.clone().requires_grad_()
    pipe.layernorm_pipe(xi, w.detach(), b.detach(), EPS).backward(dy)
    assert xi.grad is not None


def test_backward_twice_and_two_forwards_before_backward():
    n = 128
    x = torch.randn(4000, n, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.rand(n, device="cuda", dtype=torch.bfloat16).requires_grad_()
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn_like(x)
    y1 = pipe.layernorm_pipe(x, w, b, EPS)
    y2 = pipe.layernorm_pipe(x, w, b, EPS)
    y2.backward(dy, retain_graph=True)
    g1 = w.grad.clone()
    y2.backward(dy)  # the workspace cleared itself: a second pass sums the same gradient again
    assert rel(w.grad, 2 * g1) < 2e-2
    w.grad = None
    y1.backward(dy)
    assert rel(w.grad, g1) < 2e-2


def test_cuda_graph_and_compile():
    n = 128
    x = torch.randn(4096, n, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.rand(n, device="cuda", dtype=torch.bfloat16).requires_grad_()
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn_like(x)

    def step(f):
        x.grad = w.grad = b.grad = None
        f(x, w, b).backward(dy)
        return x.grad.clone(), w.grad.clone(), b.grad.clone()

    eager = lambda x, w, b: pipe.layernorm_pipe(x, w, b, EPS)
    ref = step(eager)
    compiled = torch.compile(eager)
    for a, e in zip(step(compiled), ref, strict=True):
        assert rel(a, e) < 1e-2
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            x.grad = w.grad = b.grad = None
            eager(x, w, b).backward(dy)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    graph = torch.cuda.CUDAGraph()
    x.grad = w.grad = b.grad = None
    with torch.cuda.graph(graph):
        eager(x, w, b).backward(dy)
    graph.replay()
    torch.cuda.synchronize()
    assert rel(w.grad, ref[1]) < 1e-2


def test_the_engine_entries_route_here_on_h100():
    from miniworld_engine import settings
    from miniworld_engine.kernels.layernorm.interface import _use_pipe

    x = torch.randn(2, 64, 64, 128, device="cuda", dtype=torch.bfloat16)
    assert _use_pipe(x)
    assert not _use_pipe(x[..., :40].contiguous())  # 80-byte rows: the general kernels
    settings.configure(force_portable_kernels=True)
    try:
        assert not _use_pipe(x)
    finally:
        settings.configure(force_portable_kernels=False)
    w = torch.rand(128, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    assert rel(layernorm_kernel(x, w, b, EPS), F.layer_norm(x.double(), (128,), w.double(), b.double(), EPS)) < 1e-2
    assert rel(rmsnorm_cuda(x, w, EPS), F.rms_norm(x.double(), (128,), w.double(), EPS)) < 1e-2


def test_replays_are_bit_identical():
    for n, rms in ((128, False), (768, True)):
        x0 = torch.randn(30011, n, device="cuda", dtype=torch.bfloat16)
        w0 = torch.rand(n, device="cuda", dtype=torch.bfloat16) + 0.5
        b0 = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x0)
        outs = []
        for _ in range(3):
            x, w, b = x0.clone().requires_grad_(), w0.clone().requires_grad_(), b0.clone().requires_grad_()
            y = pipe.rmsnorm_pipe(x, w, EPS) if rms else pipe.layernorm_pipe(x, w, b, EPS)
            y.backward(dy)
            outs.append((y, x.grad, w.grad) + (() if rms else (b.grad,)))
        for o in outs[1:]:
            for a, e in zip(o, outs[0], strict=True):
                assert torch.equal(a, e)
