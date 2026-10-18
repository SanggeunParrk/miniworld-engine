"""General CUDA RMSNorm (kernels/rmsnorm/cuda): any width / row count / dtype, forward and backward against an fp64 reference, with and
without a weight, bit-identical replays, CUDA-graph capture, torch.compile, and the engine RMSNorm routing to it on H100 / B200."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.rmsnorm.cuda import rmsnorm_cuda

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

WIDTHS = [1, 7, 16, 24, 32, 33, 48, 64, 100, 128, 256, 384, 768, 1000, 2048, 4096, 16384]
TOL = {torch.bfloat16: 1e-2, torch.float16: 2e-3, torch.float32: 2e-5}
EPS = 1.1920928955078125e-07


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
def test_forward_and_backward_match_fp64(n, dtype):
    for m, weighted in ((1, True), (37, False), (1003, True), (4099, False)):
        g = torch.Generator(device="cuda").manual_seed(n * 7 + m)
        x = (torch.randn(m, n, device="cuda", generator=g) * 3 + 0.5).to(dtype)
        w = (1 + 0.2 * torch.randn(n, device="cuda", generator=g)) if weighted else None
        dy = torch.randn(m, n, device="cuda", generator=g).to(dtype)
        xs = x.double().requires_grad_()
        ws = w.double().requires_grad_() if w is not None else None
        y_ref = F.rms_norm(xs, (n,), ws, EPS)
        y_ref.backward(dy.double())
        xi = x.clone().requires_grad_()
        wi = w.clone().requires_grad_() if w is not None else None
        y = rmsnorm_cuda(xi, wi, EPS)
        y.backward(dy)
        tol = TOL[dtype]
        assert y.dtype == dtype
        assert rel(y, y_ref) < tol, (n, m, "y")
        if n > 1:
            assert rel(xi.grad, xs.grad) < 4 * tol, (n, m, "dx")
        else:  # width 1: y = sign(x), dx = eps-sized cancellation except where x^2 ~ eps; compare absolutely
            err = float((xi.grad.double() - xs.grad).abs().max())
            assert err <= 4 * tol * float(xs.grad.abs().max()) + 1e-3, (n, m, "dx")
        if wi is not None:
            assert wi.grad.dtype == wi.dtype
            assert rel(wi.grad, ws.grad) < 4 * tol, (n, m, "dw")


def test_bf16_weight_gets_a_bf16_gradient_and_no_grad_path():
    x = torch.randn(4, 50, 64, device="cuda", dtype=torch.bfloat16)
    w = (torch.rand(64, device="cuda") + 0.5).to(torch.bfloat16).requires_grad_()
    rmsnorm_cuda(x, w, EPS).sum().backward()
    assert w.grad.dtype == torch.bfloat16
    with torch.no_grad():
        y = rmsnorm_cuda(x, w, EPS)
    assert rel(y, F.rms_norm(x.float(), (64,), w.float(), EPS)) < 1e-2


def test_replays_are_bit_identical():
    x = torch.randn(30000, 48, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(48, device="cuda") + 0.5
    dy = torch.randn_like(x)
    outs = []
    for _ in range(3):
        xi, wi = x.clone().requires_grad_(), w.clone().requires_grad_()
        y = rmsnorm_cuda(xi, wi, EPS)
        y.backward(dy)
        outs.append((y, xi.grad, wi.grad))
    for o in outs[1:]:
        for a, e in zip(o, outs[0], strict=True):
            assert torch.equal(a, e)


def test_cuda_graph_and_compile():
    x = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16)
    w = (torch.rand(128, device="cuda") + 0.5).requires_grad_()
    dy = torch.randn_like(x)
    xi = x.clone().requires_grad_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            rmsnorm_cuda(xi, w, EPS).backward(dy)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = rmsnorm_cuda(xi, w, EPS)
            y.backward(dy)
    torch.cuda.current_stream().wait_stream(s)
    graph.replay()
    torch.cuda.synchronize()
    ref = F.rms_norm(x.float(), (128,), w.detach(), EPS)
    assert rel(y, ref) < 1e-2
    compiled = torch.compile(lambda t: rmsnorm_cuda(t, w, EPS) * 2, fullgraph=True)
    assert rel(compiled(x), 2 * ref) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10), reason="H100 / B200 routing")
def test_engine_rmsnorm_takes_the_general_kernel_on_h100_and_b200(monkeypatch):
    """modules.RMSNorm (MINIWORLD) runs the general kernel on the cards with no Triton fallback, returns the input dtype, and matches
    the PyTorch RMSNorm in value and gradient (weighted and unweighted)."""
    from miniworld_engine.kernels.rmsnorm import cuda as general
    from miniworld_engine.modules import RMSNorm
    from miniworld_engine.modules.exceptions import ImplementationType

    calls = []
    real = general.rmsnorm_cuda
    monkeypatch.setattr(general, "rmsnorm_cuda", lambda *a, **k: calls.append(1) or real(*a, **k))
    for affine in (True, False):
        mod = RMSNorm(48, elementwise_affine=affine, implementation=ImplementationType.MINIWORLD).cuda()
        if affine:
            with torch.no_grad():
                mod.weight.normal_(1, 0.2)
        x = torch.randn(2, 300, 8, 48, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        xi = x.clone().requires_grad_()
        y = mod(xi)
        y.backward(dy)
        assert y.dtype == torch.bfloat16
        xs = x.double().requires_grad_()
        ws = mod.weight.detach().double().requires_grad_() if affine else None
        ref = F.rms_norm(xs, (48,), ws, mod.effective_eps(x.dtype))
        ref.backward(dy.double())
        assert rel(y, ref) < 1e-2
        assert rel(xi.grad, xs.grad) < 4e-2
        if affine:
            assert rel(mod.weight.grad, ws.grad) < 4e-2
    assert len(calls) == 2
