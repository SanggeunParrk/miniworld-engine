"""General CUDA LayerNorm (kernels/layernorm/cuda/general): any width / row count / dtype, forward and backward against an fp64
reference, the affine variants, bit-identical replays, CUDA-graph capture and torch.compile."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.layernorm.cuda.general import config, layernorm_cuda

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

WIDTHS = [1, 7, 16, 33, 64, 100, 128, 192, 256, 384, 512, 768, 1000, 1024, 2048, 3000, 4096, 8192, 16384]
TOL = {torch.bfloat16: 1e-2, torch.float16: 2e-3, torch.float32: 2e-5}


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _case(m, n, dtype, affine, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, n, device="cuda", generator=g) * 3 + 1).to(dtype)
    w = (1 + 0.2 * torch.randn(n, device="cuda", generator=g)) if affine in ("wb", "w") else None
    b = (0.1 * torch.randn(n, device="cuda", generator=g)) if affine == "wb" else None
    dy = torch.randn(m, n, device="cuda", generator=g).to(dtype)
    return x, w, b, dy


def _reference(x, w, b, dy, eps=1e-5):
    xs = x.double().requires_grad_()
    ws = w.double().requires_grad_() if w is not None else None
    bs = b.double().requires_grad_() if b is not None else None
    y = F.layer_norm(xs, (x.shape[-1],), ws, bs, eps)
    y.backward(dy.double())
    return y, xs.grad, None if ws is None else ws.grad, None if bs is None else bs.grad


@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
def test_forward_and_backward_match_fp64(n, dtype):
    for m, affine in ((1, "wb"), (37, "w"), (1003, "none"), (513, "wb")):
        x, w, b, dy = _case(m, n, dtype, affine, seed=n + m)
        y_ref, dx_ref, dw_ref, db_ref = _reference(x, w, b, dy)
        xi = x.clone().requires_grad_()
        wi = w.clone().requires_grad_() if w is not None else None
        bi = b.clone().requires_grad_() if b is not None else None
        y = layernorm_cuda(xi, wi, bi, 1e-5)
        y.backward(dy)
        assert y.dtype == dtype
        assert y.shape == x.shape
        tol = TOL[dtype]
        if n > 1:
            assert rel(y, y_ref) < tol, (n, m, affine, "y")
            assert rel(xi.grad, dx_ref) < 4 * tol, (n, m, affine, "dx")
        if wi is not None:
            assert wi.grad.dtype == wi.dtype
            assert rel(wi.grad, dw_ref) < 4 * tol, (n, m, affine, "dw")
        if bi is not None:
            assert rel(bi.grad, db_ref) < 4 * tol, (n, m, affine, "db")


def test_leading_dims_bf16_activation_fp32_affine_and_no_grad():
    x = torch.randn(4, 3, 50, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(64, device="cuda") + 0.5
    b = torch.randn(64, device="cuda")
    with torch.no_grad():
        y = layernorm_cuda(x, w, b)
    ref = F.layer_norm(x.float(), (64,), w, b).to(torch.bfloat16)
    assert y.shape == x.shape
    assert rel(y, ref) < 1e-2


@pytest.mark.parametrize("n", [64, 128, 1000])
def test_replays_are_bit_identical(n):
    x, w, b, dy = _case(20000, n, torch.bfloat16, "wb", seed=3)
    outs = []
    for _ in range(3):
        xi, wi, bi = x.clone().requires_grad_(), w.clone().requires_grad_(), b.clone().requires_grad_()
        y = layernorm_cuda(xi, wi, bi)
        y.backward(dy)
        outs.append((y, xi.grad, wi.grad, bi.grad))
    for o in outs[1:]:
        for a, e in zip(o, outs[0], strict=True):
            assert torch.equal(a, e)


def test_cuda_graph_and_compile():
    x, w, b, dy = _case(4096, 128, torch.bfloat16, "wb", seed=5)
    w.requires_grad_(); b.requires_grad_()
    xi = x.clone().requires_grad_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            layernorm_cuda(xi, w, b).backward(dy)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = layernorm_cuda(xi, w, b)
            y.backward(dy)
    torch.cuda.current_stream().wait_stream(s)
    graph.replay()
    torch.cuda.synchronize()
    assert rel(y, F.layer_norm(x.float(), (128,), w.detach(), b.detach())) < 1e-2
    compiled = torch.compile(lambda t: layernorm_cuda(t, w, b) * 2, fullgraph=True)
    assert rel(compiled(x), 2 * F.layer_norm(x.float(), (128,), w.detach(), b.detach())) < 1e-2


def test_config_covers_every_width():
    for elem in (2, 4):
        for n in range(1, 4100):
            tpr, ch, vec = config(n, elem)
            assert tpr * ch * vec >= n


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10), reason="H100 / B200 routing")
def test_engine_layernorms_take_the_general_kernel_on_h100_and_b200(monkeypatch):
    """modules.LayerNorm (MINIWORLD) and architecture.primitives.LayerNorm (MINIWORLD_ENGINE) run the general kernel on the cards with no
    Triton fallback, with the module's fp32 affine, and match their PyTorch reference in value and gradient."""
    from miniworld_engine.architecture.exceptions import ImplementationType as ArchImpl
    from miniworld_engine.architecture.primitives import LayerNorm as ArchLayerNorm
    from miniworld_engine.kernels.layernorm.cuda import general
    from miniworld_engine.modules import LayerNorm
    from miniworld_engine.modules.exceptions import ImplementationType

    calls = []
    real = general.layernorm_cuda
    monkeypatch.setattr(general, "layernorm_cuda", lambda *a, **k: calls.append(1) or real(*a, **k))
    for make, impl, ref_impl in ((LayerNorm, ImplementationType.MINIWORLD, ImplementationType.PYTORCH),
                                 (ArchLayerNorm, ArchImpl.MINIWORLD_ENGINE, ArchImpl.PYTORCH)):
        mod = make(128, implementation=impl).cuda()
        ref = make(128, implementation=ref_impl).cuda()
        with torch.no_grad():
            mod.weight.normal_(1, 0.2); mod.bias.normal_(0, 0.1)
        ref.load_state_dict(mod.state_dict())
        x = torch.randn(2, 96, 96, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        outs = []
        for m in (mod, ref):
            xi = x.clone().requires_grad_()
            y = m(xi)
            y.backward(dy)
            outs.append((y, xi.grad, m.weight.grad, m.bias.grad))
        for a, e in zip(*outs, strict=True):
            assert a.dtype == e.dtype
            assert rel(a, e) < 1e-2
    assert len(calls) == 2
