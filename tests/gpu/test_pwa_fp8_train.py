"""EXPERIMENTAL opt-in PWA FP8 TRAINING route (``kernels.integrations.pwa_fp8_train``, H100): FP8 key contraction in the forward and FP8
dv / dw GEMMs in the backward (cuBLASLt), the operands written by ``ln_vt8n`` / ``pwa_glue3_f8``. Off by default: the default route does
not change; on: the fused producers give the PyTorch-quantized operands' bytes, every gradient is finite and within a loose bound of the
bf16 default's error against fp64, CUDA-graph replay = eager, the step is deterministic."""

import copy

import pytest
import torch

from miniworld_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]

_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.fixture
def f8t():
    from miniworld_engine.kernels.integrations import pwa_fp8_train as F

    saved = (F.PLACES, F.IMPL, F.DO_FMT)
    F.reset()
    yield F
    F.PLACES, F.IMPL, F.DO_FMT = saved
    F.reset()


def _module(d_pair: int = 128, seed: int = 11):
    m = MSAPairWeightedAveraging(64, d_pair, 8, 32, master_weights=True, backend="cuda").cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    return m


def _inputs(s: int, n: int, d_pair: int = 128):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, s, n, 64, device="cuda", generator=gen).bfloat16()
    z = torch.randn(1, n, n, d_pair, device="cuda", generator=gen).bfloat16()
    gy = torch.randn(1, s, n, 64, device="cuda", generator=gen).bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[:, n - 3 :] = False
    return x, z, gy, mask


def _grads(m, x, z, gy, mask):
    x = x.detach().clone().requires_grad_()
    z = z.detach().clone().requires_grad_()
    ps = list(m.parameters())
    y = m(x, z, mask)
    return [y.detach(), *torch.autograd.grad(y, [x, z, *ps], gy.to(y.dtype))]


def _rel(a, b):
    d = a.double() - b.double()
    return float(d.pow(2).mean().sqrt() / b.double().pow(2).mean().sqrt())


@pytest.mark.skipif(not _H100, reason="H100")
def test_fused_producers_match_pytorch_quantization(f8t) -> None:
    from miniworld_engine.kernels.integrations import pwa_train as P

    torch.manual_seed(0)
    s, n = 64, 256
    m = torch.randn(s, n, 64, device="cuda").bfloat16()
    lw, lb = 1 + 0.1 * torch.randn(64, device="cuda"), 0.1 * torch.randn(64, device="cuda")
    wv = (torch.randn(256, 64, device="cuda") / 8).bfloat16()
    v, y = P._k(True)["lnvg"].ln_vg(m, lw, lb, wv, 1e-5, 2, 3, 1)
    amax = torch.zeros(1, device="cuda")
    inv = torch.full((1,), 8.0, device="cuda")
    vt8, v8, y8 = f8t._vt8_nat().ln_vt8n(m, lw, lb, wv, 1e-5, inv, amax, 2, 3)
    assert torch.equal(y, y8)
    assert torch.equal(vt8.view(torch.uint8), f8t._t8(v8).view(torch.uint8))  # the two layouts hold the same bytes
    ref = v.float()
    assert _rel(v8.float() / 8.0, ref) < 0.04
    assert abs(float(amax) - 8 * float(ref.abs().max())) <= 0.02 * float(amax)
    w = torch.softmax(torch.randn(8, n, n, device="cuda"), -1).bfloat16()
    from miniworld_engine.kernels.integrations import pwa_fp8 as I

    assert torch.equal(f8t._glue8().w_quant_t(w).view(torch.uint8), f8t._t8(I._ext().w_quant(w)).view(torch.uint8))


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize("fmt", ["e5m2", "e4m3"])
def test_fused_equals_proto(f8t, fmt: str) -> None:
    """The fused route and the PyTorch-quantized one use the same scales and roundings: outputs and gradients agree closely."""
    m = _module()
    x, z, gy, mask = _inputs(64, 384)
    f8t.PLACES, f8t.DO_FMT = f8t.ALL, fmt
    f8t.IMPL = "proto"
    f8t.reset()
    a = _grads(m, x, z, gy, mask)
    f8t.IMPL = "fused"
    f8t.reset()
    b = _grads(m, x, z, gy, mask)
    for ta, tb in zip(a, b, strict=True):
        assert torch.isfinite(tb).all()
        assert _rel(tb, ta) < 2e-2


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(("d_pair", "n", "s"), [(128, 384, 64), (128, 768, 32), (256, 512, 32)])
def test_route_gradients_and_default_unchanged(f8t, d_pair: int, n: int, s: int) -> None:
    m = _module(d_pair)
    x, z, gy, mask = _inputs(s, n, d_pair)
    ref = copy.deepcopy(m).double()
    ref.set_backend("pytorch")
    g64 = _grads(ref, x.double(), z.double(), gy.double(), mask)
    f8t.PLACES = frozenset()
    d0 = _grads(m, x, z, gy, mask)
    f8t.PLACES = f8t.ALL
    f8 = _grads(m, x, z, gy, mask)
    f8 = _grads(m, x, z, gy, mask)  # the delayed scale (history of the first call)
    for i, (gd, gf, gr) in enumerate(zip(d0, f8, g64, strict=True)):
        assert torch.isfinite(gf).all(), i
        assert _rel(gf, gr) < max(0.08, 6 * _rel(gd, gr)), i
    f8t.PLACES = frozenset()
    d1 = _grads(m, x, z, gy, mask)
    for a, b in zip(d0, d1, strict=True):
        assert torch.equal(a, b)
    assert f8t.overflow_count() == 0


@pytest.mark.skipif(not _H100, reason="H100")
def test_graph_replay_equals_eager_and_deterministic(f8t) -> None:
    m = _module()
    x, z, gy, mask = _inputs(64, 384)
    x.requires_grad_()
    ps = list(m.parameters())
    f8t.PLACES = f8t.ALL

    def step():
        y = m(x, z, mask)
        return torch.autograd.grad(y, [x, *ps], gy)

    step()  # seeds the scales
    e1 = step()
    e2 = step()
    for a, b in zip(e1, e2, strict=True):
        assert torch.equal(a, b)  # same inputs, same (steady) scales: bit-identical
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    g.replay()
    torch.cuda.synchronize()
    for a, b in zip(e1, out, strict=True):
        assert torch.equal(a, b)
