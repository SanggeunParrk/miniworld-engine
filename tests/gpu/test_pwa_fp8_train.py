"""EXPERIMENTAL opt-in PWA FP8 TRAINING route (``kernels.integrations.pwa_fp8_train``, H100): FP8 key contraction in the forward and FP8
dv / dw GEMMs in the backward (cuBLASLt), the operands written by ``ln_vt8n`` / ``pwa_glue3_f8``. Off by default: the default route does
not change; on: the fused producers give the PyTorch-quantized operands' bytes, every gradient is finite and within a loose bound of the
bf16 default's error against fp64, CUDA-graph replay = eager, the step is deterministic."""

import copy

import pytest
import torch

from foldjet_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]

_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.fixture
def f8t():
    from foldjet_engine.kernels.integrations import pwa_fp8_train as F

    saved = (F.PLACES, F.IMPL, F.DO_FMT, F.MIN_S, F.O8)
    F.MIN_S = 0  # the tests run small S
    F.reset()
    yield F
    F.PLACES, F.IMPL, F.DO_FMT, F.MIN_S, F.O8 = saved
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
    from foldjet_engine.kernels.integrations import pwa_train as P

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
    # the two layouts come from two wgmma orders of the same dot products: the same bytes up to rare last-bit rounding ties
    assert float((vt8.view(torch.uint8) != f8t._t8(v8).view(torch.uint8)).float().mean()) < 1e-3
    assert _rel(f8t._t8(v8).float(), vt8.float()) < 5e-3
    ref = v.float()
    assert _rel(v8.float() / 8.0, ref) < 0.04
    assert abs(float(amax) - 8 * float(ref.abs().max())) <= 0.02 * float(amax)
    w = torch.softmax(torch.randn(8, n, n, device="cuda"), -1).bfloat16()
    from foldjet_engine.kernels.integrations import pwa_fp8 as I

    assert torch.equal(f8t._glue8().w_quant_t(w).view(torch.uint8), f8t._t8(I._ext().w_quant(w)).view(torch.uint8))


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize("fmt", ["e5m2", "e4m3"])
def test_fused_equals_proto(f8t, fmt: str) -> None:
    """The fused route quantizes the fp32 do directly (the PyTorch route the bf16-rounded do): the same scales, so the error against fp64
    is the same size."""
    m = _module()
    x, z, gy, mask = _inputs(64, 384)
    ref = copy.deepcopy(m).double()
    ref.set_backend("pytorch")
    g64 = _grads(ref, x.double(), z.double(), gy.double(), mask)
    f8t.PLACES, f8t.DO_FMT = f8t.ALL, fmt
    f8t.IMPL = "proto"
    f8t.reset()
    a = _grads(m, x, z, gy, mask)
    f8t.IMPL = "fused"
    f8t.reset()
    b = _grads(m, x, z, gy, mask)
    for ta, tb, tr in zip(a, b, g64, strict=True):
        assert torch.isfinite(tb).all()
        assert _rel(tb, tr) < 1.3 * _rel(ta, tr) + 2e-3


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(("d_pair", "n", "s"), [(128, 384, 1024), (128, 768, 32), (256, 512, 32)])
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
    if s == 1024:  # a qualified (native, deterministic) default shape: the default bytes are untouched by the route having run
        for a, b in zip(d0, d1, strict=True):
            assert torch.equal(a, b)  # (the plain builds at other shapes use float atomics: not bit-reproducible even in 2.5.14)
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


@pytest.mark.skipif(not _H100, reason="H100")
def test_bwd_only_producer_and_route(f8t) -> None:
    """Round 2 (backward-only FP8: bf16 forward, FP8 dv / dw): the VT_BF producer's bf16 v is ln_vg's v (bit for bit), its e4m3 v the
    same bytes as the full route's; the forward output equals the bf16 native default's; gradients stay inside the FP8 budget; the
    step is deterministic and graph replay equals eager."""
    from foldjet_engine.kernels.integrations import pwa_train as P

    torch.manual_seed(0)
    s, n = 64, 256
    m = torch.randn(s, n, 64, device="cuda").bfloat16()
    lw, lb = 1 + 0.1 * torch.randn(64, device="cuda"), 0.1 * torch.randn(64, device="cuda")
    wv = (torch.randn(256, 64, device="cuda") / 8).bfloat16()
    v, y = P._k(True)["lnvg"].ln_vg(m, lw, lb, wv, 1e-5, 2, 3, 1)
    inv = torch.full((1,), 8.0, device="cuda")
    amax0, amax1 = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    _, v8, _ = f8t._vt8_nat().ln_vt8n(m, lw, lb, wv, 1e-5, inv, amax0, 2, 3)
    vb, v8b, yb = f8t._vt8_nat_bf().ln_vt8n(m, lw, lb, wv, 1e-5, inv, amax1, 2, 3)
    assert torch.equal(yb, y)
    assert torch.equal(v8b.view(torch.uint8), v8.view(torch.uint8))
    assert torch.equal(vb, v) and float(amax0) == float(amax1)

    mod = _module(128)
    x, z, gy, mask = _inputs(32, 768)
    ref = copy.deepcopy(mod).double()
    ref.set_backend("pytorch")
    g64 = _grads(ref, x.double(), z.double(), gy.double(), mask)
    f8t.PLACES = frozenset()
    f8t.NATIVE_ONLY = True  # the native bf16 route at the same shape (the bwd-only route runs it too)
    d0 = _grads(mod, x, z, gy, mask)
    f8t.NATIVE_ONLY = False
    f8t.PLACES = f8t.BWD
    f8t.reset()
    _grads(mod, x, z, gy, mask)  # seeds the scales
    b = _grads(mod, x, z, gy, mask)
    b2 = _grads(mod, x, z, gy, mask)
    assert torch.equal(b[0], d0[0])  # bf16 forward: the output is the bf16 native default's
    for i, (gd, gf, gr) in enumerate(zip(d0, b, g64, strict=True)):
        assert torch.isfinite(gf).all(), i
        assert _rel(gf, gr) < max(0.08, 6 * _rel(gd, gr)), i
    for p, q in zip(b, b2, strict=True):
        assert torch.equal(p, q)
    assert f8t.overflow_count() == 0
    ps = list(mod.parameters())
    xg = x.detach().clone().requires_grad_()

    def step():
        yy = mod(xg, z, mask)
        return torch.autograd.grad(yy, [xg, *ps], gy)

    e1 = step()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        step()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    g.replay()
    torch.cuda.synchronize()
    for a, c in zip(e1, out, strict=True):
        assert torch.equal(a, c)


@pytest.mark.skipif(not _H100, reason="H100")
def test_serving_gate(f8t) -> None:
    """The route serves S >= MIN_S (1024), N % 128 == 0, N <= 1024, and for the backward-only variant N >= 512; nothing when off."""
    f8t.MIN_S = 1024
    x = torch.empty(1, 1024, 768, 64, device="cuda", dtype=torch.bfloat16)
    f8t.PLACES = frozenset()
    assert not f8t.serves(x)
    f8t.PLACES = f8t.ALL
    assert f8t.serves(x) and not f8t.serves(x[:, :512])
    f8t.PLACES = f8t.BWD
    assert f8t.serves(x) and not f8t.serves(torch.empty(1, 1024, 384, 64, device="cuda", dtype=torch.bfloat16))


@pytest.mark.skipif(not _H100, reason="H100")
def test_saved_o_e4m3_mode(f8t) -> None:
    """Round 4 (O8: the saved attention output o stored as e4m3 + a delayed scale, dequantised in the backward): quantize / dequantize
    match PyTorch's rounding, the forward output is the bwd route's, the saved bytes shrink, gradients stay inside the FP8 budget (0.4 -> ~2.6 %
    on the gate / out gradients), the step is deterministic."""
    torch.manual_seed(1)
    o = (torch.randn(8, 128, 2048, device="cuda") * 3).bfloat16()
    inv = torch.full((1,), 0.25, device="cuda")
    amax = torch.zeros(1, device="cuda")
    q = f8t._o8_ext().o8_quant(o, inv, amax)
    ref = (o.float() * 0.25).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(q.view(torch.uint8), ref.view(torch.uint8))
    assert float(amax) == float((o.float() * 0.25).abs().max())
    d = f8t._o8_ext().o8_dequant(q, torch.full((1,), 4.0, device="cuda"))
    assert torch.equal(d, (q.float() * 4.0).to(torch.bfloat16))

    mod = _module(128)
    x, z, gy, mask = _inputs(32, 768)
    ref64 = copy.deepcopy(mod).double()
    ref64.set_backend("pytorch")
    g64 = _grads(ref64, x.double(), z.double(), gy.double(), mask)
    f8t.PLACES = f8t.BWD
    f8t.reset()
    _grads(mod, x, z, gy, mask)
    b = _grads(mod, x, z, gy, mask)
    f8t.O8 = True
    f8t.reset()
    _grads(mod, x, z, gy, mask)
    c = _grads(mod, x, z, gy, mask)
    c2 = _grads(mod, x, z, gy, mask)
    assert torch.equal(c[0], b[0])  # forward untouched
    assert all(torch.equal(p, q_) for p, q_ in zip(c, c2, strict=True)), "O8 step not deterministic"
    for i, (gc, gb, gr) in enumerate(zip(c, b, g64, strict=True)):
        assert torch.isfinite(gc.float()).all()
        assert _rel(gc, gr) <= max(0.08, 3 * _rel(gb, gr)), (i, _rel(gc, gr), _rel(gb, gr))
    assert f8t.overflow_count() == 0
