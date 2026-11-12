"""The H100 training dropout scale (``kernels/triangle_attention/cuda/h100_triattn.drop_scale``: one Philox launch, ``drop_scale_kernel``)
is the torch draw ``modules.functional.dropout_scale`` bit for bit: the same default-generator subsequences and counter advance, BF16
uniform rounding, threshold BF16(p) and scale BF16(1 / BF16(1 - p)). Checked eagerly, inside a CUDA graph (the offset advances per replay
as torch's own draw does), for a fixed seed (determinism), its mean (~1), and through the modules' cuda training paths (TriMul /
bidirectional TriMul / TriangleAttention outputs equal to the same call drawing the scale with torch)."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import BidirectionalTriangleMultiplication, TriangleAttention, TriangleMultiplication
from miniworld_engine.modules.functional import dropout_scale

pytestmark = [pytest.mark.gpu]

H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
needs_h100 = pytest.mark.skipif(not H100, reason="H100 (sm_90) kernel")


def _kernel():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_triattn

    return h100_triattn


def _pair(b, n, c, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(b, n, n, c, device="cuda", generator=g).to(torch.bfloat16)


@needs_h100
@pytest.mark.parametrize("p", [0.25, 0.15, 0.1, 0.4])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(("b", "n", "c"), [(1, 128, 64), (2, 96, 128), (3, 40, 384)])
def test_bitwise_the_torch_draw(p, dim, b, n, c):
    k = _kernel()
    pair = _pair(b, n, c)
    assert k.drop_scale_serves(pair, p)
    torch.manual_seed(11)
    ref = dropout_scale(pair, p, dim)
    after_ref = torch.rand(7, device="cuda")
    torch.manual_seed(11)
    got = k.drop_scale(pair, p, dim)
    after_got = torch.rand(7, device="cuda")
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got, ref)
    assert torch.equal(after_got, after_ref), "the generator's offset must advance as torch.rand's does"


@needs_h100
def test_fixed_seed_is_deterministic_and_mean_one():
    k = _kernel()
    pair = _pair(1, 512, 128)
    torch.manual_seed(3)
    a = k.drop_scale(pair, 0.25, 1)
    torch.manual_seed(3)
    b = k.drop_scale(pair, 0.25, 1)
    c = k.drop_scale(pair, 0.25, 1)
    assert torch.equal(a, b)
    assert not torch.equal(b, c)
    assert set(a.unique().float().tolist()) <= {0.0, float(torch.tensor(1 / 0.75, dtype=torch.bfloat16))}
    assert abs(float(a.float().mean()) - 1.0) < 0.01
    assert abs(float((a == 0).float().mean()) - 0.25) < 0.01


@needs_h100
def test_graph_replays_draw_as_torch_does():
    """A captured draw advances the generator per replay: the replays equal a captured torch draw's replays from the same seed."""
    k = _kernel()
    pair = _pair(1, 128, 64)

    def replays(fn):
        torch.manual_seed(5)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()  # warm (kernel load) outside the capture
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.manual_seed(5)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        seen = []
        for _ in range(3):
            g.replay()
            seen.append(out.clone())
        torch.cuda.synchronize()
        return seen

    ours = replays(lambda: k.drop_scale(pair, 0.25, 1))
    ref = replays(lambda: dropout_scale(pair, 0.25, 1))
    assert not torch.equal(ours[0], ours[1]) and not torch.equal(ours[1], ours[2])
    for a, b in zip(ours, ref, strict=True):
        assert torch.equal(a, b)


def _module_case(cls, kwargs, n):
    m = cls(**kwargs).cuda().float().train()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    m = m.set_backend("cuda")
    pair = _pair(1, n, kwargs["d_pair"], seed=1)
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[:, -n // 8 :] = False
    return m, pair, mask


@needs_h100
@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (TriangleMultiplication, {"d_pair": 64, "outgoing": True}),
        (TriangleMultiplication, {"d_pair": 128, "outgoing": False}),
        (BidirectionalTriangleMultiplication, {"d_pair": 64}),
        (TriangleAttention, {"d_pair": 128, "n_head": 4, "starting": True}),
        (TriangleAttention, {"d_pair": 128, "n_head": 4, "starting": False}),
    ],
)
def test_module_training_outputs_unchanged(cls, kwargs, monkeypatch):
    """The cuda training path with the kernel's draw == the same path drawing with torch (the same seed): outputs and gradients."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_triattn

    m, pair, mask = _module_case(cls, kwargs, 128)
    with torch.no_grad():
        why = m.unsupported(pair, mask)
    if why is not None:
        pytest.skip(f"not served: {why}")
    calls = []
    real = h100_triattn.drop_scale

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    def run(model):
        x = pair.clone().requires_grad_(True)
        torch.manual_seed(9)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x, mask)
        y.float().square().mean().backward()
        return y.detach(), x.grad

    monkeypatch.setattr(h100_triattn, "drop_scale", counted)
    y1, g1 = run(m)
    assert calls, "the cuda training path did not draw with the kernel"
    monkeypatch.setattr(h100_triattn, "drop_scale_serves", lambda *a: False)  # the torch statements (modules.functional.dropout_scale)
    y2, g2 = run(copy.deepcopy(m))
    assert torch.equal(y1, y2)
    assert g1 is not None and g2 is not None
    assert torch.equal(g1, g2)
