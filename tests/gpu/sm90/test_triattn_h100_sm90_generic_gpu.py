"""H100 TriangleAttention beyond d_pair 128 / 4 heads: the width-generic path (d_pair <= 512, heads of 16 / 32 channels, any L % 16),
inference and training, masks, padding, B > 1, both nodes, dropout, graph capture.

Accuracy is judged against an fp32 reference with the bf16 PyTorch module's own error on the same inputs as the yardstick.
"""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import triattn_h100_sm90 as triattn_h100
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# (d_pair, d_hidden, heads): the registered widths (MiniWorld, templates, Protenix / OpenDDE trunks), a 16-channel head, 8 heads at 128
CONFIGS = [
    (64, 128, 4),
    (64, 64, 2),
    (128, 128, 8),
    (256, 256, 8),
    (384, 384, 12),
    (512, 512, 16),
    (64, 64, 4),
    (128, 128, 4),
]
IDS = [f"C{c}-H{h}x{hd // h}" for c, hd, h in CONFIGS]


def relative(a, b):
    return float(
        (a.detach().float() - b.detach().float()).norm()
        / b.detach().float().norm().clamp_min(1e-12)
    )


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return module


def make(cfg, impl, starting=True, p_drop=0.0, dtype=torch.bfloat16, seed=0):
    c, hd, h = cfg
    torch.manual_seed(seed)
    m = TriangleAttention(
        c, h, d_hidden=hd, starting=starting, p_drop=p_drop, implementation=impl
    )
    return randomize(m).cuda().to(dtype)


def trio(cfg, starting=True, p_drop=0.0):
    ours = make(cfg, ImplementationType.MINIWORLD, starting, p_drop)
    ref16 = make(cfg, ImplementationType.PYTORCH, starting, p_drop)
    ref16.load_state_dict(ours.state_dict())
    ref32 = make(cfg, ImplementationType.PYTORCH, starting, p_drop, dtype=torch.float32)
    ref32.load_state_dict({k: v.float() for k, v in ours.state_dict().items()})
    return ours, ref16, ref32


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 (sm_90) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


def check_inference(ours, ref16, ref32, x, mask):
    with torch.no_grad():
        assert triattn_h100.serves(ours.eval(), x, mask)
        got = ours(x, mask)
        base = ref16.eval()(x, mask)
        want = ref32.eval()(x.float(), mask)
    assert torch.isfinite(got).all()
    err, err16 = (
        relative(got.float() - x.float(), want - x.float()),
        relative(base.float() - x.float(), want - x.float()),
    )
    assert err <= 1.25 * err16 + 1e-3, (err, err16)


def check_training(ours, ref16, ref32, x, mask, dy, scale=None):
    assert triattn_h100.serves(ours.train(), x, mask)
    got = {}
    for name, m, xin in (
        ("ours", ours, x),
        ("ref16", ref16, x),
        ("ref32", ref32, x.float()),
    ):
        xi = xin.clone().requires_grad_()
        if scale is None:
            y = m.train()(xi, mask)
        else:
            torch.manual_seed(1234)
            y = (
                xi + m.train()._attention(xi, mask) * scale.to(xin.dtype)
                if name != "ours"
                else m.train()(xi, mask)
            )
        y.backward(dy.to(y.dtype))
        got[name] = (y, [xi.grad, *(p.grad for p in m.parameters())])
    e = relative(got["ours"][0].float() - x.float(), got["ref32"][0] - x.float())
    e16 = relative(got["ref16"][0].float() - x.float(), got["ref32"][0] - x.float())
    assert e <= 1.25 * e16 + 1e-3, (e, e16)
    names = ["x", *(n for n, _ in ours.named_parameters())]
    for i, n in enumerate(names):
        err = relative(got["ours"][1][i], got["ref32"][1][i])
        err16 = relative(got["ref16"][1][i], got["ref32"][1][i])
        assert err <= 1.25 * err16 + 2e-3, (n, err, err16)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
@pytest.mark.parametrize("length", [128, 384])
def test_inference(cfg, length):
    ours, ref16, ref32 = trio(cfg)
    x = torch.randn(1, length, length, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    check_inference(ours, ref16, ref32, x, mask)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
@pytest.mark.parametrize("length", [128, 256])
def test_training(cfg, length):
    ours, ref16, ref32 = trio(cfg)
    x = torch.randn(1, length, length, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    dy = torch.randn(1, length, length, cfg[0], device="cuda") * 0.1
    check_training(ours, ref16, ref32, x, mask, dy)


@pytest.mark.parametrize(
    "cfg", [(64, 128, 4), (256, 256, 8)], ids=["C64-H4x32", "C256-H8x32"]
)
@pytest.mark.parametrize("length", [80, 192, 304])
def test_padded_lengths(cfg, length):
    """L not a multiple of 128 (any multiple of 16 is served; the pad keys are masked)."""
    if length % 16:
        pytest.skip("L % 16")
    ours, ref16, ref32 = trio(cfg)
    x = torch.randn(1, length, length, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    check_inference(ours, ref16, ref32, x, mask)
    dy = torch.randn(1, length, length, cfg[0], device="cuda") * 0.1
    check_training(ours, ref16, ref32, x, mask, dy)


@pytest.mark.parametrize("cfg", [(64, 128, 4), (128, 128, 4)], ids=["generic", "fused"])
def test_padded_lengths_no_mask(cfg):
    ours, ref16, ref32 = trio(cfg)
    x = torch.randn(1, 208, 208, cfg[0], device="cuda", dtype=torch.bfloat16)
    check_inference(ours, ref16, ref32, x, None)


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize(
    "cfg", [(64, 64, 2), (384, 384, 12)], ids=["C64-H2", "C384-H12"]
)
def test_nodes_batch_two(cfg, starting):
    ours, ref16, ref32 = trio(cfg, starting)
    x = torch.randn(2, 128, 128, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(2, 128, device="cuda") > 0.2
    check_inference(ours, ref16, ref32, x, mask)
    dy = torch.randn(2, 128, 128, cfg[0], device="cuda") * 0.1
    check_training(ours, ref16, ref32, x, mask, dy)


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize(
    "cfg", [(64, 128, 4), (256, 256, 8)], ids=["C64-H4x32", "C256-H8x32"]
)
def test_training_dropout(cfg, starting):
    ours, ref16, ref32 = trio(cfg, starting, p_drop=0.25)
    L = 192
    x = torch.randn(1, L, L, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, L, device="cuda") > 0.2
    dy = torch.randn(1, L, L, cfg[0], device="cuda") * 0.1
    torch.manual_seed(1234)
    scale = ours._make_drop_scale(x, 0.25)
    # `ours` draws the same mask from the same seed inside its forward
    assert triattn_h100.serves(ours.train(), x, mask)
    got = {}
    for name, m, xin in (
        ("ours", ours, x),
        ("ref16", ref16, x),
        ("ref32", ref32, x.float()),
    ):
        xi = xin.clone().requires_grad_()
        torch.manual_seed(1234)
        y = (
            m.train()(xi, mask)
            if name == "ours"
            else xi + m.train()._attention(xi, mask) * scale.to(xin.dtype)
        )
        y.backward(dy.to(y.dtype))
        got[name] = (y, [xi.grad, *(p.grad for p in m.parameters())])
    e = relative(got["ours"][0].float() - x.float(), got["ref32"][0] - x.float())
    e16 = relative(got["ref16"][0].float() - x.float(), got["ref32"][0] - x.float())
    assert e <= 1.25 * e16 + 1e-3, (e, e16)
    names = ["x", *(n for n, _ in ours.named_parameters())]
    for i, n in enumerate(names):
        err = relative(got["ours"][1][i], got["ref32"][1][i])
        err16 = relative(got["ref16"][1][i], got["ref32"][1][i])
        assert err <= 1.25 * err16 + 2e-3, (n, err, err16)


def test_fp32_master_parameters():
    cfg = (256, 256, 8)
    m16 = make(cfg, ImplementationType.MINIWORLD)
    m32 = make(cfg, ImplementationType.MINIWORLD, dtype=torch.float32)
    m32.load_state_dict({k: v.float() for k, v in m16.state_dict().items()})
    x = torch.randn(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    assert triattn_h100.serves(m32.train(), x)
    out = []
    for m in (m16, m32):
        xi = x.clone().requires_grad_()
        m.train()(xi).backward(dy)
        out.append([xi.grad, *(p.grad for p in m.parameters())])
    for p in m32.parameters():
        assert p.grad.dtype == torch.float32
    for a, b in zip(*out, strict=True):
        assert relative(a, b) < 1e-2


def test_deterministic():
    """Two identical runs: the output and the gradients that do not pass through the bias gradient (q, k, v, gate, out weights) agree bit for
    bit; x, the LayerNorm and bias-projection gradients carry the core's fp32 reduce-add of dbias (L2 add order), one bf16 ulp at most."""
    ours = make((256, 256, 8), ImplementationType.MINIWORLD)
    x = torch.randn(1, 256, 256, 256, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, 256, device="cuda") > 0.2
    dy = torch.randn_like(x)
    runs = []
    for _ in range(3):
        xi = x.clone().requires_grad_()
        y = ours.train()(xi, mask)
        y.backward(dy)
        runs.append([y.detach(), xi.grad, *(p.grad.clone() for p in ours.parameters())])
        ours.zero_grad()
    names = ["y", "x", *(n for n, _ in ours.named_parameters())]
    for i, n in enumerate(names):
        for r in runs[1:]:
            if n in (
                "y",
                "to_query.weight",
                "to_key.weight",
                "to_value.weight",
                "to_gate.weight",
                "to_out.weight",
            ):
                assert torch.equal(runs[0][i], r[i]), n
            else:
                assert relative(runs[0][i], r[i]) < 1e-2, n


def test_serves_rejects_other_shapes():
    m = make((64, 128, 4), ImplementationType.MINIWORLD)
    ok = torch.zeros(1, 128, 128, 64, device="cuda", dtype=torch.bfloat16)
    assert triattn_h100.serves(m, ok)
    assert not triattn_h100.serves(
        m, torch.zeros(1, 200, 200, 64, device="cuda", dtype=torch.bfloat16)
    )  # L % 16
    assert not triattn_h100.serves(
        m, torch.zeros(1, 128, 256, 64, device="cuda", dtype=torch.bfloat16)
    )  # not square
    assert not triattn_h100.serves(m, ok.float())
    big_head = TriangleAttention(
        256, 4, implementation=ImplementationType.MINIWORLD
    ).cuda()  # 64-channel heads
    assert not triattn_h100.serves(
        big_head, torch.zeros(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    )
    qkn = TriangleAttention(
        64,
        4,
        d_hidden=128,
        use_qk_norm=True,
        implementation=ImplementationType.MINIWORLD,
    ).cuda()
    assert not triattn_h100.serves(qkn, ok)
    assert not triattn_h100.serves(make((64, 128, 4), ImplementationType.PYTORCH), ok)
    m._h100_cuda = False
    assert not triattn_h100.serves(m, ok)


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("cfg", [(64, 128, 4), (384, 384, 12)], ids=["C64", "C384"])
def test_cuda_graph_capture(cfg, grad):
    m = make(cfg, ImplementationType.MINIWORLD)
    x = torch.randn(
        1, 256, 256, cfg[0], device="cuda", dtype=torch.bfloat16, requires_grad=grad
    )
    mask = torch.rand(1, 256, device="cuda") > 0.2
    params = [x, *m.parameters()] if grad else []
    dy = torch.randn_like(x)

    def step():
        with torch.set_grad_enabled(grad):
            y = m(x, mask)
            return torch.autograd.grad(y, params, dy) if grad else (y,)

    eager = [t.clone() for t in step()]
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
    for a, b in zip(out, eager, strict=True):
        assert relative(a, b) < 1e-3


@pytest.mark.parametrize(
    "cfg",
    [(64, 128, 4), (128, 128, 4), (256, 256, 8)],
    ids=["generic-C64", "fused-C128", "generic-C256"],
)
def test_compiled_module_captures_in_a_cuda_graph(cfg):
    """The benchmark harness' regime: the module under torch.compile, replayed from a CUDA graph (the launches are opaque custom ops;
    traced weight packs were recompiled inside the capture and failed it)."""
    import torch._dynamo as dynamo

    dynamo.reset()
    m = make(cfg, ImplementationType.MINIWORLD).eval()
    mc = torch.compile(m, dynamic=False)
    x = torch.randn(1, 256, 256, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 256, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        eager = m(x, mask)
        for _ in range(3):
            mc(x, mask)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mc(x, mask)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = mc(x, mask)
        g.replay()
        torch.cuda.synchronize()
    assert relative(y, eager) < 1e-3
