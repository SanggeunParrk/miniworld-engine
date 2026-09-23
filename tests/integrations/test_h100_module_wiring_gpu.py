"""Exercise production module dispatch with no external payload or opt-in variables."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit, trimul_h100
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


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


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    for name in (
        "TRIMUL_NATIVE_BUILD_DIR",
        "OPT_CORE_DIR",
        "MINIWORLD_PWA_TRAIN",
        "MINIWORLD_OPM_TRAIN",
        "MINIWORLD_PWA_INFER",
    ):
        monkeypatch.delenv(name, raising=False)
    old = settings.configure(
        engine_backend="auto", trimul_h100_training_widths=(64, 128, 256, 384, 512)
    )
    try:
        yield
    finally:
        settings.configure(**vars(old))


@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("length", [256, 384, 768])
@pytest.mark.parametrize("width", [128])
def test_trimul_inference_without_payload(bidirectional, length, width):
    torch.manual_seed(309)
    cls = (
        BidirectionalTriangleMultiplication if bidirectional else TriangleMultiplication
    )
    m = randomize(
        cls(width, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    ).eval()
    ref = cls(width, implementation=ImplementationType.PYTORCH).cuda().bfloat16().eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=bidirectional)
        got = m(x, mask)
        want = ref(x, mask)
        assert relative(got, want) < 0.006
        # Weights are live, including the graph replay path.
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            m(x, mask)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            out = m(x, mask)
        x.mul_(0.97)
        m.to_out.weight.mul_(0.93)
        graph.replay()
        assert relative(out, m(x, mask)) < 1e-6


def test_trimul_compile_backward_and_saved_tensor_ownership(monkeypatch):
    torch.manual_seed(401)
    m = randomize(
        BidirectionalTriangleMultiplication(
            128, implementation=ImplementationType.MINIWORLD, p_drop=0.25
        )
        .cuda()
        .bfloat16()
    )
    x = torch.randn(
        1, 384, 384, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.rand(1, 384, device="cuda") > 0.2
    ds = (torch.rand(1, 1, 384, 128, device="cuda") > 0.25).bfloat16() * 4 / 3
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    assert trimul_h100.serves(m, x)
    dy = torch.randn_like(x)
    args = (x, *m.parameters())
    y = m(x, mask)
    expected = torch.autograd.grad(y, args, dy)
    compiled = torch.compile(m, fullgraph=True, options={"triton.cudagraphs": False})
    z = compiled(x, mask)
    actual = torch.autograd.grad(z, args, dy)
    assert relative(z, y) < 1e-6
    for got, want in zip(actual, expected, strict=True):
        assert relative(got, want) < 1e-6
    # A later forward must not overwrite the first forward's saved activations.
    first = m(x, mask)
    m(x * 0.83, mask)
    other = torch.autograd.grad(first, args, dy)
    for got, want in zip(other, expected, strict=True):
        assert relative(got, want) < 1e-6


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_token_dit_inference_live_inputs_weights_and_mask(dtype):
    torch.manual_seed(811)
    m = randomize(
        DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(dtype)
    ).eval()
    ref = DiTBlock(implementation=ImplementationType.PYTORCH).cuda().to(dtype).eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(1, 1, 384, 768, device="cuda", dtype=dtype).transpose(-1, -2).contiguous().transpose(-1, -2)
    c = torch.randn(1, 1, 384, 384, device="cuda", dtype=dtype)
    p = torch.randn(1, 384, 384, 256, device="cuda", dtype=dtype)[..., ::2]
    mask = torch.rand(1, 384, device="cuda") > 0.2
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        got = m(x, c, p, mask)
        want = ref(x, c, p, mask)
        assert relative(got, want) < 0.025
        compiled = torch.compile(
            m, fullgraph=True, options={"triton.cudagraphs": False}
        )
        assert relative(compiled(x, c, p, mask), got) < 1e-5
        old = got.clone()
        m.transition.squeeze.weight.mul_(0.8)
        new = m(x, c, p, mask)
        assert not torch.equal(old, new)
        # No stale pair-bias cache: a live changed pair must change the output.
        assert not torch.equal(new, m(x, c, p * 0.7, mask))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = m(x, c, p, mask)
        p.mul_(.9)
        graph.replay()
        assert relative(captured, m(x, c, p, mask)) < 1e-5



@pytest.mark.parametrize("D", [64, 128, 256, 384, 512])
@pytest.mark.parametrize("L", [384, 768])
def test_trimul_width_training(D, L, monkeypatch):
    """All connected widths: masks, dropout, output and every gradient vs PyTorch."""
    import torch.nn.functional as F

    # Ten independent oracle shapes exceed Dynamo's default per-code cache limit.
    torch.compiler.reset()
    torch.manual_seed(9023)
    m = (
        BidirectionalTriangleMultiplication(
            D, implementation=ImplementationType.MINIWORLD, p_drop=0.25
        )
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=D**-0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.rand(1, L, device="cuda") > 0.15
    ds = (torch.rand(1, 1, L, D, device="cuda") > 0.25).bfloat16() * (4 / 3)
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    weights = (
        m.to_left.weight,
        m.to_left_gate.weight,
        m.to_right.weight,
        m.to_right_gate.weight,
        m.to_gate.weight,
        m.to_out.weight,
        m.ln_pair.weight,
        m.ln_pair.bias,
        m.ln_out.weight,
        m.ln_out.bias,
    )

    def ref(x, *w):
        wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = w
        H = 2 * D
        xn = F.layer_norm(x.float(), (D,), gi, bi, 1e-5).bfloat16()
        pm = (mask[:, :, None] & mask[:, None, :])[..., None]
        left = torch.sigmoid(F.linear(xn, wlg)) * F.linear(xn, wl) * pm
        right = torch.sigmoid(F.linear(xn, wrg)) * F.linear(xn, wr) * pm
        tri = torch.cat(
            (
                torch.einsum("bikd,bjkd->bijd", left[..., :D], right[..., :D]),
                torch.einsum("bkid,bkjd->bijd", left[..., D:], right[..., D:]),
            ),
            dim=-1,
        )
        out = F.layer_norm(tri.float(), (H,), go, bo, 1e-5).bfloat16()
        return x + F.linear(out, wp) * torch.sigmoid(F.linear(xn, wg)) * ds

    assert trimul_h100.serves(m, x), "did not select new route"
    y = m(x, mask)
    dy = torch.randn_like(y)
    g = torch.autograd.grad(y, (x, *weights), dy)
    compiled_ref = torch.compile(ref, fullgraph=True, options={"triton.cudagraphs": False})
    z = compiled_ref(x, *weights)
    h = torch.autograd.grad(z, (x, *weights), dy)
    e = [
        float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
        for a, b in zip((y, *g), (z, *h), strict=True)
    ]
    print("CHECK", D, L, e, flush=True)
    assert e[0] < 0.005
    assert max(e[1:]) < 0.01


@pytest.mark.parametrize("width", [64, 128])
def test_trimul_training_graph_replay_live_weights(width, monkeypatch):
    """A captured backward must initialize its own scratch and read live weights."""
    torch.compiler.reset()
    m = randomize(BidirectionalTriangleMultiplication(width, implementation=ImplementationType.MINIWORLD,
                                                      p_drop=.25).cuda().bfloat16())
    x = torch.randn(1, 384, 384, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.rand(1, 384, device="cuda") > .2
    ds = (torch.rand(1, 1, 384, width, device="cuda") > .25).bfloat16() * (4/3)
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    dy = torch.randn_like(x)
    args = (x, *m.parameters())

    def run():
        y = m(x, mask)
        return y, torch.autograd.grad(y, args, dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got, grads = run()
    with torch.no_grad():
        x.mul_(.91)
        m.to_left.weight.mul_(.83)
    for _ in range(2):
        graph.replay()
        want, expected = run()
        assert relative(got, want) < 1e-6
        for a, b in zip(grads, expected, strict=True):
            assert relative(a, b) < 1e-5


@pytest.mark.parametrize(("width", "bidirectional"), [(64,False),(64,True),(256,False),(384,False)])
def test_trimul_inference_width_coverage(width,bidirectional):
    test_trimul_inference_without_payload(bidirectional,384,width)
