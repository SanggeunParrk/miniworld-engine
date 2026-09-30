"""B200 TriMul: module dispatch, inference and training accuracy (D128 bidirectional at every L; every width D64-D512)."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import trimul_b200
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

LENGTHS = [128, 256, 384, 512, 640, 768]


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
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


@pytest.mark.parametrize("length", LENGTHS)
def test_inference(length):
    torch.manual_seed(309)
    m = randomize(BidirectionalTriangleMultiplication(128, implementation=ImplementationType.MINIWORLD)
                  .cuda().bfloat16()).eval()
    ref = BidirectionalTriangleMultiplication(128, implementation=ImplementationType.PYTORCH).cuda().eval()
    ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    x = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    with torch.no_grad():
        assert trimul_b200.serves(m, x)
        got = m(x, mask)
        want = ref(x.float(), mask)
    assert relative(got, want) < 0.006


@pytest.mark.parametrize("length", LENGTHS)
def test_training(length):
    """bf16 input, fp32 master parameters; every gradient against an fp32 autograd reference."""
    torch.manual_seed(311)
    m = randomize(BidirectionalTriangleMultiplication(128, p_drop=0.0, implementation=ImplementationType.MINIWORLD)
                  .cuda()).train()
    ref = BidirectionalTriangleMultiplication(128, p_drop=0.0, implementation=ImplementationType.PYTORCH).cuda().train()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    mask = torch.rand(1, length, device="cuda") > 0.2
    dy = torch.randn(1, length, length, 128, device="cuda") * 0.1
    assert trimul_b200.serves(m, x)
    y = m(x, mask)
    y.backward(dy.to(y.dtype))
    yr = ref(xr, mask)
    yr.backward(dy)
    assert relative(y, yr) < 0.006
    assert relative(x.grad, xr.grad) < 0.01
    for (name, p), (_, pr) in zip(m.named_parameters(), ref.named_parameters(), strict=True):
        assert relative(p.grad, pr.grad) < 0.01, name


def test_nograd_dropout_matches_training_forward():
    """No-grad dropout forward (the saving K3 into throwaway buffers) equals the grad-enabled forward."""
    torch.manual_seed(313)
    m = randomize(BidirectionalTriangleMultiplication(128, p_drop=0.25, implementation=ImplementationType.MINIWORLD)
                  .cuda().bfloat16()).train()
    x = torch.randn(1, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(5)
    with torch.no_grad():
        a = m(x)
    torch.manual_seed(5)
    b = m(x.clone().requires_grad_())
    assert torch.equal(a, b.detach())


def test_serves_train_d128_one_direction_only():
    """D128: one direction goes to b200_train, bidirectional stays with b200_bidir (``serves``)."""
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    uni = _wide_module("out", 128, ImplementationType.MINIWORLD).cuda()
    bi = _wide_module("bidir", 128, ImplementationType.MINIWORLD).cuda()
    assert trimul_b200.serves_train(uni, x, bidirectional=False)
    assert not trimul_b200.serves_train(bi, x, bidirectional=True)
    assert trimul_b200.serves(bi, x)


def test_serves_rejects_other_shapes():
    m = BidirectionalTriangleMultiplication(128, implementation=ImplementationType.MINIWORLD).cuda()
    for shape in ((1, 200, 200, 128), (2, 128, 128, 128)):
        assert not trimul_b200.serves(m, torch.zeros(shape, device="cuda", dtype=torch.bfloat16))
    assert not trimul_b200.serves(m, torch.zeros(1, 128, 128, 128, device="cuda"))
    m64 = BidirectionalTriangleMultiplication(64, implementation=ImplementationType.MINIWORLD).cuda()
    assert not trimul_b200.serves(m64, torch.zeros(1, 128, 128, 64, device="cuda", dtype=torch.bfloat16))


def test_pack_wp_matches_the_extension():
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_bidir

    wp = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(b200_bidir.pack_wp(wp), b200_bidir._ext().k3_pack_wp(wp))


@pytest.mark.parametrize("grad", [False, True])
def test_cuda_graph_capture(grad):
    """The module path launches nothing host-dependent: it captures and replays in a CUDA graph."""
    torch.manual_seed(317)
    m = randomize(BidirectionalTriangleMultiplication(128, p_drop=0.0, implementation=ImplementationType.MINIWORLD)
                  .cuda().bfloat16())
    x = torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    params = [x, *m.parameters()] if grad else []
    dy = torch.randn_like(x)

    def step():
        with torch.set_grad_enabled(grad):
            y = m(x)
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
        assert relative(a, b) < 1e-3   # weight gradients accumulate with atomics: order-dependent


def _wide_module(kind, width, impl):
    if kind == "bidir":
        return BidirectionalTriangleMultiplication(width, implementation=impl)
    return TriangleMultiplication(width, d_hidden=width, outgoing=kind == "out", implementation=impl)


@pytest.mark.parametrize("kind", ["bidir", "out", "in"])
@pytest.mark.parametrize("length", [128, 384])
@pytest.mark.parametrize("width", [64, 128, 256, 384, 512])
def test_inference_every_width(width, length, kind):
    torch.manual_seed(331)
    ref = randomize(_wide_module(kind, width, ImplementationType.PYTORCH).cuda()).eval()
    m = _wide_module(kind, width, ImplementationType.MINIWORLD).cuda().bfloat16().eval()
    m.load_state_dict(ref.state_dict())
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    with torch.no_grad():
        assert trimul_b200.serves_inference(m, x, bidirectional=kind == "bidir")
        got = m(x, mask)
        want = ref(x.float(), mask)
    assert relative(got, want) < 0.006


def test_wide_inference_graph_capture():
    torch.manual_seed(337)
    m = randomize(_wide_module("bidir", 256, ImplementationType.MINIWORLD).cuda().bfloat16()).eval()
    x = torch.randn(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = m(x).clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(x)
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize("kind", ["bidir", "out", "in"])
@pytest.mark.parametrize("width", [256, 384, 512])
@pytest.mark.parametrize("length", [128, 256])
def test_wide_training(length, width, kind):
    """bf16 input, fp32 master parameters, masked; every gradient against an fp32 autograd reference. (L256: gate_bwd's
    grid-stride tail, where M D / 8 is not a multiple of its thread count.)"""
    torch.manual_seed(341)
    ref = randomize(_wide_module(kind, width, ImplementationType.PYTORCH).cuda()).train()
    ref.p_drop = 0.0
    m = _wide_module(kind, width, ImplementationType.MINIWORLD).cuda().train()
    m.p_drop = 0.0
    m.load_state_dict(ref.state_dict())
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    mask = torch.rand(1, length, device="cuda") > 0.2
    dy = torch.randn(1, length, length, width, device="cuda") * 0.1
    assert trimul_b200.serves_train(m, x, bidirectional=kind == "bidir")
    y = m(x, mask)
    y.backward(dy.to(y.dtype))
    yr = ref(xr, mask)
    yr.backward(dy)
    assert relative(y, yr) < 0.006
    assert relative(x.grad, xr.grad) < 0.01
    for (name, p), (_, pr) in zip(m.named_parameters(), ref.named_parameters(), strict=True):
        assert relative(p.grad, pr.grad) < 0.012, name


def test_wide_training_dropout():
    """The row-dropout scale enters the output and every gradient: compare with ds applied to the fp32 reference by hand."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import trimul_train

    torch.manual_seed(343)
    width, length = 256, 128
    ref = randomize(_wide_module("bidir", width, ImplementationType.PYTORCH).cuda()).train()
    ref.p_drop = 0.0
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    ds = (torch.rand(length, width, device="cuda") > 0.25).float() / 0.75
    dy = torch.randn(1, length, length, width, device="cuda") * 0.1
    bf = torch.bfloat16
    ws = [p.detach().to(bf).clone().requires_grad_() for p in (ref.to_left.weight, ref.to_left_gate.weight, ref.to_right.weight,
                                                               ref.to_right_gate.weight, ref.to_gate.weight, ref.to_out.weight)]
    lns = [p.detach().float().clone().requires_grad_() for p in (ref.ln_pair.weight, ref.ln_pair.bias, ref.ln_out.weight, ref.ln_out.bias)]
    y = trimul_train([x, *ws, *lns], None, ds.to(bf).contiguous(), 0)
    y.backward(dy.to(bf))
    yr = xr + (ref(xr) - xr) * ds
    yr.backward(dy)
    assert relative(y, yr) < 0.006
    assert relative(x.grad, xr.grad) < 0.01
    for w, name in zip(ws, ("to_left", "to_left_gate", "to_right", "to_right_gate", "to_gate", "to_out"), strict=True):
        assert relative(w.grad, getattr(ref, name).weight.grad) < 0.012, name


@pytest.mark.parametrize("kind", ["bidir", "out", "in"])
@pytest.mark.parametrize("length", [128, 256])
@pytest.mark.parametrize("width", [64, 128])
def test_small_training(width, length, kind):
    """D64 (b1s / b7m) and D128 one direction (b1g / b7g) fused training: bf16 input, fp32 master parameters, masked; every
    gradient against fp32 autograd. (D128 bidirectional is b200_bidir's: test_training.)"""
    if width == 128 and kind == "bidir":
        pytest.skip("D128 bidirectional training is b200_bidir (test_training)")
    torch.manual_seed(347)
    ref = randomize(_wide_module(kind, width, ImplementationType.PYTORCH).cuda()).train()
    ref.p_drop = 0.0
    m = _wide_module(kind, width, ImplementationType.MINIWORLD).cuda().train()
    m.p_drop = 0.0
    m.load_state_dict(ref.state_dict())
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    mask = torch.rand(1, length, device="cuda") > 0.2
    dy = torch.randn(1, length, length, width, device="cuda") * 0.1
    assert trimul_b200.serves_train(m, x, bidirectional=kind == "bidir")
    y = m(x, mask)
    y.backward(dy.to(y.dtype))
    yr = ref(xr, mask)
    yr.backward(dy)
    assert relative(y, yr) < 0.006
    assert relative(x.grad, xr.grad) < 0.01
    for (name, p), (_, pr) in zip(m.named_parameters(), ref.named_parameters(), strict=True):
        assert relative(p.grad, pr.grad) < 0.012, name


@pytest.mark.parametrize("width", [64, 128])
def test_small_training_dropout(width):
    """D64 / D128 one direction: the row-dropout scale enters the output and every gradient (ds applied to the fp32 reference
    by hand)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import trimul_train

    torch.manual_seed(349)
    length = 256
    ref = randomize(_wide_module("out", width, ImplementationType.PYTORCH).cuda()).train()
    ref.p_drop = 0.0
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    ds = (torch.rand(length, width, device="cuda") > 0.25).float() / 0.75
    dy = torch.randn(1, length, length, width, device="cuda") * 0.1
    bf = torch.bfloat16
    ws = [p.detach().to(bf).clone().requires_grad_() for p in (ref.to_left.weight, ref.to_left_gate.weight, ref.to_right.weight,
                                                               ref.to_right_gate.weight, ref.to_gate.weight, ref.to_out.weight)]
    lns = [p.detach().float().clone().requires_grad_() for p in (ref.ln_pair.weight, ref.ln_pair.bias, ref.ln_out.weight, ref.ln_out.bias)]
    y = trimul_train([x, *ws, *lns], None, ds.to(bf).contiguous(), 1)
    y.backward(dy.to(bf))
    yr = xr + (ref(xr) - xr) * ds
    yr.backward(dy)
    assert relative(y, yr) < 0.006
    assert relative(x.grad, xr.grad) < 0.01
    for w, name in zip(ws, ("to_left", "to_left_gate", "to_right", "to_right_gate", "to_gate", "to_out"), strict=True):
        assert relative(w.grad, getattr(ref, name).weight.grad) < 0.012, name
    for p, name in zip(lns, ("ln_pair.weight", "ln_pair.bias", "ln_out.weight", "ln_out.bias"), strict=True):
        pr = ref.get_parameter(name)
        assert relative(p.grad, pr.grad) < 0.012, name


@pytest.mark.parametrize(("width", "kind"), [(64, "bidir"), (128, "out")])
def test_small_training_graph_capture(width, kind):
    """D64 / D128 forward + backward replay in a CUDA graph (the ring flags return to 0 inside every launch)."""
    torch.manual_seed(351)
    m = randomize(_wide_module(kind, width, ImplementationType.MINIWORLD).cuda()).train().to(torch.bfloat16)
    m.p_drop = 0.0
    x = torch.randn(1, 256, 256, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)

    def step():
        x.grad = None
        for p in m.parameters():
            p.grad = None
        m(x).backward(dy)
        assert x.grad is not None
        return x.grad.clone()

    eager = step()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize(("width", "kind"), [(64, "bidir"), (64, "out"), (128, "out"), (256, "bidir"), (512, "out")])
def test_training_layernorm_gradients_deterministic(width, kind):
    """The fp32 LayerNorm gradients are fixed-order sums: two backward passes give identical bits (the benchmark harness
    compares its timed execution against a preparation run at 1e-4 relative for fp32 tensors)."""
    torch.manual_seed(353)
    m = randomize(_wide_module(kind, width, ImplementationType.MINIWORLD).cuda()).train().to(torch.bfloat16)
    m.p_drop = 0.0
    x = torch.randn(1, 256, 256, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    assert m.ln_pair.weight.dtype == torch.float32      # the module keeps its LayerNorms in fp32
    runs = []
    for _ in range(2):
        for p in m.parameters():
            p.grad = None
        m(x).backward(dy)
        runs.append([m.ln_pair.weight.grad.clone(), m.ln_pair.bias.grad.clone(), m.ln_out.weight.grad.clone(),
                     m.ln_out.bias.grad.clone()])
    for a, b in zip(*runs, strict=True):
        assert torch.equal(a, b)
