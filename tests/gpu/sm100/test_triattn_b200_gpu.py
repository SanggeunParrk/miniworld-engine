"""B200 TriangleAttention (d_pair 128, 4 heads): module dispatch, inference and training accuracy, masks, graph capture.

Accuracy is judged against an fp32 reference, with the bf16 PyTorch module's own error on the same inputs as the yardstick:
the kernels may not be meaningfully worse than the reference computed in bf16.
"""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import triattn_b200
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

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


def make(impl, starting=True, p_drop=0.0, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    m = TriangleAttention(128, 4, starting=starting, p_drop=p_drop, implementation=impl)
    return randomize(m).cuda().to(dtype)


def trio(starting=True):
    """(ours, PyTorch in bf16, PyTorch in fp32), sharing one set of bf16-representable parameters."""
    ours = make(ImplementationType.MINIWORLD, starting)
    ref16 = make(ImplementationType.PYTORCH, starting)
    ref16.load_state_dict(ours.state_dict())
    ref32 = make(ImplementationType.PYTORCH, starting, dtype=torch.float32)
    ref32.load_state_dict({k: v.float() for k, v in ours.state_dict().items()})
    return ours, ref16, ref32


def pad_mask(length, masked):
    """[1, L] key mask with the given key positions masked out."""
    m = torch.ones(1, length, dtype=torch.bool, device="cuda")
    m[0, masked] = False
    return m


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


def check_inference(ours, ref16, ref32, x, mask):
    with torch.no_grad():
        assert triattn_b200.serves(ours.eval(), x, mask)
        got = ours(x, mask)
        base = ref16.eval()(x, mask)
        want = ref32.eval()(x.float(), mask)
    assert torch.isfinite(got).all()
    # the attention update (output minus the residual) is what the kernels compute
    err, err16 = relative(got.float() - x.float(), want - x.float()), relative(base.float() - x.float(), want - x.float())
    assert err <= 1.25 * err16 + 1e-3, (err, err16)


@pytest.mark.parametrize("length", LENGTHS)
def test_inference(length):
    ours, ref16, ref32 = trio()
    x = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    check_inference(ours, ref16, ref32, x, mask)


@pytest.mark.parametrize("length", LENGTHS)
def test_training(length):
    """Output and every gradient against fp32 autograd, yardstick = the bf16 PyTorch module."""
    ours, ref16, ref32 = trio()
    x = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    dy = torch.randn(1, length, length, 128, device="cuda") * 0.1
    assert triattn_b200.serves(ours.train(), x, mask)
    grads = {}
    for name, m, xin in (("ours", ours, x), ("ref16", ref16, x), ("ref32", ref32, x.float())):
        xi = xin.clone().requires_grad_()
        y = m.train()(xi, mask)
        y.backward(dy.to(y.dtype))
        grads[name] = [xi.grad, *(p.grad for p in m.parameters())]
    names = ["x", *(n for n, _ in ours.named_parameters())]
    for i, n in enumerate(names):
        err = relative(grads["ours"][i], grads["ref32"][i])
        err16 = relative(grads["ref16"][i], grads["ref32"][i])
        assert err <= 1.25 * err16 + 2e-3, (n, err, err16)


def test_fp32_master_parameters():
    """fp32 parameters with a bf16 activation: gradients come back in fp32 and match the bf16-parameter run."""
    ours16 = make(ImplementationType.MINIWORLD)
    ours32 = make(ImplementationType.MINIWORLD, dtype=torch.float32)
    ours32.load_state_dict({k: v.float() for k, v in ours16.state_dict().items()})
    x = torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    assert triattn_b200.serves(ours32.train(), x)
    out = []
    for m in (ours16, ours32):
        xi = x.clone().requires_grad_()
        m.train()(xi).backward(dy)
        out.append([xi.grad, *(p.grad for p in m.parameters())])
    for p in ours32.parameters():
        assert p.grad.dtype == torch.float32
    for a, b in zip(*out, strict=True):
        assert relative(a, b) < 1e-2


def test_ending_node():
    ours, ref16, ref32 = trio(starting=False)
    x = torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16)
    check_inference(ours, ref16, ref32, x, pad_mask(256, slice(200, None)))


def test_batch_two():
    ours, ref16, ref32 = trio()
    x = torch.randn(2, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(2, 128, device="cuda") > 0.2
    check_inference(ours, ref16, ref32, x, mask)


@pytest.mark.parametrize("masked", [slice(300, None), slice(0, 32), slice(0, 128)], ids=["tail-pad", "first-32", "first-128"])
def test_mask_layouts(masked):
    """Padding at the end (the usual crop), and keys masked at the start: the softmax offset comes from the first keys."""
    ours, ref16, ref32 = trio()
    x = torch.randn(1, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    check_inference(ours, ref16, ref32, x, pad_mask(384, masked))


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
def test_training_dropout(starting):
    """Training with the module's broadcast dropout: the kernels apply the same draw (regenerated from the seed) as
    pair + drop o attention(pair); output and every gradient against fp32 autograd, yardstick = the bf16 PyTorch module."""
    ours, ref16, ref32 = trio(starting)
    for m in (ours, ref16, ref32):
        m.p_drop = 0.25
    L = 256
    x = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, L, device="cuda") > 0.2
    dy = torch.randn(1, L, L, 128, device="cuda") * 0.1
    assert triattn_b200.serves(ours.train(), x, mask)
    torch.manual_seed(1234)
    xo = x.clone().requires_grad_()
    y = ours.train()(xo, mask)
    y.backward(dy.to(y.dtype))
    torch.manual_seed(1234)
    scale = ours._make_drop_scale(x, 0.25)                  # the same draw, same shape and dtype
    got = {"ours": (y, [xo.grad, *(p.grad for p in ours.parameters())])}
    for name, m, xin in (("ref16", ref16, x), ("ref32", ref32, x.float())):
        xi = xin.clone().requires_grad_()
        yr = xi + m.train()._attention(xi, mask) * scale.to(xin.dtype)
        yr.backward(dy.to(yr.dtype))
        got[name] = (yr, [xi.grad, *(p.grad for p in m.parameters())])
    e = relative(got["ours"][0].float() - x.float(), got["ref32"][0] - x.float())
    e16 = relative(got["ref16"][0].float() - x.float(), got["ref32"][0] - x.float())
    assert e <= 1.25 * e16 + 1e-3, (e, e16)
    names = ["x", *(n for n, _ in ours.named_parameters())]
    for i, n in enumerate(names):
        err = relative(got["ours"][1][i], got["ref32"][1][i])
        err16 = relative(got["ref16"][1][i], got["ref32"][1][i])
        assert err <= 1.25 * err16 + 2e-3, (n, err, err16)


def test_dropout_modes_served():
    m = make(ImplementationType.MINIWORLD, p_drop=0.25)
    x = torch.zeros(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    assert triattn_b200.serves(m.train(), x)
    assert triattn_b200.serves(m.eval(), x)


def test_serves_rejects_other_shapes():
    m = make(ImplementationType.MINIWORLD)
    for shape in ((1, 200, 200, 128), (1, 128, 256, 128)):
        assert not triattn_b200.serves(m, torch.zeros(shape, device="cuda", dtype=torch.bfloat16))
    assert not triattn_b200.serves(m, torch.zeros(1, 128, 128, 128, device="cuda"))
    wide = TriangleAttention(256, 4, implementation=ImplementationType.MINIWORLD).cuda()
    assert not triattn_b200.serves(wide, torch.zeros(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16))
    eight = TriangleAttention(128, 8, implementation=ImplementationType.MINIWORLD).cuda()
    assert not triattn_b200.serves(eight, torch.zeros(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16))
    ref = make(ImplementationType.PYTORCH)
    assert not triattn_b200.serves(ref, torch.zeros(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16))
    x = torch.zeros(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    assert triattn_b200.serves(make(ImplementationType.TRITON), x)   # the module's TRITON backend hosts the CUDA path
    m._b200_cuda = False                                              # the switch keeps the Triton kernels
    assert not triattn_b200.serves(m, x)


@pytest.mark.parametrize("grad", [False, True])
def test_cuda_graph_capture(grad):
    """The module path launches nothing host-dependent: it captures and replays in a CUDA graph."""
    m = make(ImplementationType.MINIWORLD)
    x = torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
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


# ---- other widths (inference): the registered model shapes (d_pair, d_hidden, heads) ----
WIDE = [(64, 64, 4), (64, 128, 4), (64, 64, 2), (256, 256, 8), (384, 384, 12), (512, 512, 16)]
WIDE_IDS = ["64-4x16", "64-4x32", "64-2x32", "256-8x32", "384-12x32", "512-16x32"]


def make_wide(impl, d_pair, d_hidden, n_head, starting=True, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    m = TriangleAttention(d_pair, n_head, d_hidden=d_hidden, starting=starting, p_drop=0.0, implementation=impl)
    return randomize(m).cuda().to(dtype)


def trio_wide(cfg, starting=True):
    ours = make_wide(ImplementationType.MINIWORLD, *cfg, starting=starting)
    ref16 = make_wide(ImplementationType.PYTORCH, *cfg, starting=starting)
    ref16.load_state_dict(ours.state_dict())
    ref32 = make_wide(ImplementationType.PYTORCH, *cfg, starting=starting, dtype=torch.float32)
    ref32.load_state_dict({k: v.float() for k, v in ours.state_dict().items()})
    return ours, ref16, ref32


def check_wide(ours, ref16, ref32, x, mask):
    with torch.no_grad():
        assert triattn_b200.serves_wide(ours.eval(), x, mask)
        assert not triattn_b200.serves(ours, x, mask)
    check_inference_any(ours, ref16, ref32, x, mask)


def check_inference_any(ours, ref16, ref32, x, mask):
    with torch.no_grad():
        got = ours.eval()(x, mask)
        base = ref16.eval()(x, mask)
        want = ref32.eval()(x.float(), mask)
    assert torch.isfinite(got).all()
    err, err16 = relative(got.float() - x.float(), want - x.float()), relative(base.float() - x.float(), want - x.float())
    assert err <= 1.25 * err16 + 1e-3, (err, err16)


@pytest.mark.parametrize("length", [128, 256])
@pytest.mark.parametrize("cfg", WIDE, ids=WIDE_IDS)
def test_wide_inference(cfg, length):
    ours, ref16, ref32 = trio_wide(cfg)
    x = torch.randn(1, length, length, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    check_wide(ours, ref16, ref32, x, mask)


@pytest.mark.parametrize("cfg", [WIDE[0], WIDE[3]], ids=[WIDE_IDS[0], WIDE_IDS[3]])
def test_wide_ending_node_and_first_keys_masked(cfg):
    ours, ref16, ref32 = trio_wide(cfg, starting=False)
    x = torch.randn(1, 256, 256, cfg[0], device="cuda", dtype=torch.bfloat16)
    check_wide(ours, ref16, ref32, x, pad_mask(256, slice(0, 64)))


def test_wide_repacks_after_a_weight_update():
    ours, ref16, ref32 = trio_wide(WIDE[3])
    x = torch.randn(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ours.eval()(x)                                   # packs
        for m in (ours, ref16, ref32):
            m.to_value.weight.mul_(-0.5)
    check_wide(ours, ref16, ref32, x, None)


def test_wide_not_served_for_other_head_dims():
    four = make_wide(ImplementationType.MINIWORLD, 256, 256, 4)         # 64-channel heads
    x = torch.zeros(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert not triattn_b200.serves_wide(four.eval(), x)
    assert triattn_b200.serves_wide(make_wide(ImplementationType.MINIWORLD, 256, 256, 8).train(), x)   # training is served


def wide_grads(cfg, starting, length, p_drop, seed=1234):
    """(output, [dx, dparams...]) of ours / bf16 PyTorch / fp32 PyTorch on one draw of x, mask, dy and the dropout."""
    ours, ref16, ref32 = trio_wide(cfg, starting)
    for m in (ours, ref16, ref32):
        m.p_drop = p_drop
    x = torch.randn(1, length, length, cfg[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    dy = torch.randn(1, length, length, cfg[0], device="cuda") * 0.1
    assert triattn_b200.serves_wide(ours.train(), x, mask)
    torch.manual_seed(seed)
    xo = x.clone().requires_grad_()
    y = ours.train()(xo, mask)
    y.backward(dy.to(y.dtype))
    got = {"ours": (y, [xo.grad, *(p.grad for p in ours.parameters())])}
    scale = None
    if p_drop:
        torch.manual_seed(seed)
        scale = ours._make_drop_scale(x, p_drop)                  # the same draw, same shape and dtype
    for name, m, xin in (("ref16", ref16, x), ("ref32", ref32, x.float())):
        xi = xin.clone().requires_grad_()
        att = m.train()._attention(xi, mask)
        yr = xi + (att * scale.to(xin.dtype) if scale is not None else att)
        yr.backward(dy.to(yr.dtype))
        got[name] = (yr, [xi.grad, *(p.grad for p in m.parameters())])
    return x, got, ["x", *(n for n, _ in ours.named_parameters())]


def check_wide_grads(x, got, names):
    e = relative(got["ours"][0].float() - x.float(), got["ref32"][0] - x.float())
    e16 = relative(got["ref16"][0].float() - x.float(), got["ref32"][0] - x.float())
    assert e <= 1.25 * e16 + 1e-3, (e, e16)
    for i, n in enumerate(names):
        err = relative(got["ours"][1][i], got["ref32"][1][i])
        err16 = relative(got["ref16"][1][i], got["ref32"][1][i])
        assert err <= 1.25 * err16 + 2e-3, (n, err, err16)


@pytest.mark.parametrize("length", [128, 256])
@pytest.mark.parametrize("cfg", WIDE, ids=WIDE_IDS)
def test_wide_training(cfg, length):
    """Output and every gradient against fp32 autograd, yardstick = the bf16 PyTorch module."""
    check_wide_grads(*wide_grads(cfg, True, length, 0.0))


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize("cfg", [WIDE[0], WIDE[3]], ids=[WIDE_IDS[0], WIDE_IDS[3]])
def test_wide_training_dropout(cfg, starting):
    check_wide_grads(*wide_grads(cfg, starting, 256, 0.25))


def test_wide_fp32_master_parameters():
    """fp32 parameters with a bf16 activation: gradients come back in fp32 and match the bf16-parameter run."""
    m16 = make_wide(ImplementationType.MINIWORLD, 256, 256, 8)
    m32 = make_wide(ImplementationType.MINIWORLD, 256, 256, 8, dtype=torch.float32)
    m32.load_state_dict({k: v.float() for k, v in m16.state_dict().items()})
    x = torch.randn(1, 256, 256, 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    out = []
    for m in (m16, m32):
        xi = x.clone().requires_grad_()
        m.train()(xi).backward(dy)
        out.append([xi.grad, *(p.grad for p in m.parameters())])
    for p in m32.parameters():
        assert p.grad.dtype == torch.float32
    for a, b in zip(*out, strict=True):
        assert relative(a, b) < 1e-2
