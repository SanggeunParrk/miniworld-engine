"""Single-direction native H100 training: both contractions and autograd contracts."""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.integrations import trimul_h100
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def relative(a, b):
    return float(
        (a.detach().float() - b.detach().float()).norm()
        / b.detach().float().norm().clamp_min(1e-12)
    )


def setup(length, outgoing, dropout=0.25):
    torch.manual_seed(19023)
    m = (
        TriangleMultiplication(
            128, outgoing=outgoing, implementation="miniworld", p_drop=dropout
        )
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=128**-0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    x = torch.randn(
        1, length, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.rand(1, length, device="cuda") > 0.15
    ds = (torch.rand(1, 1, length, 128, device="cuda") > dropout).bfloat16() / (
        1 - dropout
    )
    m._make_drop_row_scale = lambda pair, p: ds
    return m, x, mask, ds


@pytest.fixture(autouse=True)
def native_policy():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    if torch.cuda.get_device_properties(0).multi_processor_count != 132:
        pytest.skip("full H100 required")
    old = settings.configure(engine_backend="auto")
    torch.compiler.reset()
    yield
    settings.configure(**vars(old))


@pytest.mark.parametrize("length", [384, 768])
@pytest.mark.parametrize("outgoing", [True, False])
def test_single_output_and_all_gradients(length, outgoing):
    m, x, mask, ds = setup(length, outgoing)
    assert trimul_h100.serves_single(m, x)
    w = [
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
    ]

    def ref(x, *w):
        wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = w
        xn = F.layer_norm(x.float(), (128,), gi, bi, 1e-5).bfloat16()
        pm = (mask[:, :, None] & mask[:, None, :])[..., None]
        a = F.linear(xn, wl) * F.linear(xn, wlg).sigmoid() * pm
        b = F.linear(xn, wr) * F.linear(xn, wrg).sigmoid() * pm
        t = torch.einsum("bikd,bjkd->bijd" if outgoing else "bkid,bkjd->bijd", a, b)
        z = F.layer_norm(t.float(), (128,), go, bo, 1e-5).bfloat16()
        return x + F.linear(z, wp) * F.linear(xn, wg).sigmoid() * ds

    y = m(x, mask)
    dy = torch.randn_like(x)
    grad = torch.autograd.grad(y, (x, *w), dy)
    z = torch.compile(ref, fullgraph=True, options={"triton.cudagraphs": False})(x, *w)
    want = torch.autograd.grad(z, (x, *w), dy)
    errors = [relative(a, b) for a, b in zip((y, *grad), (z, *want), strict=True)]
    print("SINGLE", length, outgoing, errors, flush=True)
    assert errors[0] < 0.005
    assert max(errors[1:]) < 0.01


@pytest.mark.parametrize("outgoing", [True, False])
def test_single_compile_replay_live_values_and_ownership(outgoing):
    m, x, mask, ds = setup(384, outgoing)
    dy = torch.randn_like(x)
    args = (x, *m.parameters())
    fn = torch.compile(m, fullgraph=True, options={"triton.cudagraphs": False})
    # Autograd nodes and capture must share one stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):

        def run(f):
            y = f(x, mask)
            return y, torch.autograd.grad(y, args, dy)

        eager, eg = run(m)
        compiled, cg = run(fn)
        assert relative(compiled, eager) < 1e-6
        for a, b in zip(cg, eg, strict=True):
            assert relative(a, b) < 1e-5
        first = m(x, mask)
        m(x * 0.9, ~mask)
        owned = torch.autograd.grad(first, args, dy)
        for a, b in zip(owned, eg, strict=True):
            assert relative(a, b) < 1e-5
        for _ in range(2):
            run(fn)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y, g = run(fn)
        with torch.no_grad():
            x.mul_(0.93)
            m.to_left.weight.mul_(0.85)
            m.to_out.weight.mul_(0.91)
            m.ln_pair.weight[::2] = 0
            m.ln_out.weight[::2] = 0
            mask[:, ::3] = False
            ds[..., ::3] = 0
        for _ in range(2):
            graph.replay()
            z, h = run(m)
            assert relative(y, z) < 1e-6
            for a, b in zip(g, h, strict=True):
                assert relative(a, b) < 1e-5
    torch.cuda.current_stream().wait_stream(stream)


def test_single_zero_dropout_scale_residual():
    m, x, mask, ds = setup(384, True)
    ds.zero_()
    y = m(x, mask)
    dy = torch.randn_like(x)
    g = torch.autograd.grad(y, (x, *m.parameters()), dy)
    torch.testing.assert_close(y, x, atol=0, rtol=0)
    torch.testing.assert_close(g[0], dy, atol=0, rtol=0)
    for dw in g[1:]:
        assert torch.count_nonzero(dw) == 0


def test_single_layernorm_gradients_strict():
    """FP64 LN oracle with actual BF16 incoming derivatives, including zero gamma.

    Diagnostic variants only add derivative stores; production does not write
    these intermediates to HBM. BF16 GEMM differences cannot mask LN errors.
    """
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_single import Plan
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_b7 import (
        Plan as FrontBack,
    )
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_output import (
        Plan as Back,
    )

    m, x, mask, ds = setup(384, True)
    with torch.no_grad():
        m.ln_pair.weight[::3] = 0
        m.ln_out.weight[::3] = 0
    w = [
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
    ]
    pm = (mask[:, :, None] & mask[:, None, :]).bfloat16()
    dy = torch.randn_like(x)
    T._launch_module()._make_context_current(0)
    with torch.no_grad():
        plan = Plan(x, *w, pm, ds, dy)
        plan.forward()
        plan.backward()
        b = Back(x, plan.xn, plan.tri, w[5], w[4], w[8], w[9], ds, dy, debug=True)
        b.backward()
        fb = plan.front_back
        dxn = torch.empty_like(x)
        f = FrontBack(
            fb.d, dy, *fb.inputs[:3], xn=plan.xn, debug=dxn, **plan.config["b7"]
        )
        f()
    for input_ln in (True, False):
        source = x.reshape(-1, 128) if input_ln else plan.tri.reshape(128, -1).t()
        grad = (dxn if input_ln else b.y).reshape(-1, 128)
        gamma = m.ln_pair.weight if input_ln else m.ln_out.weight
        beta = m.ln_pair.bias if input_ln else m.ln_out.bias
        src = source.detach().double().requires_grad_()
        ga = gamma.detach().double().requires_grad_()
        be = beta.detach().double().requires_grad_()
        out = F.layer_norm(src, (128,), ga, be, 1e-5)
        dx, dg, db = torch.autograd.grad(out, (src, ga, be), grad.double())
        actual = (f.dgam, f.dbeta) if input_ln else (b.dgamma, b.dbeta)
        errors = (relative(actual[0], dg), relative(actual[1], db))
        print("LN_STRICT", input_ln, errors, flush=True)
        assert max(errors) < 5e-6
        expected_dx = (dx.bfloat16() + dy.reshape_as(dx)) if input_ln else dx.bfloat16()
        got_dx = f.dx.reshape_as(dx) if input_ln else b.dt.reshape(128, -1).t()
        assert relative(got_dx, expected_dx) < 1e-4
