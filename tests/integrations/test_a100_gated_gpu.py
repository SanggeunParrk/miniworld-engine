"""The A100 (sm_80) hand-CUDA gated projections: ``fused_gate_out`` / ``ops.gated_linear`` (the registry's ``gated_linear`` rows), the one-pass gates
(``sigmoid_gate_fused``, ``gated_residual``), the triangle-multiplication stages ``tm1`` / ``tm2`` and the TriMul output gate (``trimul_inproj/cuda/sm80_gate``), forward
and backward, bf16.  Each is held to the bf16 PyTorch composition's own error against the fp32 reference; the dispatch really reaches the kernels; ``MINIWORLD_GATED_SM80=0``
and a Triton-forced engine route elsewhere; compiled and graph-captured calls equal eager."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

AMPERE = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)
needs_ampere = pytest.mark.skipif(not AMPERE, reason="the sm80 gated projections are sm_80 only")
BF = torch.bfloat16
ROWS = [(64, 64), (128, 64), (128, 128), (256, 256), (384, 384), (64, 128), (768, 768)]       # (d_hidden, d_out): the registry's gated_linear rows


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _ok(got, ref32, base, what):
    mine, yard = _rel(got, ref32), _rel(base, ref32)
    assert mine <= max(1.25 * yard, 2e-3), f"{what}: ours {mine:.3e} vs the bf16 composition {yard:.3e}"


def _leaf(t):
    return t.detach().clone().requires_grad_()


def _graph_replays(step):
    """Capture ``step`` (a call returning tensors) in a CUDA graph and check the replay equals the eager call."""
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
        assert torch.equal(a, b)


# ------------------------------------------------------------------------------------------------------------------------- the gated output projection
@needs_ampere
@pytest.mark.parametrize(("dh", "n"), ROWS)
@pytest.mark.parametrize("m", [130, 4136])
def test_fused_gate_out_forward_and_backward(dh, n, m):
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    torch.manual_seed(0)
    g = torch.randn(m, dh, device="cuda", dtype=BF).requires_grad_()
    r = torch.randn(m, dh, device="cuda", dtype=BF).requires_grad_()
    wo = (torch.randn(n, dh, device="cuda") * dh ** -0.5).to(BF).requires_grad_()
    dy = torch.randn(m, n, device="cuda", dtype=BF)
    out = sm80.fused_gate_out(g, r, wo)
    out.backward(dy)
    leaves32 = [t.detach().float().requires_grad_() for t in (g, r, wo)]
    o32 = F.linear(torch.sigmoid(leaves32[0]) * leaves32[1], leaves32[2])
    o32.backward(dy.float())
    leaves_b = [_leaf(t) for t in (g, r, wo)]
    ob = F.linear(torch.sigmoid(leaves_b[0]) * leaves_b[1], leaves_b[2])
    ob.backward(dy)
    _ok(out.detach(), o32.detach(), ob.detach(), "out")
    for name, mine, ref, base in zip(("dgate", "dvalue", "dweight"), (g.grad, r.grad, wo.grad), (t.grad for t in leaves32), (t.grad for t in leaves_b), strict=True):
        _ok(mine, ref, base, name)


@needs_ampere
@pytest.mark.parametrize(("dh", "n"), [(64, 64), (128, 128), (384, 384), (64, 128)])
@pytest.mark.parametrize("shape", ["pair", "msa", "atom", "flat"])
def test_the_registry_op_gated_linear_on_every_stream_shape(dh, n, shape):
    """``ops.gated_linear`` (what the registry's rows call) on the stream layouts: the pair [1, L, L, D], the MSA [1, 8, L, D], the atom / token [1, L, D] and [L, D]."""
    from miniworld_engine import ops

    lead = {"pair": (1, 40, 40), "msa": (1, 8, 72), "atom": (1, 1024), "flat": (300,)}[shape]
    torch.manual_seed(1)
    g = torch.randn(*lead, dh, device="cuda", dtype=BF).requires_grad_()
    v = torch.randn(*lead, dh, device="cuda", dtype=BF).requires_grad_()
    w = (torch.randn(n, dh, device="cuda") * dh ** -0.5).to(BF).requires_grad_()
    dy = torch.randn(*lead, n, device="cuda", dtype=BF)
    out = ops.gated_linear(g, v, w)
    assert out.shape == (*lead, n)
    out.backward(dy)
    l32 = [t.detach().float().requires_grad_() for t in (g, v, w)]
    o32 = F.linear(torch.sigmoid(l32[0]) * l32[1], l32[2])
    o32.backward(dy.float())
    lb = [_leaf(t) for t in (g, v, w)]
    ob = F.linear(torch.sigmoid(lb[0]) * lb[1], lb[2])
    ob.backward(dy)
    _ok(out.detach(), o32.detach(), ob.detach(), "out")
    for name, t, t32, tb in zip(("dgate", "dvalue", "dweight"), (g, v, w), l32, lb, strict=True):
        _ok(t.grad, t32.grad, tb.grad, name)


@needs_ampere
def test_the_registry_op_keeps_its_bias_and_its_unsupported_dtypes():
    from miniworld_engine import ops

    torch.manual_seed(2)
    g, v = torch.randn(2, 33, 64, device="cuda", dtype=BF), torch.randn(2, 33, 64, device="cuda", dtype=BF)
    w = torch.randn(64, 64, device="cuda", dtype=BF) * 0.1
    b = torch.randn(64, device="cuda", dtype=BF)
    got = ops.gated_linear(g, v, w, b)
    ref = F.linear(torch.sigmoid(g.float()) * v.float(), w.float(), b.float())
    assert _rel(got, ref) < 6e-3
    g32, v32, w32 = g.float(), v.float(), w.float()                              # fp32 keeps the PyTorch equation
    assert torch.equal(ops.gated_linear(g32, v32, w32), F.linear(torch.sigmoid(g32) * v32, w32))


# ----------------------------------------------------------------------------------------------------------------------------------- the one-pass gates
@needs_ampere
@pytest.mark.parametrize(("m", "d"), [(1000, 64), (4096, 384), (777, 130)])
def test_sigmoid_gate_and_gated_residual(m, d):
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    torch.manual_seed(3)
    g, o, da = (torch.randn(m, d, device="cuda", dtype=BF) for _ in range(3))
    g, o = g.requires_grad_(), o.requires_grad_()
    a = sm80.sigmoid_gate_fused(g, o)
    a.backward(da)
    g32, o32 = _leaf(g.detach().float()), _leaf(o.detach().float())
    r32 = torch.sigmoid(g32) * o32
    r32.backward(da.float())
    gb, ob = _leaf(g), _leaf(o)
    rb = torch.sigmoid(gb) * ob
    rb.backward(da)
    _ok(a.detach(), r32.detach(), rb.detach(), "a")
    _ok(g.grad, g32.grad, gb.grad, "dgate")
    _ok(o.grad, o32.grad, ob.grad, "dvalue")
    x, gg, bb = (torch.randn(m, d, device="cuda", dtype=BF).requires_grad_() for _ in range(3))
    y = sm80.gated_residual(x, gg, bb)
    y.backward(da)
    y32 = x.detach().float() + gg.detach().float() * bb.detach().float()
    _ok(y.detach(), y32, x.detach() + gg.detach() * bb.detach(), "y")
    assert torch.equal(x.grad, da)                                                        # the residual's gradient is the incoming one
    _ok(gg.grad, da.float() * bb.detach().float(), da * bb.detach(), "dgate_res")
    _ok(bb.grad, da.float() * gg.detach().float(), da * gg.detach(), "dbranch")


# ------------------------------------------------------------------------------------------------------------------------------------------- dispatch
def _spy_gate_out(monkeypatch):
    """Count the calls of the CUDA and of the Triton ``fused_gate_out`` entries (both pass through to the real kernels)."""
    from miniworld_engine.kernels.bias_only_attention.triton import (
        gate_out as triton_gate_out,
    )
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    calls = {"cuda": 0, "triton": 0}
    cuda_original, triton_original = sm80.fused_gate_out, triton_gate_out.fused_gate_out
    monkeypatch.setattr(sm80, "fused_gate_out", lambda *a: (calls.__setitem__("cuda", calls["cuda"] + 1), cuda_original(*a))[1])
    monkeypatch.setattr(triton_gate_out, "fused_gate_out", lambda *a: (calls.__setitem__("triton", calls["triton"] + 1), triton_original(*a))[1])
    return calls


def _gate_out_inputs():
    torch.manual_seed(4)
    g, v = torch.randn(1, 64, 64, 128, device="cuda", dtype=BF), torch.randn(1, 64, 64, 128, device="cuda", dtype=BF)
    return g, v, (torch.randn(128, 128, device="cuda") * 0.09).to(BF)


@needs_ampere
def test_the_registry_op_reaches_the_cuda_kernels(monkeypatch):
    from miniworld_engine import ops

    calls = _spy_gate_out(monkeypatch)
    g, v, w = _gate_out_inputs()
    got = ops.gated_linear(g, v, w)
    assert calls == {"cuda": 1, "triton": 0}
    ref = F.linear(torch.sigmoid(g.float()) * v.float(), w.float())
    assert _rel(got, ref) < 6e-3


@needs_ampere
def test_the_env_switch_routes_the_dispatcher_to_the_triton_kernels(monkeypatch):
    """The dispatcher itself is called: ``ops.gated_linear`` would first calibrate the Triton gate backend, which persists its choice in the repo's data tree."""
    from miniworld_engine.kernels.gated_projection import dispatch

    calls = _spy_gate_out(monkeypatch)
    g, v, w = _gate_out_inputs()
    got = dispatch.fused_gate_out(g, v, w)
    assert calls == {"cuda": 1, "triton": 0}
    monkeypatch.setenv("MINIWORLD_GATED_SM80", "0")
    other = dispatch.fused_gate_out(g, v, w)
    assert calls == {"cuda": 1, "triton": 1}
    assert _rel(other, got) < 1e-2
    monkeypatch.delenv("MINIWORLD_GATED_SM80")
    dispatch.fused_gate_out(g, v, w)
    assert calls == {"cuda": 2, "triton": 1}


@needs_ampere
def test_a_triton_forced_engine_routes_the_dispatcher_to_the_triton_kernels(monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.gated_projection import dispatch
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    calls = _spy_gate_out(monkeypatch)
    g, v, w = _gate_out_inputs()
    previous = settings.current().engine_backend
    settings.configure(engine_backend="triton")
    try:
        out = dispatch.fused_gate_out(g, v, w)
        assert not sm80.serves(128, 128, torch.device("cuda"), BF)
    finally:
        settings.configure(engine_backend=previous)
    assert calls == {"cuda": 0, "triton": 1}
    assert torch.isfinite(out).all()


@needs_ampere
def test_the_gate_predicate():
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    cuda = torch.device("cuda")
    for dh, n in ROWS:
        assert sm80.serves(dh, n, cuda, BF), (dh, n)
    assert not sm80.serves(128, 128, cuda, torch.float32)
    assert not sm80.serves(128, 128, cuda, torch.float16)
    assert not sm80.serves(96, 128, cuda, BF)
    assert not sm80.serves(128, 128, torch.device("cpu"), BF)
    a, b = torch.zeros(4, 64, device="cuda", dtype=BF), torch.zeros(4, 64, device="cuda", dtype=BF)
    assert sm80.serves_elementwise(a, b)
    assert not sm80.serves_elementwise(a, b.float())
    assert not sm80.serves_elementwise(a, b[:2])
    assert not sm80.serves_elementwise(a.float(), b.float())


@needs_ampere
@pytest.mark.parametrize("n", [128, 384])
def test_the_wide_outputs_take_the_split_route_and_the_narrow_ones_the_fused_gemm(n, monkeypatch):
    """From ``SPLIT_MIN_N`` output columns the one-pass gate + cuBLAS is faster than the fused GEMM: the dispatcher routes there, with the same result."""
    from miniworld_engine.kernels.gated_projection.cuda import sm80

    fused = []
    original = sm80._gate_out_fwd
    monkeypatch.setattr(sm80, "_gate_out_fwd", lambda *a: (fused.append(1), original(*a))[1])
    g, v = torch.randn(512, n, device="cuda", dtype=BF), torch.randn(512, n, device="cuda", dtype=BF)
    w = (torch.randn(n, n, device="cuda") * n ** -0.5).to(BF)
    out = sm80.fused_gate_out(g, v, w)
    assert bool(fused) == (n < sm80.SPLIT_MIN_N)
    ref = F.linear(torch.sigmoid(g.float()) * v.float(), w.float())
    _ok(out, ref, F.linear(torch.sigmoid(g) * v, w), "out")


# ------------------------------------------------------------------------------------------------------------------------- compiled and captured calls
class _Block(torch.nn.Module):
    """A gated projection and the gated residual after it: the two registry ops of one layer."""

    def __init__(self, dh, n):
        super().__init__()
        torch.manual_seed(11)
        self.w = torch.nn.Parameter((torch.randn(n, dh) * dh ** -0.5).to(BF))

    def forward(self, gate, value, x, cond):
        from miniworld_engine import ops

        return ops.gated_residual(x, cond, ops.gated_linear(gate, value, self.w))


def _block_inputs(lead, dh, n, requires_grad=True):
    torch.manual_seed(12)
    g, v = (torch.randn(*lead, dh, device="cuda", dtype=BF, requires_grad=requires_grad) for _ in range(2))
    x, c = (torch.randn(*lead, n, device="cuda", dtype=BF, requires_grad=requires_grad) for _ in range(2))
    return g, v, x, c


@needs_ampere
@pytest.mark.parametrize(("dh", "n"), [(128, 128), (384, 384)])
def test_compiled_calls_equal_eager(dh, n):
    torch._dynamo.reset()
    lead = (1, 32, 32)
    block = _Block(dh, n).cuda()
    inputs = _block_inputs(lead, dh, n)
    dy = torch.randn(*lead, n, device="cuda", dtype=BF)
    out = block(*inputs)
    grads = torch.autograd.grad(out, [*inputs, block.w], dy)
    compiled = torch.compile(block, fullgraph=True, dynamic=False)
    cinputs = [t.detach().clone().requires_grad_() for t in inputs]
    cout = compiled(*cinputs)
    cgrads = torch.autograd.grad(cout, [*cinputs, block.w], dy)
    assert torch.equal(out, cout)
    for a, b in zip(grads, cgrads, strict=True):                         # the dense backward GEMMs are plain torch ops (Inductor may order them differently): bf16 tolerance
        assert _rel(a, b) < 5e-3
    with torch.no_grad():
        assert torch.equal(torch.compile(block, fullgraph=True, dynamic=False)(*[t.detach() for t in inputs]), out)


@needs_ampere
@pytest.mark.parametrize(("dh", "n"), [(128, 128), (384, 384)])
def test_cuda_graph_capture_and_replay(dh, n):
    lead = (1, 32, 32)
    block = _Block(dh, n).cuda()
    inputs = _block_inputs(lead, dh, n)
    dy = torch.randn(*lead, n, device="cuda", dtype=BF)

    def step():
        out = block(*inputs)
        return torch.autograd.grad(out, [*inputs, block.w], dy)

    _graph_replays(step)


# ------------------------------------------------------------------------------------------------------------------------------------------ tm1 / tm2
@needs_ampere
@pytest.mark.parametrize("d", [64, 128, 256, 384])
@pytest.mark.parametrize("m", [130, 4104])
def test_tm2_forward_and_backward(d, m):
    from miniworld_engine.kernels.tm2.interface import cuda_tm2, cuda_tm2_serves

    torch.manual_seed(5)
    x, y = (torch.randn(m, d, device="cuda", dtype=BF).requires_grad_() for _ in range(2))
    wg, wo = ((torch.randn(d, d, device="cuda") * d ** -0.5).to(BF).requires_grad_() for _ in range(2))
    assert cuda_tm2_serves(x, y, wg, wo)
    dy = torch.randn(m, d, device="cuda", dtype=BF)
    out = cuda_tm2(x, y, wg, wo)
    out.backward(dy)
    t32 = [t.detach().float().requires_grad_() for t in (x, y, wg, wo)]
    o32 = torch.sigmoid(t32[0] @ t32[2]) * (t32[1] @ t32[3])
    o32.backward(dy.float())
    tb = [_leaf(t) for t in (x, y, wg, wo)]
    ob = torch.sigmoid(tb[0] @ tb[2]) * (tb[1] @ tb[3])
    ob.backward(dy)
    _ok(out.detach(), o32.detach(), ob.detach(), "out")
    for name, t, t32_, tb_ in zip(("dx", "dy", "dWg", "dWo"), (x, y, wg, wo), t32, tb, strict=True):
        _ok(t.grad, t32_.grad, tb_.grad, name)


@needs_ampere
@pytest.mark.parametrize("d", [64, 128, 256, 384])
@pytest.mark.parametrize("m", [130, 4104])
def test_tm1_forward_and_backward(d, m):
    from miniworld_engine.kernels.tm1.interface import cuda_tm1, cuda_tm1_serves

    torch.manual_seed(6)
    x = torch.randn(m, d, device="cuda", dtype=BF).requires_grad_()
    ws = [(torch.randn(d, d, device="cuda") * d ** -0.5).to(BF).requires_grad_() for _ in range(4)]       # WL, WLg, WR, WRg
    assert cuda_tm1_serves(x, *ws)
    gl, gr = (torch.randn(m, d, device="cuda", dtype=BF) for _ in range(2))
    left, right = cuda_tm1(x, *ws)
    torch.autograd.backward([left, right], [gl, gr])

    def comp(xx, w):
        wl, wlg, wr, wrg = w
        return torch.sigmoid(xx @ wlg) * (xx @ wl), torch.sigmoid(xx @ wrg) * (xx @ wr)

    x32, w32 = _leaf(x.detach().float()), [_leaf(w.detach().float()) for w in ws]
    l32, r32 = comp(x32, w32)
    torch.autograd.backward([l32, r32], [gl.float(), gr.float()])
    xb, wb = _leaf(x), [_leaf(w) for w in ws]
    lb, rb = comp(xb, wb)
    torch.autograd.backward([lb, rb], [gl, gr])
    _ok(left.detach(), l32.detach(), lb.detach(), "left")
    _ok(right.detach(), r32.detach(), rb.detach(), "right")
    _ok(x.grad, x32.grad, xb.grad, "dx")
    for name, w, w32_, wb_ in zip(("dWL", "dWLg", "dWR", "dWRg"), ws, w32, wb, strict=True):
        _ok(w.grad, w32_.grad, wb_.grad, name)


@needs_ampere
def test_tm1_and_tm2_compiled_and_captured_calls_equal_eager():
    from miniworld_engine.kernels.tm1.interface import cuda_tm1
    from miniworld_engine.kernels.tm2.interface import cuda_tm2

    torch._dynamo.reset()
    d, m = 128, 1000
    torch.manual_seed(7)
    x, y = (torch.randn(m, d, device="cuda", dtype=BF, requires_grad=True) for _ in range(2))
    ws = [(torch.randn(d, d, device="cuda") * d ** -0.5).to(BF).requires_grad_() for _ in range(4)]
    dy = torch.randn(m, d, device="cuda", dtype=BF)

    def both(xx, yy, w0, w1, w2, w3):
        left, right = cuda_tm1(xx, w0, w1, w2, w3)
        return cuda_tm2(left, right, w0, w1) + yy

    args = [x, y, *ws]
    eager = both(*args)
    egrads = torch.autograd.grad(eager, args, dy)
    cargs = [t.detach().clone().requires_grad_() for t in args]
    comp = torch.compile(both, fullgraph=True, dynamic=False)(*cargs)
    cgrads = torch.autograd.grad(comp, cargs, dy)
    assert torch.equal(eager, comp)
    for a, b in zip(egrads, cgrads, strict=True):                        # the weight / dgrad GEMMs and the dx sum are plain torch ops: bf16 tolerance
        assert _rel(a, b) < 5e-3
    del eager, comp, egrads, cgrads                                                    # the autograd nodes of these leaves would be stale (default stream) in a capture
    gargs = [t.detach().clone().requires_grad_() for t in args]
    _graph_replays(lambda: torch.autograd.grad(both(*gargs), gargs, dy))


# --------------------------------------------------------------------------------------------------------------------------------- the TriMul output gate
@needs_ampere
@pytest.mark.parametrize("n", [64, 128, 256, 384])
@pytest.mark.parametrize(("length", "planes"), [(96, 1), (128, 2)])
def test_the_trimul_output_gate_forward_and_backward(n, length, planes):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_gate as G

    m = planes * length * length
    torch.manual_seed(8)
    xn, proj, res, dy = (torch.randn(m, n, device="cuda", dtype=BF) for _ in range(4))
    wg = (torch.randn(n, n, device="cuda") * n ** -0.5).to(BF)
    ds = ((torch.rand(length, n, device="cuda") > 0.25).float() / 0.75).to(BF)
    dsr = ds.repeat(m // length, 1)                                                     # row m scales by ds[m % L]
    assert G.serves(xn, n)
    y, gate = G.gate_elem_train(xn, proj, wg, res, ds, length)
    g32 = torch.sigmoid(xn.float() @ wg.float())
    gb = torch.sigmoid(xn @ wg)
    _ok(y, res.float() + dsr.float() * (proj.float() * g32), res + dsr * (proj * gb), "y")
    _ok(gate, g32, gb, "gate")
    d_proj, dx, dwg = G.gate_elem_bwd(dy, xn, proj, gate, wg, ds, length)
    gate_in = gate.float()
    dglog32 = dy.float() * dsr.float() * proj.float() * gate_in * (1 - gate_in)
    dglogb = (dy * dsr) * proj * gate * (1 - gate)
    _ok(d_proj, dy.float() * dsr.float() * gate_in, (dy * dsr) * gate, "d_proj")
    _ok(dx, dglog32 @ wg.float().t(), dglogb @ wg.t(), "dx_gate")
    _ok(dwg, xn.float().t() @ dglog32, xn.t() @ dglogb, "dWg")
    _, dg = G.gate_elem_bwd_ew(dy, proj, xn @ wg, ds, length, from_preact=True)         # the pre-activation variant
    pre = torch.sigmoid((xn @ wg).float())
    _ok(dg, dy.float() * dsr.float() * proj.float() * pre * (1 - pre), (dy * dsr) * proj * gb * (1 - gb), "d_glogit(preact)")


@needs_ampere
def test_the_trimul_output_gate_compiled_and_captured_equal_eager():
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_gate as G

    torch._dynamo.reset()
    n, length = 128, 64
    m = length * length
    torch.manual_seed(9)
    xn, proj, res, dy = (torch.randn(m, n, device="cuda", dtype=BF) for _ in range(4))
    wg = (torch.randn(n, n, device="cuda") * n ** -0.5).to(BF)
    ds = torch.ones(length, n, device="cuda", dtype=BF)

    def step():
        y, gate = G.gate_elem_train(xn, proj, wg, res, ds, length)
        return (y, *G.gate_elem_bwd(dy, xn, proj, gate, wg, ds, length))

    eager = step()
    for a, b in zip(torch.compile(step, fullgraph=True, dynamic=False)(), eager, strict=True):
        assert torch.equal(a, b)
    _graph_replays(step)


@needs_ampere
def test_the_trimul_output_gate_serves_only_its_dtype_and_widths():
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_gate as G

    x = torch.zeros(16, 64, device="cuda", dtype=BF)
    assert G.serves(x, 64)
    assert not G.serves(x.float(), 64)
    assert not G.serves(x, 60)                                                       # a width the 16-byte vector does not divide
