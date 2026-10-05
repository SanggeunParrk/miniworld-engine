"""The A100 (sm_80) width-generic hand-CUDA TriMul (``kernels/trimul_inproj/cuda/sm80_wide.py``, dispatched by ``integrations/trimul_sm80.py``): D = 64 / 128 /
256 / 384, one direction (outgoing, incoming) and bidirectional, bf16 and fp32 (TF32), inference and training.  Output, dz and all ten parameter gradients are
held to the bf16 PyTorch module's own error against the fp32 module (fp32 inputs: to the TF32 band of the Triton path's tests); with a token mask and the dropout row
scale; the module really dispatches to it; compiled and graph-captured calls equal eager; the env switches route elsewhere."""
import contextlib
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

AMPERE = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)
needs_ampere = pytest.mark.skipif(not AMPERE, reason="the sm80 TriMul is sm_80 only")
P_DROP = 0.25
KINDS = ["outgoing", "incoming", "bidir"]
NAMES = ["to_left.weight", "to_left_gate.weight", "to_right.weight", "to_right_gate.weight", "to_gate.weight", "to_out.weight", "ln_pair.weight",
         "ln_pair.bias", "ln_out.weight", "ln_out.bias"]


def _module(kind, d, impl, seed=1234):
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    torch.manual_seed(seed)
    if kind == "bidir":
        m = BidirectionalTriangleMultiplication(d, implementation=impl, p_drop=P_DROP)
    else:
        m = TriangleMultiplication(d, d_hidden=d, outgoing=kind == "outgoing", implementation=impl, p_drop=P_DROP)
    m = m.cuda()
    with torch.no_grad():                        # the zero-initialised gates would make most gradients exactly zero
        for name, t in m.named_parameters():
            if t.ndim >= 2:
                t.normal_(std=t.shape[-1] ** -0.5)
            elif "weight" in name:
                t.copy_(1 + 0.1 * torch.randn_like(t))
            else:
                t.normal_(std=0.05)
    return m


def _inputs(n, d, seed=90323):
    torch.manual_seed(seed)
    z = torch.randn(1, n, n, d, device="cuda")
    mask = torch.rand(1, n, device="cuda") > 0.1
    ds = (torch.rand(1, 1, n, d, device="cuda") > P_DROP).float() / (1 - P_DROP)
    dy = torch.randn(1, n, n, d, device="cuda")
    return z, mask, ds, dy


def _run(module, z, mask, ds, dy, *, train=True, dtype=torch.bfloat16, implementation=None):
    m = copy.deepcopy(module)
    if implementation is not None:
        from miniworld_engine.modules.dispatch import resolve_triangle_multiplication

        m.implementation, m._backend = implementation, resolve_triangle_multiplication(implementation)
    m = m.to(dtype) if dtype is not torch.float32 else m.float()
    m.train(train)
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)        # the same row scale on every path
    zz = z.to(dtype).clone().requires_grad_(train)
    with torch.set_grad_enabled(train):
        y = m(zz, mask)
    res = {"out": y.detach().float()}
    if train:
        y.backward(dy.to(dtype))
        res["dz"] = zz.grad.detach().float()
        params = dict(m.named_parameters())
        res.update({name: params[name].grad.detach().float() for name in NAMES})
    return res


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


SHAPES = [(64, 64), (64, 384), (256, 64), (256, 384), (384, 64), (384, 256)]      # (width, length); bf16 at D128 is the fused path's own test file


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_training_is_no_less_accurate_than_the_bf16_module(kind, d, n):
    module = _module(kind, d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(n, d)
    ref = _run(module, z, mask, ds, dy, dtype=torch.float32, implementation=ImplementationType.PYTORCH)
    bf16 = _run(module, z, mask, ds, dy, implementation=ImplementationType.PYTORCH)
    got = _run(module, z, mask, ds, dy)
    for name in ref:
        mine, base = _rel(got[name], ref[name]), _rel(bf16[name], ref[name])
        assert mine <= max(1.25 * base, 2e-3), f"{name}: sm80 wide {mine:.3e} vs bf16 module {base:.3e}"


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_inference_is_no_less_accurate_than_the_bf16_module(kind, d, n):
    module = _module(kind, d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(n, d)
    ref = _run(module, z, mask, ds, dy, train=False, dtype=torch.float32, implementation=ImplementationType.PYTORCH)
    bf16 = _run(module, z, mask, ds, dy, train=False, implementation=ImplementationType.PYTORCH)
    got = _run(module, z, mask, ds, dy, train=False)
    mine, base = _rel(got["out"], ref["out"]), _rel(bf16["out"], ref["out"])
    assert mine <= max(1.25 * base, 2e-3), f"sm80 wide {mine:.3e} vs bf16 module {base:.3e}"


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("d", [64, 256])
def test_the_modules_dispatch_to_it_and_the_switches_turn_it_off(kind, d, monkeypatch):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    calls = []
    original = sm80_wide.trimul
    monkeypatch.setattr(sm80_wide, "trimul", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    module = _module(kind, d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(64, d)
    _run(module, z, mask, ds, dy)
    _run(module, z, mask, ds, dy, train=False)
    assert len(calls) == 2, "the wide path was not entered for both training and inference"
    for switch in ("MINIWORLD_TRIMUL_SM80", "MINIWORLD_TRIMUL_SM80_WIDE"):
        monkeypatch.setenv(switch, "0")
        got = _run(module, z, mask, ds, dy, train=False)
        assert len(calls) == 2, f"{switch}=0 still entered the wide path"
        assert torch.isfinite(got["out"]).all()                      # the Triton path serves
        monkeypatch.delenv(switch)


@needs_ampere
def test_the_gate_takes_what_it_is_built_for_and_nothing_else():
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    z = torch.randn(1, 64, 64, 256, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 64, device="cuda", dtype=torch.bool)
    assert sm80_wide.supports(z, 256, mask)
    assert sm80_wide.supports(z, 256, hs=512)                                                     # bidirectional: hidden 2 D
    assert not sm80_wide.supports(z, 256, hs=128)                                                 # d_hidden != d_pair
    assert not sm80_wide.supports(z.half(), 256)                                                  # a dtype it does not take
    assert sm80_wide.supports(torch.randn(1, 64, 64, 128, device="cuda", dtype=torch.bfloat16), 128)   # D128 is in the width set (the integration tries the fused D128 kernels first)
    assert not sm80_wide.supports(torch.randn(1, 64, 64, 96, device="cuda", dtype=torch.bfloat16), 96)      # an unregistered width
    assert not sm80_wide.supports(torch.randn(1, 40, 40, 256, device="cuda", dtype=torch.bfloat16), 256)     # L % 16
    assert not sm80_wide.supports(z, 256, mask.float())
    assert not sm80_wide.supports(z.cpu(), 256)
    assert not sm80_wide.supports(z.transpose(1, 2), 256)                                          # non-contiguous
    assert not sm80_wide.supports(z, 256, torch.ones(2, 64, device="cuda", dtype=torch.bool))    # a mask for another batch


@needs_ampere
def test_no_mask_matches_an_all_true_mask():
    module = _module("outgoing", 256, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(64, 256)
    a = _run(module, z, None, ds, dy)
    b = _run(module, z, torch.ones_like(mask), ds, dy)
    assert all(torch.equal(a[k], b[k]) for k in a)


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
def test_the_residual_is_the_kernels_own(kind):
    """A freshly initialised TriMul has a zero output projection: its update is exactly zero, so the output is the input bit for bit and its gradient exactly one."""
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    torch.manual_seed(7)
    if kind == "bidir":
        m = BidirectionalTriangleMultiplication(256, implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    else:
        m = TriangleMultiplication(256, d_hidden=256, outgoing=kind == "outgoing", implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    m = m.cuda().bfloat16().train()
    z = torch.randn(1, 64, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = m(z, torch.rand(1, 64, device="cuda") > 0.1)
    assert torch.equal(out, z)
    out.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))


@needs_ampere
def test_fp32_parameters_get_fp32_gradients_equal_to_the_bf16_ones():
    """Mixed precision keeps fp32 master parameters under bf16 activations: the gradients come back in the parameters' dtype, equal to those of the same values
    held in bf16."""
    module = _module("outgoing", 256, ImplementationType.MINIWORLD).bfloat16()          # bf16-exact values, so the fp32 copy below holds the same numbers
    z, mask, ds, dy = _inputs(64, 256)
    want = _run(module, z, mask, ds, dy)
    m = copy.deepcopy(module).float().train()                   # the same values in fp32; activations stay bf16
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)
    zz = z.bfloat16().clone().requires_grad_(True)
    m(zz, mask).backward(dy.bfloat16())
    for name, p in m.named_parameters():
        assert p.grad.dtype is torch.float32, name
        if p.numel() > 64:                                      # the kernels' fp32 accumulators, never rounded to bf16 (the cast is outside autograd)
            assert not torch.equal(p.grad, p.grad.bfloat16().float()), name
        assert _rel(p.grad.float(), want[name]) < 4e-3, name    # the bf16 parameters' gradient is this one rounded once (~2^-9 per element)


@needs_ampere
def test_replay_is_bit_identical():
    module = _module("bidir", 256, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(128, 256)
    a = _run(module, z, mask, ds, dy)
    b = _run(module, z, mask, ds, dy)
    assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])]


@needs_ampere
@pytest.mark.parametrize("kind", ["outgoing", "bidir"])
def test_a_batch_runs_plane_by_plane_and_equals_the_planes_run_alone(kind):
    n, d = 64, 256
    module = _module(kind, d, ImplementationType.MINIWORLD)
    torch.manual_seed(5)
    z = torch.randn(2, n, n, d, device="cuda")
    mask = torch.rand(2, n, device="cuda") > 0.1
    ds = (torch.rand(2, 1, n, d, device="cuda") > P_DROP).float() / (1 - P_DROP)
    dy = torch.randn(2, n, n, d, device="cuda")
    both = _run(module, z, mask, ds, dy)
    singles = [_run(module, z[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1]) for i in range(2)]
    assert torch.equal(both["out"], torch.cat([s["out"] for s in singles]))
    assert torch.equal(both["dz"], torch.cat([s["dz"] for s in singles]))
    for name in both:
        if name not in ("out", "dz"):
            assert _rel(both[name], singles[0][name] + singles[1][name]) < 1e-2, name


@needs_ampere
def test_inference_with_a_live_dropout_scale_equals_the_training_forward():
    """No-grad with a live row scale (``train()`` under ``no_grad``) takes the saving-free forward with the scale: it equals the grad-enabled forward."""
    module = _module("outgoing", 256, ImplementationType.MINIWORLD)
    z, mask, ds, _ = _inputs(128, 256)
    m = copy.deepcopy(module).bfloat16().train()
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)
    zb = z.bfloat16()
    with torch.no_grad():
        a = m(zb, mask)
    b = m(zb.clone().requires_grad_(True), mask)
    assert torch.equal(a, b.detach())


@needs_ampere
@pytest.mark.parametrize("d", [64, 256])        # D64 runs the statistics-in-kernel variants
@pytest.mark.parametrize("kind", ["outgoing", "bidir"])
def test_compiled_calls_equal_eager(kind, d):
    """``torch.compile(fullgraph=True)``: inference and a training step equal eager bit for bit (the opaque ops' fakes: shapes, strides, no aliasing)."""
    torch._dynamo.reset()
    n = 128
    module = _module(kind, d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(n, d)
    eager_inf = _run(module, z, mask, ds, dy, train=False)
    eager = _run(module, z, mask, ds, dy)

    m = copy.deepcopy(module).bfloat16().eval()
    comp = torch.compile(m, fullgraph=True, dynamic=False)
    with torch.no_grad():
        assert torch.equal(comp(z.bfloat16(), mask).float(), eager_inf["out"])

    m = copy.deepcopy(module).bfloat16().train()
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)
    zz = z.bfloat16().clone().requires_grad_(True)
    torch.compile(m, fullgraph=True, dynamic=False)(zz, mask).backward(dy.bfloat16())
    assert torch.equal(zz.grad.float(), eager["dz"])
    params = dict(m.named_parameters())
    for name in NAMES:
        assert _rel(params[name].grad.float(), eager[name]) < 1e-3, name


@needs_ampere
@pytest.mark.parametrize("d", [64, 384])
@pytest.mark.parametrize("grad", [False, True])
def test_cuda_graph_capture_and_replay(grad, d):
    """The module path launches nothing host-dependent: a captured call (inference, or a full training step) replays bit-identically."""
    n = 128
    module = _module("outgoing", d, ImplementationType.MINIWORLD).bfloat16()
    module.train(grad)
    z = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    mask = torch.rand(1, n, device="cuda") > 0.1
    dy = torch.randn_like(z)
    ds = ((torch.rand(1, 1, n, d, device="cuda") > P_DROP).float() / (1 - P_DROP)).bfloat16()
    module._make_drop_row_scale = lambda pair, p: ds
    params = [z, *module.parameters()] if grad else []

    def step():
        with torch.set_grad_enabled(grad):
            y = module(z, mask)
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
        assert torch.equal(a, b)


@needs_ampere
@pytest.mark.parametrize("d", [64, 256, 384])
@pytest.mark.parametrize("bidir", [False, True])
def test_one_launch_pack_matches_its_torch_definition(d, bidir):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide as W

    hs = 2 * d if bidir else d
    torch.manual_seed(5)
    ws = [torch.randn(hs, d, device="cuda").bfloat16() for _ in range(4)]
    wg = torch.randn(d, d, device="cuda").bfloat16()
    wo = torch.randn(d, hs, device="cuda").bfloat16()
    ln = [1 + 0.1 * torch.randn(d, device="cuda"), 0.1 * torch.randn(d, device="cuda"), 1 + 0.1 * torch.randn(hs, device="cuda"), 0.1 * torch.randn(hs, device="cuda")]
    for front in ("row_major", "in_out"):                          # the bidirectional D128 module stores its front matrices [in, out]
        w = ws if front == "row_major" else [t.t().contiguous().t() for t in ws]
        got = W._pack(*w, wg, wo, *ln)
        want = W._pack_reference(*w, wg, wo, *ln)
        for key in ("w1", "wo", "wg"):
            assert torch.equal(got[key], want[key]), key
        for key in ("vs", "vb", "so", "eo", "sg", "eg"):
            torch.testing.assert_close(got[key], want[key], rtol=1e-5, atol=1e-5, msg=key)


# ------------------------------------------------------------------------------------------------------------------------------------ fp32 (TF32 tensor cores)
SHAPES32 = [(64, 64), (128, 96), (256, 64), (384, 64), (128, 384)]            # fp32 at D128 is the wide path's too: the fused D128 kernels are bf16 only


@contextlib.contextmanager
def _tf32(on):
    """The global TF32 switch of the PyTorch composition (the sm80 kernels turn TF32 on for their own GEMMs whatever it says)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("d", "n"), SHAPES32)
def test_fp32_training_is_no_less_accurate_than_the_tf32_pytorch_module(kind, d, n):
    """Output, dz and the ten parameter gradients against the fp32 module with TF32 off; the yardstick is the same PyTorch module with TF32 on (what a TF32 run of
    the plain composition gets)."""
    module = _module(kind, d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(n, d)
    with _tf32(False):
        ref = _run(module, z, mask, ds, dy, dtype=torch.float32, implementation=ImplementationType.PYTORCH)
    with _tf32(True):
        base = _run(module, z, mask, ds, dy, dtype=torch.float32, implementation=ImplementationType.PYTORCH)
    got = _run(module, z, mask, ds, dy, dtype=torch.float32)
    for name in ref:
        mine, yard = _rel(got[name], ref[name]), _rel(base[name], ref[name])
        assert mine <= max(1.25 * yard, 2e-3), f"{name}: sm80 wide fp32 {mine:.3e} vs TF32 PyTorch {yard:.3e}"


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("d", "n"), [(64, 64), (128, 128), (384, 96)])
def test_fp32_inference_is_no_less_accurate_than_the_tf32_pytorch_module(kind, d, n):
    module = _module(kind, d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(n, d)
    with _tf32(False):
        ref = _run(module, z, mask, ds, dy, train=False, dtype=torch.float32, implementation=ImplementationType.PYTORCH)
    with _tf32(True):
        base = _run(module, z, mask, ds, dy, train=False, dtype=torch.float32, implementation=ImplementationType.PYTORCH)
    got = _run(module, z, mask, ds, dy, train=False, dtype=torch.float32)
    mine, yard = _rel(got["out"], ref["out"]), _rel(base["out"], ref["out"])
    assert mine <= max(1.25 * yard, 2e-3), f"sm80 wide fp32 {mine:.3e} vs TF32 PyTorch {yard:.3e}"


@needs_ampere
@pytest.mark.parametrize("d", [128, 256])
def test_fp32_modules_dispatch_to_the_wide_path_and_the_switches_turn_it_off(d, monkeypatch):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80, sm80_wide

    wide, fused = [], []
    original = sm80_wide.trimul
    monkeypatch.setattr(sm80_wide, "trimul", lambda *a, **k: (wide.append(1), original(*a, **k))[1])
    monkeypatch.setattr(sm80, "trimul", lambda *a, **k: fused.append(1))
    module = _module("outgoing", d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(64, d)
    _run(module, z, mask, ds, dy, dtype=torch.float32)
    _run(module, z, mask, ds, dy, train=False, dtype=torch.float32)
    assert len(wide) == 2, "fp32 must run the wide path"
    assert not fused, "the fused D128 kernels are bf16 only"
    for switch in ("MINIWORLD_TRIMUL_SM80", "MINIWORLD_TRIMUL_SM80_WIDE"):
        monkeypatch.setenv(switch, "0")
        got = _run(module, z, mask, ds, dy, train=False, dtype=torch.float32)
        assert len(wide) == 2, f"{switch}=0 still entered the wide path"
        assert torch.isfinite(got["out"]).all()
        monkeypatch.delenv(switch)


@needs_ampere
def test_the_fp32_gate_takes_fp32():
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    z = torch.randn(1, 64, 64, 256, device="cuda")
    mask = torch.ones(1, 64, device="cuda", dtype=torch.bool)
    assert sm80_wide.supports(z, 256, mask)
    assert sm80_wide.supports(torch.randn(1, 64, 64, 128, device="cuda"), 128)               # D128 fp32: the wide path


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
def test_fp32_residual_is_the_kernels_own(kind):
    """The zero-initialised output projection: the update is exactly zero in fp32 too (input bit for bit, gradient exactly one)."""
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    torch.manual_seed(7)
    if kind == "bidir":
        m = BidirectionalTriangleMultiplication(128, implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    else:
        m = TriangleMultiplication(128, d_hidden=128, outgoing=kind == "outgoing", implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    m = m.cuda().float().train()
    z = torch.randn(1, 64, 64, 128, device="cuda", requires_grad=True)
    out = m(z, torch.rand(1, 64, device="cuda") > 0.1)
    assert torch.equal(out, z)
    out.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))


@needs_ampere
@pytest.mark.parametrize("kind", ["outgoing", "bidir"])
def test_fp32_compiled_calls_equal_eager(kind):
    torch._dynamo.reset()
    d, n = 128, 128
    module = _module(kind, d, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(n, d)
    eager_inf = _run(module, z, mask, ds, dy, train=False, dtype=torch.float32)
    eager = _run(module, z, mask, ds, dy, dtype=torch.float32)

    m = copy.deepcopy(module).float().eval()
    with torch.no_grad():
        assert torch.equal(torch.compile(m, fullgraph=True, dynamic=False)(z, mask).float(), eager_inf["out"])

    m = copy.deepcopy(module).float().train()
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)
    zz = z.clone().requires_grad_(True)
    torch.compile(m, fullgraph=True, dynamic=False)(zz, mask).backward(dy)
    assert torch.equal(zz.grad.float(), eager["dz"])
    params = dict(m.named_parameters())
    for name in NAMES:
        assert _rel(params[name].grad.float(), eager[name]) < 1e-3, name


@needs_ampere
@pytest.mark.parametrize("grad", [False, True])
def test_fp32_cuda_graph_capture_and_replay(grad):
    d, n = 256, 128
    module = _module("outgoing", d, ImplementationType.MINIWORLD).float()
    module.train(grad)
    z = torch.randn(1, n, n, d, device="cuda", requires_grad=grad)
    mask = torch.rand(1, n, device="cuda") > 0.1
    dy = torch.randn_like(z)
    ds = ((torch.rand(1, 1, n, d, device="cuda") > P_DROP).float() / (1 - P_DROP))
    module._make_drop_row_scale = lambda pair, p: ds
    params = [z, *module.parameters()] if grad else []

    def step():
        with torch.set_grad_enabled(grad):
            y = module(z, mask)
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
        assert torch.equal(a, b)


@needs_ampere
@pytest.mark.parametrize("d", [64, 128, 384])
@pytest.mark.parametrize("bidir", [False, True])
def test_fp32_pack_matches_its_torch_definition(d, bidir):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide as W

    hs = 2 * d if bidir else d
    torch.manual_seed(5)
    ws = [torch.randn(hs, d, device="cuda") for _ in range(4)]
    wg = torch.randn(d, d, device="cuda")
    wo = torch.randn(d, hs, device="cuda")
    ln = [1 + 0.1 * torch.randn(d, device="cuda"), 0.1 * torch.randn(d, device="cuda"), 1 + 0.1 * torch.randn(hs, device="cuda"), 0.1 * torch.randn(hs, device="cuda")]
    got = W._pack(*ws, wg, wo, *ln)
    want = W._pack_reference(*ws, wg, wo, *ln)
    for key in ("w1", "wo", "wg"):
        assert got[key].dtype is torch.float32, key
        assert torch.equal(got[key], want[key]), key                # the TF32-rounded folded weights, bit for bit
    for key in ("vs", "vb", "so", "eo", "sg", "eg"):
        torch.testing.assert_close(got[key], want[key], rtol=1e-5, atol=1e-5, msg=key)


@needs_ampere
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_plane_that_starts_off_a_16_byte_boundary_is_copied_not_misread(dtype):
    """The kernels move 16-byte granules: a contiguous view into a larger buffer that starts 2 (or 4) bytes off a boundary gives the aligned plane's result."""
    module = _module("outgoing", 256, ImplementationType.MINIWORLD).to(dtype).eval()
    z, mask, _, _ = _inputs(64, 256)
    aligned = z.to(dtype).contiguous()
    buf = torch.empty(aligned.numel() + 8, device="cuda", dtype=dtype)
    view = buf[1:1 + aligned.numel()].view_as(aligned)
    view.copy_(aligned)
    assert view.data_ptr() % 16 != 0
    with torch.no_grad():
        assert torch.equal(module(view, mask), module(aligned, mask))


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
def test_the_wide_kernels_also_serve_d128_bf16_when_the_fused_d128_path_is_out_of_the_way(kind, monkeypatch):
    """D128 bf16 runs the fused D128 kernels by default; with those declined the wide kernels take it (it is also their statistics-in-kernel case): no less accurate than the bf16 module."""
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80, sm80_wide

    calls = []
    original = sm80_wide.trimul
    monkeypatch.setattr(sm80_wide, "trimul", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    monkeypatch.setattr(sm80, "available", lambda *a, **k: False)
    monkeypatch.setattr(sm80, "supports", lambda *a, **k: False)
    module = _module(kind, 128, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(128, 128)
    ref = _run(module, z, mask, ds, dy, dtype=torch.float32, implementation=ImplementationType.PYTORCH)
    bf16 = _run(module, z, mask, ds, dy, implementation=ImplementationType.PYTORCH)
    got = _run(module, z, mask, ds, dy)
    assert calls, "the wide path was not entered"
    for name in ref:
        mine, base = _rel(got[name], ref[name]), _rel(bf16[name], ref[name])
        assert mine <= max(1.25 * base, 2e-3), f"{name}: sm80 wide at D128 {mine:.3e} vs bf16 module {base:.3e}"
