"""ConditionedTransition on A100 (integrations/conditioned_transition_sm80.py: hand-CUDA row passes + cuBLAS, the fused atom kernels at d = dc = 128, expansion 2) against the fp32 PyTorch
module: the registry widths, bf16 and fp32 (TF32), ``forward`` (the residual output) and ``delta``, a conditioning per sample and one shared by the samples, forward and every gradient, eager and
compiled, CUDA-graph replay, the env switches.  Errors are held to the bf16 PyTorch module's own error in the same regime (fp32: to TF32 accuracy)."""

import copy
import os

import pytest
import torch

from miniworld_engine.integrations import conditioned_transition_sm80 as ct80
from miniworld_engine.modules import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

WIDTHS = [(128, 128), (768, 384), (768, 768)]               # (d_hidden, d_cond): atom, token, ESMFold2 token
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _modules(d, dc, dtype, n=2, seed=0):
    torch.manual_seed(seed)
    ref = ConditionedTransition(d, dc, n, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():                   # non-default parameters: the zero-initialised squeeze, the unit norm weight
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * (0.3 if p.ndim == 1 else 0.5 * p.shape[1] ** -0.5))
        ref.ada_ln_in.ln_cond.weight.add_(1.0)
    ours = ConditionedTransition(d, dc, n, implementation=ImplementationType.MINIWORLD)
    ours.load_state_dict(ref.state_dict())
    return ref.cuda(), ours.cuda().to(dtype), copy.deepcopy(ref).cuda().to(dtype)


def _inputs(A, L, d, dc, shared=False, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(A, 1, L, d, device="cuda", generator=g)
    c = torch.randn(1 if shared else A, 1, L, dc, device="cuda", generator=g).expand(A, 1, L, dc)
    return x, c


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("L", [50, 128, 200, 384])
@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_inference_bf16(d, dc, L, shared, delta):
    ref, ours, tb = _modules(d, dc, BF)
    x, c = _inputs(5, L, d, dc, shared)
    xb, cb = x.to(BF), c.to(BF)
    f = (lambda m, *a: m.delta(*a)) if delta else (lambda m, *a: m(*a))
    with torch.no_grad():
        assert ct80.serves(ours, xb, cb, BF, False)
        want, got, base = f(ref, x, c), f(ours, xb, cb), f(tb, xb, cb)
    assert got.dtype is BF
    assert got.shape == x.shape
    e, e0 = _rel(got, want), _rel(base, want)
    assert e < 1.1 * e0 + 1e-3, (e, e0)


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("L", [50, 200, 1024])
@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_inference_fp32_tf32(d, dc, L, shared, delta):
    """fp32 on TF32 tensor cores (at d = dc = 128 the fused AdaLN + tail kernels: ragged row counts L = 50, 5 x 1024; one conditioning shared by the samples; ``delta``) within TF32 accuracy of the fp32 module."""
    ref, ours, _ = _modules(d, dc, torch.float32)
    x, c = _inputs(5, L, d, dc, shared)
    f = (lambda m, *a: m.delta(*a)) if delta else (lambda m, *a: m(*a))
    with torch.no_grad():
        assert ct80.serves(ours, x, c, torch.float32, False)
        assert _rel(f(ours, x, c), f(ref, x, c)) < 3e-3


def _train(mod, x, c, dy, dt, delta=False):
    for p in mod.parameters():
        p.grad = None
    xx, cc = x.to(dt).detach().clone().requires_grad_(), c.to(dt).detach().clone().requires_grad_()
    y = mod.delta(xx, cc) if delta else mod(xx, cc)
    y.backward(dy.to(dt))
    return y.detach(), {"x": xx.grad, "cond": cc.grad, **{n: p.grad.clone() for n, p in mod.named_parameters()}}


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("L", [64, 128, 200])
@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_training_bf16(d, dc, L, delta):
    ref, ours, tb = _modules(d, dc, BF)
    x, c = _inputs(6, L, d, dc)
    dy = torch.randn_like(x)
    assert ct80.serves(ours, x.to(BF), c.to(BF), BF, True)
    yw, gw = _train(ref, x, c, dy, torch.float32, delta)
    yg, gg = _train(ours, x, c, dy, BF, delta)
    yb, gb = _train(tb, x, c, dy, BF, delta)
    assert _rel(yg, yw) < 1.1 * _rel(yb, yw) + 1e-3
    for n in gw:
        e, e0 = _rel(gg[n], gw[n]), _rel(gb[n], gw[n])
        assert gg[n].dtype == gb[n].dtype, n
        assert e < 1.2 * e0 + 2e-3, (n, e, e0)


@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_training_fp32_tf32(d, dc):
    ref, ours, _ = _modules(d, dc, torch.float32)
    x, c = _inputs(4, 128, d, dc)
    dy = torch.randn_like(x)
    yw, gw = _train(ref, x, c, dy, torch.float32)
    yg, gg = _train(ours, x, c, dy, torch.float32)
    assert _rel(yg, yw) < 3e-3
    for n in gw:
        assert _rel(gg[n], gw[n]) < 5e-3, n


@pytest.mark.parametrize("dtype", [BF, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("length", [128, 1024, 1152, 4096, 4608])
def test_unexpanded_shared_conditioning_training(dtype, compiled, length):
    ref, ours, _ = _modules(128, 128, dtype)
    x, c = _inputs(4, length, 128, 128)
    c = c[:1].clone()
    dy = torch.randn_like(x)
    assert ct80.serves(ours, x.to(dtype), c.to(dtype), dtype, True)
    yw, gw = _train(ref, x, c, dy, torch.float32)
    target = torch.compile(ours, fullgraph=True) if compiled else ours
    yg, gg = _train(target, x, c, dy, dtype)
    gg = {n.removeprefix("_orig_mod."): g for n, g in gg.items()}
    assert _rel(yg, yw) < (0.02 if dtype == BF else 0.003)
    for name in gw:
        assert _rel(gg[name], gw[name]) < (0.04 if dtype == BF else 0.005), name


def test_ct_a100_expansion_four_and_the_gate_conditions():
    """Any expansion and shared conditioning are supported; reject incompatible dtype, width and conditioning shapes."""
    ref4, ours4, _ = _modules(128, 128, BF, n=4)
    x, c = _inputs(3, 64, 128, 128)
    xb, cb = x.to(BF), c.to(BF)
    with torch.no_grad():
        assert ct80.serves(ours4, xb, cb, BF, False)
        assert _rel(ours4(xb, cb), ref4(x, c)) < 1.2e-2
    ours = _modules(128, 128, BF)[1]
    assert ct80.serves(ours, xb, cb, BF, True)
    assert not ct80.serves(ours, xb.half(), cb.half(), torch.float16, False)
    assert not ct80.serves(ours, xb[..., :64].contiguous(), cb, BF, False)
    assert not ct80.serves(ours, xb, cb[:, :, :32], BF, False)
    assert not ct80.serves(ours, xb.cpu(), cb.cpu(), BF, False)
    assert ct80.serves(ours, xb, cb[:1], BF, False)
    assert ct80.serves(ours, xb, cb[:1], BF, True)


def test_ct_a100_env_switches_keep_the_triton_path():
    ours = _modules(128, 128, BF)[1]
    x, c = _inputs(5, 256, 128, 128)
    xb, cb = x.to(BF), c.to(BF)
    with torch.no_grad():
        got = ours(xb, cb)
        os.environ["MINIWORLD_CONDTRANS_SM80"] = os.environ["MINIWORLD_ADALN_SM80"] = "0"
        try:
            assert not ct80.serves(ours, xb, cb, BF, False)
            base = ours(xb, cb)
        finally:
            del os.environ["MINIWORLD_CONDTRANS_SM80"], os.environ["MINIWORLD_ADALN_SM80"]
    assert _rel(got, base) < 1.5e-2


@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_compiled(d, dc):
    """Inference and a training step through torch.compile (fullgraph) match eager."""
    torch._dynamo.reset()
    _, ours, _ = _modules(d, dc, BF)
    x, c = _inputs(5, 128, d, dc)
    xb, cb = x.to(BF), c.to(BF)
    comp = torch.compile(ours, fullgraph=True)
    with torch.no_grad():
        assert _rel(comp(xb, cb), ours(xb, cb)) < 1e-6
    dy = torch.randn_like(x)
    ye, ge = _train(ours, x, c, dy, BF)
    yc, gc = _train(comp, x, c, dy, BF)
    assert _rel(yc, ye) < 1e-6
    for n in ge:
        assert _rel(gc.get(n, gc.get("_orig_mod." + n)), ge[n]) < 1e-3, n


@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_cuda_graph_replay(d, dc):
    """A captured call replays bit-identically (the packs and the extensions are resolved before capture)."""
    _, ours, _ = _modules(d, dc, BF)
    x, c = _inputs(5, 128, d, dc)
    xb, cb = x.to(BF), c.to(BF)
    with torch.no_grad():
        want = ours(xb, cb)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ours(xb, cb)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ours(xb, cb)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, want)


@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_training_is_bit_reproducible(d, dc):
    """No atomics anywhere: every column sum is a fixed-order sum of per-block partial rows, so two backward passes agree bit for bit."""
    _, ours, _ = _modules(d, dc, BF)
    x, c = _inputs(4, 128, d, dc)
    dy = torch.randn_like(x)
    _, g1 = _train(ours, x, c, dy, BF)
    _, g2 = _train(ours, x, c, dy, BF)
    for n in g1:
        assert torch.equal(g1[n], g2[n]), n


@pytest.fixture
def triton_launches(monkeypatch):
    """The names of the Triton kernels launched while the fixture is active (every ``@triton.jit`` launch, autotuned or not, goes through ``JITFunction.run``)."""
    from triton.runtime.jit import JITFunction

    calls = []
    run = JITFunction.run

    def counting_run(self, *args, **kwargs):
        calls.append(getattr(self, "__name__", repr(self)))
        return run(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", counting_run)
    return calls


@pytest.mark.parametrize("dtype", [BF, torch.float32])
@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_default_dispatch_launches_no_triton_kernel(d, dc, dtype, triton_launches):
    """The registry widths in both dtypes: ``forward`` and ``delta`` in inference (a conditioning per sample, one shared) and a training step never reach a Triton kernel."""
    _, ours, _ = _modules(d, dc, dtype)
    x, c = _inputs(3, 128, d, dc)
    shared = _inputs(3, 128, d, dc, shared=True)[1].to(dtype)
    with torch.no_grad():
        ours(x.to(dtype), c.to(dtype))
        ours(x.to(dtype), shared)
        ours.delta(x.to(dtype), shared)
    xg, cg = x.to(dtype).requires_grad_(), c.to(dtype).contiguous().requires_grad_()
    ours(xg, cg).backward(torch.randn_like(xg))
    torch.cuda.synchronize()
    assert not triton_launches, triton_launches


@pytest.mark.parametrize("dtype", [BF, torch.float32])
@pytest.mark.parametrize(("d", "dc"), WIDTHS)
def test_ct_a100_training_step_in_a_cuda_graph(d, dc, dtype):
    """A forward + backward step captured in a CUDA graph (the composition's side-stream branches fork and join inside the capture, as in the benches' training graphs) replays bit-identically to eager."""
    _, ours, _ = _modules(d, dc, dtype)
    x, c = _inputs(4, 128, d, dc)
    xg, cg = x.to(dtype).requires_grad_(), c.to(dtype).contiguous().requires_grad_()
    dy = torch.randn_like(xg)
    params = list(ours.parameters())

    def step():
        xg.grad = cg.grad = None
        for p in params:
            p.grad = None
        ours(xg, cg).backward(dy)

    step()
    torch.cuda.synchronize()
    want = [t.grad.clone() for t in (xg, cg, *params)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    torch.cuda.synchronize()
    for a, t in zip(want, (xg, cg, *params), strict=True):
        assert torch.equal(a, t.grad)


def test_ct_a100_side_stream_branches_change_nothing():
    """``MINIWORLD_ADALN_BRANCH=0`` runs the gate and the weight-gradient GEMMs on the current stream: the same bits."""
    _, ours, _ = _modules(768, 384, BF)
    x, c = _inputs(4, 128, 768, 384)
    xg, cg = x.to(BF).requires_grad_(), c.to(BF).contiguous().requires_grad_()
    dy = torch.randn_like(xg)

    def run():
        for t in (xg, cg, *ours.parameters()):
            t.grad = None
        y = ours(xg, cg)
        y.backward(dy)
        torch.cuda.synchronize()
        return [y.detach().clone()] + [t.grad.clone() for t in (xg, cg, *ours.parameters())]

    on = run()
    os.environ["MINIWORLD_ADALN_BRANCH"] = "0"
    try:
        off = run()
    finally:
        del os.environ["MINIWORLD_ADALN_BRANCH"]
    for a, b in zip(on, off, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize(("d", "dc"), [(128, 128), (768, 384)])
def test_ct_a100_inference_follows_updated_weights(autocast, d, dc):
    """The packed weights are cached per parameter version: an in-place update (an optimizer step, a checkpoint load) must reach the next call -- also when the weights are the temporary casts of fp32 master
    parameters under autocast, and at fp32 (the TF32 packs)."""
    _, ours, _ = _modules(d, dc, torch.float32 if autocast else BF)
    x, c = _inputs(3, 128, d, dc)
    dt = torch.float32 if autocast else BF
    xb, cb = x.to(dt), c.to(dt)

    def call():
        with torch.no_grad(), torch.autocast("cuda", dtype=BF, enabled=autocast):
            return ours(xb, cb)

    y1 = call()
    with torch.no_grad():
        for p in ours.parameters():
            p.mul_(1.5)
    y2 = call()
    os.environ["MINIWORLD_CONDTRANS_SM80"] = os.environ["MINIWORLD_ADALN_SM80"] = "0"
    try:
        want = call()
    finally:
        del os.environ["MINIWORLD_CONDTRANS_SM80"], os.environ["MINIWORLD_ADALN_SM80"]
    assert not torch.equal(y1, y2)
    assert _rel(y2, want) < 3e-2


@pytest.mark.parametrize("d", [128, 768])
def test_ct_a100_fp32_inference_follows_updated_weights(d):
    """The fp32 fused tail's TF32 packs follow an in-place weight update too."""
    dc = 128 if d == 128 else 384
    _, ours, _ = _modules(d, dc, torch.float32)
    x, c = _inputs(3, 128, d, dc)
    with torch.no_grad():
        y1 = ours(x, c)
        for p in ours.parameters():
            p.mul_(1.5)
        y2 = ours(x, c)
        os.environ["MINIWORLD_CONDTRANS_SM80"] = os.environ["MINIWORLD_ADALN_SM80"] = "0"
        try:
            want = ours(x, c)
        finally:
            del os.environ["MINIWORLD_CONDTRANS_SM80"], os.environ["MINIWORLD_ADALN_SM80"]
    assert not torch.equal(y1, y2)
    assert _rel(y1, y2) > 0.05                      # the update moved the output ...
    assert _rel(y2, want) < 1e-2                    # ... and the fused kernels follow it as the Triton path (both TF32, weights scaled 1.5x: a stale pack would differ by O(1))
