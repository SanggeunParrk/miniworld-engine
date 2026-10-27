"""The wide-width sm_90a Transition (D = 64, 256, 384, 512) matches the Triton residual path it replaces, and only takes
the calls it is built for. Same claims as ``test_transition_fused_sm90a_gpu`` (D = 128), per width.
"""

import copy

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
HOPPER = CUDA and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(
    not HOPPER, reason="the wide Transition kernels are sm_90a only"
)
WIDTHS = (64, 256, 384, 512)


@pytest.fixture(autouse=True)
def _tuned_builds_not_the_generic_small_row_preference(monkeypatch):
    """These tests target the fused / wide builds themselves; the generic kernels are preferred at small row counts by default."""
    override_setting(monkeypatch, "transition_generic_small", "0")


def _weights(d, h=None, dev="cuda"):
    h = h or 4 * d
    return (
        torch.randn(h, d, device=dev, dtype=torch.bfloat16),
        torch.randn(d, h, device=dev, dtype=torch.bfloat16),
    )


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
@pytest.mark.parametrize("d", WIDTHS)
def test_gate_rejects_everything_it_is_not_built_for(d):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide

    wa, ws = _weights(d)
    x = torch.randn(128, d, device="cuda", dtype=torch.bfloat16)
    assert wide.supported(x, wa, ws) is HOPPER
    assert not wide.supported(x.float(), wa, ws)
    # hidden must be 4 D (or 2 D where built), not 3 D
    assert not wide.supported(x, *_weights(d, 3 * d))
    assert not wide.supported(
        torch.randn(127, d, device="cuda", dtype=torch.bfloat16), wa, ws
    )
    assert not wide.supported(x.cpu(), wa.cpu(), ws.cpu())
    assert (
        wide.supported(
            torch.randn(2, 8, 8, d, device="cuda", dtype=torch.bfloat16), wa, ws
        )
        is HOPPER
    )


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_widths_without_a_build_and_the_env_switch(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide

    for d in (32, 96, 128, 192, 768):  # 128 is fused_sm90a's
        assert not wide.supported(
            torch.randn(128, d, device="cuda", dtype=torch.bfloat16), *_weights(d)
        )
    wa, ws = _weights(256)
    override_setting(monkeypatch, "transition_wide_sm90a", "0")
    assert not wide.supported(
        torch.randn(128, 256, device="cuda", dtype=torch.bfloat16), wa, ws
    )


def _build(shape, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    d = shape[-1]
    module = (
        Transition(d, n=4, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    )
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                param.normal_(
                    std=param.shape[-1] ** -0.5
                )  # the squeeze weight is zero-init
            elif param is module.ln_in.weight:
                param.copy_(
                    1 + 0.2 * torch.randn_like(param)
                )  # default gamma / beta hide LN bugs
            else:
                param.normal_(std=0.2)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return module, x, dy


def _run(module, x, dy, *, fused, fp32=False):
    from miniworld_engine import settings

    mod = copy.deepcopy(module)  # .float() is in place
    if fp32:
        mod = mod.float()
        x, dy = x.float(), dy.float()
    settings.configure(
        force_portable_kernels=not fused, transition_fused_sm90a=fused
    )
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {
        "out": y.detach().float(),
        "dx": xx.grad.detach().float(),
        **{name: p.grad.detach().float() for name, p in mod.named_parameters()},
    }


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


SHAPES = [(1, 16, 16), (2, 16, 32)]


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
@pytest.mark.parametrize("lead", SHAPES)
def test_is_no_less_accurate_than_the_triton_path(d, lead, monkeypatch):
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((*lead, d))
    reference = _run(module, x, dy, fused=False, fp32=True)
    wide = _run(module, x, dy, fused=True)
    triton = _run(module, x, dy, fused=False)
    for name in reference:
        got, base = (
            _rel(wide[name], reference[name]),
            _rel(triton[name], reference[name]),
        )
        assert got <= max(base * 1.5, 1e-6), (
            f"D{d} {name}: wide {got:.3e} vs triton {base:.3e}"
        )


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_agrees_with_the_triton_path_to_bf16(d, monkeypatch):
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((1, 16, 32, d))
    wide = _run(module, x, dy, fused=True)
    triton = _run(module, x, dy, fused=False)
    tolerances = {"out": 4e-3, "dx": 5e-3}
    for name in triton:
        assert _rel(wide[name], triton[name]) < tolerances.get(name, 6e-3), (
            f"D{d} {name}"
        )


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_the_module_actually_dispatches_to_it(d, monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    calls = []
    original = wide.transition_wide_sm90a
    monkeypatch.setattr(
        wide,
        "transition_wide_sm90a",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    module, x, dy = _build((1, 16, 16, d))
    _run(module, x, dy, fused=True)
    assert calls, "the wide path was never entered"


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_no_grad_forward_is_the_training_forward(d, monkeypatch):
    """The inference build skips the saves at run time; the output must not move by a bit."""
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    module, x, _ = _build((1, 16, 16, d))
    with torch.no_grad():
        inference = module(x)
    training = module(x.clone().requires_grad_())
    assert torch.equal(inference, training.detach())


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_replay_is_bit_identical(d, monkeypatch):
    """Deterministic everywhere except the engine LayerNorm backward used at D >= 384, whose dgamma / dbeta are summed
    with atomics -- those two are exempt there, exactly as they are on the Triton path."""
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((1, 16, 32, d))
    first, second = _run(module, x, dy, fused=True), _run(module, x, dy, fused=True)
    exempt = {"ln_in.weight", "ln_in.bias"} if d >= 384 else set()
    for name in first:
        if name not in exempt:
            assert torch.equal(first[name], second[name]), f"D{d} {name}"


@needs_hopper
def test_d64_persistent_input_buffer_reuse(monkeypatch):
    """Exercise many refills per CTA, including under Compute Sanitizer.

    The small width tests have fewer tiles than input CTAs and cannot expose
    a leader releasing the shared input slot before the other warps finish.
    This shape is the training MSA (1024 rows x 384 tokens).
    """
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    module, x, dy = _build((1, 1024, 384, 64))
    x.requires_grad_()
    params = [x, *module.parameters()]
    for p in params:
        p.grad = torch.zeros_like(p)

    def step():
        for p in params:
            p.grad.zero_()
        module(x).backward(dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    expected = [p.grad.clone() for p in params]
    for i in range(30):
        if i % 2:
            graph.replay()
        else:
            step()
        for p, reference in zip(params, expected, strict=False):
            torch.testing.assert_close(p.grad, reference, rtol=0, atol=0)


@needs_hopper
@pytest.mark.parametrize("d", [384, 512])
def test_saved_h_changes_nothing_but_the_kernel(d, monkeypatch):
    """settings.transition_wide_save_h=1 keeps the forward's h and skips its store in the gate kernel. Only dWs reads h, and
    the forward's h comes from a different GEMM tiling than the gate kernel's recomputed one (same bf16 xn and weights, another
    fp32 summation order), so dWs may move in the last bits; everything else is bit-identical."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((1, 16, 32, d))
    override_setting(monkeypatch, "transition_wide_save_h", "0")
    # D384 keeps h by default on H100; force the recompute cell through the private constant
    monkeypatch.setattr(fused_wide_sm90a, "_SAVE_H_D384", False)
    recomputed = _run(module, x, dy, fused=True)
    override_setting(monkeypatch, "transition_wide_save_h", "1")
    saved = _run(module, x, dy, fused=True)
    for name in recomputed:
        if name in ("ln_in.weight", "ln_in.bias"):  # engine LN backward: atomics
            assert _rel(saved[name], recomputed[name]) < 1e-5, name
        elif name == "squeeze.weight":
            assert _rel(saved[name], recomputed[name]) < 2e-3, name
        else:
            assert torch.equal(saved[name], recomputed[name]), name


@needs_hopper
@pytest.mark.parametrize("d", [128, 384])
@pytest.mark.parametrize("bad", ["dtype", "rows", "expansion"])
def test_shapes_beyond_the_tuned_builds_run_the_generic_kernels(d, bad, monkeypatch):
    """fp32 (TF32), a ragged row count and n = 3 have no tuned build; the generic sm_90a kernels serve them natively (no Triton fallback)."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a
    from miniworld_engine.modules import Transition

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    module = Transition(
        d,
        n=3 if bad == "expansion" else 4,
        implementation=ImplementationType.MINIWORLD,
    ).cuda()
    x = torch.randn(
        127 if bad == "rows" else 128,
        d,
        device="cuda",
        dtype=torch.float32 if bad == "dtype" else torch.bfloat16,
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)   # restored after the test: no global state left behind
    calls = []
    original = fused_generic_sm90a.transition_generic_sm90a
    monkeypatch.setattr(
        fused_generic_sm90a,
        "transition_generic_sm90a",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    with torch.no_grad():
        y = module(x)
    assert calls, "the generic kernels were never entered"
    assert torch.isfinite(y).all()


@needs_hopper
@pytest.mark.parametrize("bad", ["width", "build"])
def test_unserved_cell_runs_the_pytorch_reference(bad, monkeypatch):
    """What no native build serves runs the PyTorch reference on H100, never Triton and never an error: a width that is not a
    multiple of 64, and every build unavailable."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import (
        fused_generic_sm90a,
        fused_sm90a,
        fused_wide_sm90a,
    )
    from miniworld_engine.modules import Transition

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    d = 100 if bad == "width" else 128
    module = Transition(d, n=4, implementation=ImplementationType.MINIWORLD).cuda()
    x = torch.randn(128, d, device="cuda", dtype=torch.bfloat16)
    if bad == "build":
        for kern in (fused_sm90a, fused_wide_sm90a, fused_generic_sm90a):
            monkeypatch.setattr(kern, "available", lambda *_: False)
    with torch.no_grad():
        actual = module(x)
        expected = module._torch_forward(x) + x
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@needs_hopper
@pytest.mark.parametrize(
    ("d", "save_ab"),
    [
        (64, False),
        (128, False),
        (256, False),
        (384, False),
        (512, False),
        (384, True),
        (512, True),
    ],
    ids=[
        "D64-save0",
        "D128-save0",
        "D256-save0",
        "D384-save0",
        "D512-save0",
        "D384-save1",
        "D512-save1",
    ],
)
@pytest.mark.parametrize("n", [384, 768])
@pytest.mark.parametrize("master", [False, True])
def test_native_required_buckets_pytorch_and_live_graph(
    d, save_ab, n, master, monkeypatch
):
    """Every gradient has an independent PyTorch oracle; graph replay uses live leaves."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a
    from miniworld_engine.modules import Transition

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # Force the retained-a/b cell (the shape default retains them only for D512 at 768 * 768 rows).
    monkeypatch.setattr(
        fused_wide_sm90a,
        "save_ab_enabled",
        lambda d, rows=0: d == 768 or save_ab,
    )
    settings.configure(transition_wide_save_h=True)
    module, x, dy = _build((1, n, n, d), seed=702 + d)
    if master:
        with torch.no_grad():
            for p in module.parameters():
                if p.ndim == 2:
                    p.data = p.float()
    reference = Transition(d, implementation=ImplementationType.PYTORCH).cuda().float()
    reference.load_state_dict(module.state_dict())
    xr = x.float().requires_grad_()
    yr = reference(xr)
    yr.backward(dy.float())
    x.requires_grad_()
    y = module(x)
    y.backward(dy)
    expected = [yr, xr.grad, *(p.grad for p in reference.parameters())]
    actual = [y, x.grad, *(p.grad for p in module.parameters())]
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert got is not None
        assert want is not None
        cosine = torch.nn.functional.cosine_similarity(
            got.float().flatten(), want.float().flatten(), dim=0
        )
        assert cosine >= (0.9999 if index == 0 else 0.999), (index, float(cosine))
    del expected, actual, got, want, reference, yr, xr, y

    def step():
        x.grad = None
        for p in module.parameters():
            p.grad = None
        out = module(x)
        out.backward(dy)
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = step()
    graph.replay()
    observed = [out, x.grad, *(p.grad for p in module.parameters())]
    first = [v.clone() for v in observed]
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(first, observed, strict=True))
    with torch.no_grad():
        x.normal_()
        dy.normal_()
        for p in module.parameters():
            p.add_(0.001)
    graph.replay()
    changed = [v.clone() for v in observed]
    fresh = step()
    eager = [fresh, x.grad, *(p.grad for p in module.parameters())]
    assert all(torch.equal(a, b) for a, b in zip(changed, eager, strict=True))
    assert not torch.equal(first[0], changed[0])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="sm_90a only",
)
@pytest.mark.parametrize(
    ("rows", "d", "n"), [(16384, 64, 4), (12288, 64, 2), (16384, 128, 4)]
)
def test_backward_as_the_first_call_of_a_process(rows, d, n):
    """The autograd engine runs backward on a thread of its own. A tensor-map encode (a driver call) there needs a current context, and with
    every allocation served from the caching allocator's pool nothing had made it current: CUDA_ERROR_INVALID_CONTEXT, but only for a first
    call at 12 288 .. 20 480 rows (a smaller call earlier in the process hid it).  Fresh interpreter, one training step."""
    import subprocess
    import sys

    code = (
        "import torch\n"
        "from miniworld_engine.modules import Transition\n"
        f"m = Transition({d}, n={n}, implementation='miniworld').cuda()\n"
        f"x = torch.randn({rows}, {d}, device='cuda', dtype=torch.bfloat16).requires_grad_()\n"
        "y = m(x); torch.autograd.grad(y, (x, *m.parameters()), torch.randn_like(y)); torch.cuda.synchronize(); print('OK')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
    )
    assert out.returncode == 0, out.stderr[-600:]
    assert "OK" in out.stdout, out.stderr[-600:]


@needs_hopper
def test_d512_default_training_retains_ab_at_384_and_768_pairs(monkeypatch):
    """D512 n = 4 keeps the forward's a / b (gate backward = dh GEMM only) for the 384 x 384 and 768 x 768 pair; other row counts recompute."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw
    from miniworld_engine.modules import Transition

    assert fw.save_ab_enabled(512, 384 * 384) and fw.split_fwd_enabled(512, 384 * 384)
    assert fw.save_ab_enabled(512, 768 * 768)
    assert not fw.save_ab_enabled(512, 256 * 256) and not fw.save_ab_enabled(512, 384 * 384 + 128)
    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    n = 384
    module, x, dy = _build((1, n, n, 512), seed=1234)
    reference = Transition(512, implementation=ImplementationType.PYTORCH).cuda().float()
    reference.load_state_dict(module.state_dict())
    xr = x.float().requires_grad_()
    reference(xr).backward(dy.float())
    x.requires_grad_()
    module(x).backward(dy)
    for got, want in zip((x.grad, *(p.grad for p in module.parameters())), (xr.grad, *(p.grad for p in reference.parameters())), strict=True):
        cosine = torch.nn.functional.cosine_similarity(got.float().flatten(), want.float().flatten(), dim=0)
        assert cosine >= 0.999, float(cosine)
