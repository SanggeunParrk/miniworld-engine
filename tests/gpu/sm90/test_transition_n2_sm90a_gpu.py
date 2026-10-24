"""n = 2 (hidden = 2 D) Transition on sm_90a: the wide / D128 hand-CUDA builds match the Triton residual path, the
PyTorch fp32 reference bounds both, and the module really dispatches to the native kernel. The n = 4 claims are in
``test_transition_wide_sm90a_gpu`` / ``test_transition_fused_sm90a_gpu``; the widths here are the ones with an n = 2 build."""
import copy

import pytest
from settings_helpers import override_setting
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
HOPPER = CUDA and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(not HOPPER, reason="the Transition kernels are sm_90a only")
N2_WIDTHS = (64, 128, 256, 384, 512, 768)


@pytest.fixture(autouse=True)
def _tuned_builds_not_the_generic_small_row_preference(monkeypatch):
    """These tests target the fused / wide builds themselves; the generic kernels are preferred at small row counts by default."""
    override_setting(monkeypatch, "transition_generic_small", "0")


def _build(shape, n=2, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    d = shape[-1]
    module = Transition(d, n=n, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                param.normal_(std=param.shape[-1] ** -0.5)
            elif param is module.ln_in.weight:
                param.copy_(1 + 0.2 * torch.randn_like(param))
            else:
                param.normal_(std=0.2)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return module, x, dy


def _run(module, x, dy, *, fused, fp32=False):
    from miniworld_engine import settings

    mod = copy.deepcopy(module)
    if fp32:
        mod = mod.float()
        x, dy = x.float(), dy.float()
    settings.configure(force_triton_kernels=not fused, transition_fused_sm90a=fused)
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().float(), "dx": xx.grad.detach().float(),
            **{name: p.grad.detach().float() for name, p in mod.named_parameters()}}


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


@needs_hopper
@pytest.mark.parametrize("d", N2_WIDTHS)
@pytest.mark.parametrize("lead", [(1, 16, 16), (2, 16, 32)])
def test_n2_is_no_less_accurate_than_the_triton_path(d, lead, monkeypatch):
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((*lead, d))
    reference = _run(module, x, dy, fused=False, fp32=True)
    native = _run(module, x, dy, fused=True)
    triton = _run(module, x, dy, fused=False)
    for name in reference:
        got, base = _rel(native[name], reference[name]), _rel(triton[name], reference[name])
        assert got <= max(base * 1.5, 1e-6), f"D{d} n=2 {name}: native {got:.3e} vs triton {base:.3e}"


@needs_hopper
@pytest.mark.parametrize("d", N2_WIDTHS)
def test_n2_module_dispatches_to_the_native_kernel(d, monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_sm90a, fused_wide_sm90a

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    calls = []
    for mod, name in ((fused_sm90a, "transition_fused_sm90a"), (fused_wide_sm90a, "transition_wide_sm90a")):
        original = getattr(mod, name)
        monkeypatch.setattr(mod, name, lambda *a, _o=original, **k: (calls.append(1), _o(*a, **k))[1])
    module, x, dy = _build((1, 16, 16, d))
    _run(module, x, dy, fused=True)
    assert calls, "the native path was never entered"


@needs_hopper
@pytest.mark.parametrize("d", N2_WIDTHS)
def test_n2_no_grad_forward_is_the_training_forward(d, monkeypatch):
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, _ = _build((1, 16, 32, d))
    settings.configure(force_triton_kernels=False, transition_fused_sm90a=True)
    with torch.no_grad():
        inference = module(x)
    train = module(x.clone().requires_grad_())
    # The tuned inference build (LayerNorm in the wgmma fragment, Newton rsqrt) is not bit-identical to the saving training build:
    # at most one bf16 ulp apart.
    torch.testing.assert_close(inference, train, rtol=2**-7, atol=2**-7)
