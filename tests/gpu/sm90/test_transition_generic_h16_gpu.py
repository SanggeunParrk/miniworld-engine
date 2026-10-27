"""fp32 activations on TF32 tensor cores, hand-CUDA path (``fused_generic_sm90a``): from 16 384 rows the (M, HID) intermediate between
the expand and squeeze GEMMs is kept in fp16 -- the TF32 mantissa in half the bytes, which are what bound both GEMMs. fp16 has less range
than fp32, so ``h`` and ``Ws`` are carried divided by device scalars derived from a bound on |a|, |b| and max|Ws| (``_h16_scales``): no
overflow whatever the weights are, scale 1 for ordinary magnitudes, no host sync.

The claim (the repo's yardstick): at most 1.5x the error of the PyTorch statements on TF32 against an fp64 run of the same inputs and
cotangent, at ordinary magnitudes and with expand weights scaled so that the pre-activations are ~1e5 (h ~ 1e10: fp16 unscaled would be inf)."""
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _grads(mod, x, dy):
    mod = copy.deepcopy(mod)
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().double(), "dx": xx.grad.double(), **{n: p.grad.double() for n, p in mod.named_parameters()}}


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


@pytest.mark.skipif(not HOPPER, reason="the generic sm_90a Transition")
@pytest.mark.parametrize("scale", [1.0, 2e4])
def test_fp16_intermediate_holds_the_tf32_yardstick_and_never_overflows(scale, monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as generic
    from miniworld_engine.modules import Transition

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.manual_seed(3)
        d, shape = 128, (1, 144, 144, 128)                       # 20 736 rows: past _H16_MIN_ROWS
        assert shape[1] * shape[2] >= generic._H16_MIN_ROWS
        module = Transition(d, n=4, implementation=ImplementationType.MINIWORLD).cuda().float()
        with torch.no_grad():
            for name, p in module.named_parameters():
                if p.ndim == 2:
                    p.normal_(std=d**-0.5)
                    if "expand" in name:
                        p.mul_(scale)
        x = torch.randn(shape, device="cuda")
        dy = torch.randn(shape, device="cuda")
        inv_sh, s, ws16 = generic._h16_scales(
            x.reshape(-1, d), module.ln_in.weight, module.ln_in.bias, module.expand_a.weight, module.expand_b.weight, module.squeeze.weight)
        assert (float(s) > 1.0) == (scale > 1), float(s)
        assert torch.isfinite(ws16).all()

        pt = Transition(d, n=4, implementation=ImplementationType.PYTORCH).cuda().float()
        pt.load_state_dict(module.state_dict())
        ref = _grads(copy.deepcopy(pt).double(), x.double(), dy.double())
        got, base = _grads(module, x, dy), _grads(pt, x, dy)
        for name, want in ref.items():
            assert torch.isfinite(got[name]).all(), name
            assert _rel(got[name], want) <= max(1.5 * _rel(base[name], want), 1e-5), (
                name, _rel(got[name], want), _rel(base[name], want))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.skipif(not HOPPER, reason="the generic sm_90a Transition")
@pytest.mark.parametrize("d,hid,scale", [(128, 256, 1.0), (128, 256, 3e4), (256, 512, 1.0), (768, 1536, 1.0), (384, 768, 1e5)])
def test_the_scale_kernel_agrees_with_the_torch_statements(d, hid, scale):
    """``h16_scales`` (memset + two kernels) returns the torch statements' values: inv_sh, s and the fp16 Ws, at ordinary and at huge weights."""
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as generic

    torch.manual_seed(0)
    x = torch.randn(8, d, device="cuda")
    gamma, beta = torch.rand(d, device="cuda") + 0.5, torch.randn(d, device="cuda")
    wa, wb = (torch.randn(hid, d, device="cuda") * d**-0.5 * scale for _ in range(2))
    ws = torch.randn(d, hid, device="cuda") * hid**-0.5 * scale
    got = generic._h16_scales(x, gamma, beta, wa, wb, ws)
    want = generic._h16_scales_torch(x, gamma, beta, wa, wb, ws)
    torch.testing.assert_close(got[0], want[0], rtol=1e-5, atol=0)
    torch.testing.assert_close(got[1], want[1], rtol=1e-5, atol=0)
    assert torch.equal(got[2], want[2]) or float((got[2].float() - want[2].float()).abs().max()) <= 1e-3 * float(want[2].float().abs().max())


@pytest.mark.skipif(not HOPPER, reason="the generic sm_90a Transition")
def test_fp32_inference_reuses_the_rounded_weights_and_never_serves_a_stale_or_foreign_copy(monkeypatch):
    """The TF32-rounded weight copies of the fp32 inference forward are made once per parameter version: reused on a repeated call, rebuilt after an
    in-place update, never handed to another module that lands on a freed weight's address, and not filled under grad mode."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import _weights
    from miniworld_engine.modules import Transition

    settings.configure(force_portable_kernels=False, transition_fused_sm90a=True)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        def make(seed):
            torch.manual_seed(seed)
            m = Transition(768, n=2, implementation=ImplementationType.MINIWORLD).cuda().float()
            with torch.no_grad():
                for p in m.parameters():
                    if p.ndim == 2:
                        p.normal_(std=768**-0.5)
            return m

        def fresh(m, x):
            _weights._ROUNDED.clear()
            with torch.no_grad():
                return m(x)

        x = torch.randn(1, 128, 768, device="cuda")
        m = make(1)
        _weights._ROUNDED.clear()
        with torch.no_grad():
            y1 = m(x)
            assert len(_weights._ROUNDED) == 1, "the first inference call did not fill the cache"
            entry = next(iter(_weights._ROUNDED.values()))[1]
            y2 = m(x)
            assert next(iter(_weights._ROUNDED.values()))[1] is entry, "a repeated call rebuilt the copies"
        assert torch.equal(y1, y2)
        with torch.no_grad():
            m.expand_a.weight.mul_(1.5)                                  # in place: the version counter moves
            y3 = m(x)
        assert not torch.equal(y3, y1)
        assert torch.equal(y3, fresh(m, x)), "a stale copy was served after an in-place update"
        del m, entry
        for seed in (2, 3, 4):                                           # new weights, quite possibly on the freed addresses
            m2 = make(seed)
            with torch.no_grad():
                y = m2(x)
            assert torch.equal(y, fresh(m2, x)), "another module was served a dead weight's copies"
            del m2
        m3 = make(5)                                                     # grad mode with a parameter requiring grad: nothing is cached
        _weights._ROUNDED.clear()
        m3(x)
        assert not _weights._ROUNDED
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
