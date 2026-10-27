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
