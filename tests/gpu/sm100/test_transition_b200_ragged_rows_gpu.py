"""B200 Transition at a row count off the 128-row tiles (a pair track at a ragged L): the module pads zero rows for the sm_100a kernels
and cuts them back (``modules.transition.module._residual_forward``). Output and every gradient against the fp32 PyTorch module, held
to the bf16 PyTorch module's own error, as the kernels' tests do."""

import copy

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_sm100a, fused_wide_sm100a
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.transition.module import Transition

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100) required"
    ),
]


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _calls(monkeypatch):
    seen = []
    for mod, name in ((fused_sm100a, "transition_fused_sm100a"), (fused_wide_sm100a, "transition_wide_sm100a")):
        real = getattr(mod, name)

        def wrapped(*a, _real=real, **k):
            seen.append(a[0].shape[0] if a[0].ndim == 2 else a[0].numel() // a[0].shape[-1])
            return _real(*a, **k)

        monkeypatch.setattr(mod, name, wrapped)
    return seen


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("shape", [(1, 203, 203), (1, 100, 7)])
@pytest.mark.parametrize("train", [False, True])
def test_ragged_rows(monkeypatch, d, shape, train):
    torch.manual_seed(0)
    ref = Transition(d, 4, implementation=ImplementationType.PYTORCH).cuda()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ours = Transition(d, 4, implementation=ImplementationType.MINIWORLD).cuda()
    ours.load_state_dict(ref.state_dict())
    base = copy.deepcopy(ref)
    x = torch.randn(*shape, d, device="cuda")
    seen = _calls(monkeypatch)
    xb = x.bfloat16().requires_grad_(train)
    xr = x.clone().requires_grad_(train)
    xbase = x.bfloat16().requires_grad_(train)
    with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16):
        got = ours(xb)
    with torch.set_grad_enabled(train):
        want = ref(xr)
    with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16):
        bl = base(xbase)
    assert seen and seen[0] % 128 == 0, seen      # the padded kernel call ran
    assert got.shape == x.shape
    e, e0 = _rel(got, want), _rel(bl, want)
    assert e < 1.5 * e0 + 1e-3, (e, e0)
    if train:
        dy = torch.randn_like(want)
        got.backward(dy.to(got.dtype))
        want.backward(dy)
        bl.backward(dy.to(bl.dtype))
        assert _rel(xb.grad, xr.grad) < 1.5 * _rel(xbase.grad, xr.grad) + 1e-3
        for (n, p), (_, q), (_, r) in zip(ours.named_parameters(), ref.named_parameters(), base.named_parameters(), strict=True):
            assert _rel(p.grad, q.grad) < 1.5 * _rel(r.grad, q.grad) + 1e-3, n
