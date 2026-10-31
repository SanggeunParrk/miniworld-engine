"""The B200 wide-width portable inference forward (kernels/transition/triton/wide.py: TMA expand + SwiGLU and squeeze + residual) against
fp64 and the split pointer kernels it replaces, on ragged row counts, both expansion factors."""
import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.transition import Transition

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="B200 (sm_100) only")


@pytest.mark.parametrize("d,n,shape", [(256, 4, (1, 47, 48)), (384, 4, (1, 33, 30)), (512, 2, (2, 20, 21)), (256, 2, (1, 300))])
def test_wide_portable_inference_matches_reference(d, n, shape, monkeypatch):
    import miniworld_engine.kernels.transition.triton.wide as wide

    settings.reset()
    settings.configure(force_portable_kernels=True)
    try:
        torch.manual_seed(0)
        m = Transition(d, n=n, implementation=ImplementationType.MINIWORLD).cuda().float().eval()
        with torch.no_grad():
            m.squeeze.weight.normal_(std=(d * n) ** -0.5)
            m.ln_in.weight.uniform_(0.5, 1.5)
            m.ln_in.bias.uniform_(-0.2, 0.2)
        ref = Transition(d, n=n, implementation=ImplementationType.PYTORCH).cuda().double()
        ref.load_state_dict(m.state_dict())
        x = torch.randn(*shape, d, device="cuda", dtype=torch.bfloat16)
        calls = []
        real = wide.transition_wide
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            monkeypatch.setattr(wide, "transition_wide", lambda *a, **k: calls.append(1) or real(*a, **k))
            y = m(x).double()
            assert calls, "the B200 portable inference did not take wide.py"
            monkeypatch.setattr(wide, "serves", lambda *a, **k: False)
            ys = m(x).double()
        yr = ref._torch_forward(x.double()) + x.double()
        e, es = ((y - yr).norm() / yr.norm()).item(), ((ys - yr).norm() / yr.norm()).item()
        assert e <= 1.1 * es + 1e-5 and e < 1e-2, (e, es)
    finally:
        settings.reset()


def test_wide_serves_gate():
    from miniworld_engine.kernels.transition.triton.wide import serves

    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    wa = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    ws = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    assert serves(x, wa, ws, False)
    assert not serves(x, wa, ws, True)                                       # training keeps the split path
    assert not serves(x.float(), wa.float(), ws.float(), False)
    x768 = torch.randn(4, 768, device="cuda", dtype=torch.bfloat16)
    assert not serves(x768, wa.new_empty(3072, 768), ws.new_empty(768, 3072), False)   # d 768: the split path (see wide.WIDTHS)
