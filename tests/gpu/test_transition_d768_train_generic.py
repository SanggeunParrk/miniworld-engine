"""H100 Transition D 768 (n 2) training above 768 rows runs the generic kernels (cuda:h100_sm90 7): the step's output and every gradient stay
inside the engine accuracy standard against a float64 reference (tolerance max(error of the bf16 PyTorch reference, 8e-3), rms relative)."""

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as G
from miniworld_engine.modules.transition import Transition

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 (sm_90) only")


def _rms_rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _step(m: Transition, x: torch.Tensor, dy: torch.Tensor, dtype: torch.dtype):
    x = x.detach().to(dtype).requires_grad_(True)
    y = m(x)
    params = list(m.parameters())
    return (y, *torch.autograd.grad(y, [x, *params], dy.to(dtype)))


@pytest.mark.parametrize("rows", [1024, 2048, 8192])
def test_d768_training_generic_accuracy(rows: int) -> None:
    torch.manual_seed(0)
    assert G.prefers_generic(torch.empty(rows, 768, dtype=torch.bfloat16), training=True, n=2)
    m = Transition(768, 2, master_weights=True, backend="cuda").cuda()
    ref = Transition(768, 2, master_weights=True, backend="pytorch").cuda()
    ref.load_state_dict(m.state_dict())
    ref64 = Transition(768, 2, backend="pytorch").cuda().double()
    ref64.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    x = torch.randn(1, rows, 768, device="cuda").bfloat16()
    dy = torch.randn_like(x)
    got = _step(m, x, dy, torch.bfloat16)
    pt = _step(ref, x, dy, torch.bfloat16)
    ex = _step(ref64, x.double(), dy.double(), torch.float64)
    for i, (g, p, e) in enumerate(zip(got, pt, ex, strict=True)):
        assert _rms_rel(g, e) <= max(_rms_rel(p, e), 8e-3), f"output {i}: engine {_rms_rel(g, e):.3e} pytorch {_rms_rel(p, e):.3e}"
