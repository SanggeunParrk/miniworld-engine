"""H100 Transition training at D 768 (n 2, above 768 rows), D 384 / 512 (above 16384 / 12288 rows) runs the generic kernels (cuda:h100_sm90 7): the step's output and every gradient stay
inside the engine accuracy standard against a float64 reference (tolerance max(error of the bf16 PyTorch reference, 8e-3), rms relative)."""

import pytest
import torch

from foldjet_engine.kernels.transition.cuda import fused_generic_sm90a as G
from foldjet_engine.modules.transition import Transition

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 (sm_90) only")


def _rms_rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def _step(m: Transition, x: torch.Tensor, dy: torch.Tensor, dtype: torch.dtype):
    x = x.detach().to(dtype).requires_grad_(True)
    y = m(x)
    params = list(m.parameters())
    return (y, *torch.autograd.grad(y, [x, *params], dy.to(dtype)))


@pytest.mark.parametrize(
    ("d", "n", "rows"), [(768, 2, 1024), (768, 2, 2048), (768, 2, 8192), (384, 2, 36864), (512, 2, 16384), (512, 4, 36864)]
)
def test_training_generic_accuracy(d: int, n: int, rows: int) -> None:
    torch.manual_seed(0)
    assert G.prefers_generic(torch.empty(rows, d, dtype=torch.bfloat16), training=True, n=n)
    m = Transition(d, n, master_weights=True, backend="cuda").cuda()
    with torch.no_grad():  # the squeeze starts at zero: random parameters, so every gradient is non-trivial
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5 if p.dim() > 1 else 1 + 0.1 * torch.randn_like(p))
    ref = Transition(d, n, master_weights=True, backend="pytorch").cuda()
    ref.load_state_dict(m.state_dict())
    ref64 = Transition(d, n, backend="pytorch").cuda().double()
    ref64.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    x = torch.randn(1, rows, d, device="cuda").bfloat16()
    dy = torch.randn_like(x)
    got = _step(m, x, dy, torch.bfloat16)
    pt = _step(ref, x, dy, torch.bfloat16)
    ex = _step(ref64, x.double(), dy.double(), torch.float64)
    for i, (g, p, e) in enumerate(zip(got, pt, ex, strict=True)):
        print(
            f"ACC d{d} n{n} rows{rows} out{i}: engine rms {_rms_rel(g, e):.3e} max {(g.double() - e).abs().max().item():.3e}; pytorch rms {_rms_rel(p, e):.3e}"
        )
        assert e.abs().max() > 0
        assert _rms_rel(g, e) <= max(_rms_rel(p, e), 8e-3), f"output {i}: engine {_rms_rel(g, e):.3e} pytorch {_rms_rel(p, e):.3e}"
