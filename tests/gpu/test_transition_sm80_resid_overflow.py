"""sm_80 family (A100 / RTX 5090) one-kernel Transition forward: |x| above the f16 range stays finite (FoldForge AF3 5I28 n=1 on the 5090, engine 3.0.5:
``msa_transition`` input with |x| = 65536 gave one -inf, 'non-finite AF3 coordinates').  The f16 squeeze's accumulator starts at the residual x; it now
carries (x + Ws h) / 16 (``tr_fwd_sm80.cuh`` TR_RESID_SCALE, pack flag bit 30), finite up to |x| ~ 1e6."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_fwd_sm80
from miniworld_engine.modules.transition import Transition

pytestmark = [pytest.mark.gpu]  # the sm_80 family: A100 and RTX 5090 (skipped elsewhere below)


def _sm80_family() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (12, 0))


@pytest.mark.skipif(not _sm80_family(), reason="sm_80 / sm_120 one-kernel Transition forward")
@pytest.mark.parametrize(("d_hidden", "n"), [(64, 2), (64, 4), (128, 2), (128, 4)])
def test_one_kernel_forward_finite_beyond_f16_range(d_hidden: int, n: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    m = Transition(d_hidden=d_hidden, n=n).cuda().bfloat16().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.copy_((torch.randn(p.shape, device="cuda", generator=g) / p.shape[1] ** 0.5).to(p.dtype))
    m.set_backend("cuda")
    rows = fused_fwd_sm80.MIN_ROWS[(d_hidden, d_hidden * n)] * 2
    x = torch.randn(rows, d_hidden, device="cuda", generator=g) * 1000.0
    spike = torch.rand(x.shape, device="cuda", generator=g) < 0.01
    x = torch.where(spike, 65536.0 * (torch.randint(0, 2, x.shape, device="cuda", generator=g) * 2 - 1), x).bfloat16()
    with torch.no_grad():
        y = m(x[None])[0]
    assert bool(torch.isfinite(y).all())
    xf = x.double()
    mu = xf.mean(-1, keepdim=True)
    xn = (xf - mu) / torch.sqrt(((xf - mu) ** 2).mean(-1, keepdim=True) + m.ln_in.eps) * m.ln_in.weight.double() + m.ln_in.bias.double()
    h = torch.nn.functional.silu(xn @ m.expand_a.weight.double().T) * (xn @ m.expand_b.weight.double().T)
    ref = xf + h @ m.squeeze.weight.double().T
    err = (y.double() - ref).abs()
    assert float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()) < 1e-3  # bf16 output rounding is 2e-3 of the rms at these magnitudes
