"""The CTA-pair (cta_group::2) builds of the generic B200 GEMM (sm100/generic/tg_gemm.cu -DCTA2) that large-M calls take: the same
products as the single-CTA builds (bit for bit: the K order per output element is the same), and actually launched; plus GATE_AB's
TMA-loaded a, b against fp64."""
import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.transition import Transition

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="B200 (sm_100) only")


def _step(m, x, dy, dt):
    for p in m.parameters():
        p.grad = None
    xx = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dt is torch.bfloat16):
        y = m(xx)
    y.backward(dy)
    return [y.detach().float(), xx.grad.float()] + [p.grad.float() for p in m.parameters()]


@pytest.mark.parametrize("dt,shape,n", [(torch.bfloat16, (1, 96, 100), 4), (torch.float32, (1, 150, 150), 4), (torch.bfloat16, (1, 100, 90), 2)])
def test_pair_builds_match_single_cta_and_are_taken(dt, shape, n, monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a as g

    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        d = 768 if n == 4 else 1024
        torch.manual_seed(0)
        m = Transition(d, n=n, implementation=ImplementationType.MINIWORLD).cuda().float()
        with torch.no_grad():
            m.squeeze.weight.normal_(std=(d * n) ** -0.5)
        x = torch.randn(*shape, d, device="cuda", dtype=dt)
        dy = torch.randn_like(x)
        names = []
        real = g._K.gemm

        def spy(self, name, mm, nn, *a, **kw):
            names.append((name, mm, nn))
            return real(self, name, mm, nn, *a, **kw)

        monkeypatch.setattr(g._K, "gemm", spy)
        pair = _step(m, x, dy, dt)
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        taken = [nm for nm, mm, nn in names if nm in (g._BNS2_F32 if dt is torch.float32 else g._BNS2)
                 and -(-mm // 256) * (nn // (g._BNS2_F32 if dt is torch.float32 else g._BNS2)[nm]) >= nsm]
        assert taken, names
        monkeypatch.setattr(g, "CTA2", False)
        single = _step(m, x, dy, dt)
        for i, (u, v) in enumerate(zip(pair, single)):
            assert torch.equal(u, v), (i, (u - v).abs().max().item())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_gate_ab_tma_loads_match_fp64():
    torch.manual_seed(0)
    d, n = 768, 4
    m = Transition(d, n=n, implementation=ImplementationType.MINIWORLD).cuda().float()
    with torch.no_grad():
        m.squeeze.weight.normal_(std=(d * n) ** -0.5)
        m.ln_in.weight.uniform_(0.5, 1.5)
        m.ln_in.bias.uniform_(-0.2, 0.2)
    ref = Transition(d, n=n, implementation=ImplementationType.PYTORCH).cuda().double()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(1, 37, 41, d, device="cuda", dtype=torch.bfloat16)        # ragged rows: the last a, b box is zero-filled
    dy = torch.randn_like(x)
    got = _step(m, x, dy, torch.bfloat16)
    xr = x.double().requires_grad_(True)
    yr = ref._torch_forward(xr) + xr
    yr.backward(dy.double())
    want = [yr.detach(), xr.grad] + [p.grad for p in ref.parameters()]
    for i, (u, v) in enumerate(zip(got, want)):
        assert ((u.double() - v).norm() / v.norm()).item() < 1e-2, i
