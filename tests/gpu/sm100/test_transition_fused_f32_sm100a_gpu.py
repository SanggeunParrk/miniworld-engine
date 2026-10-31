"""The one-launch fp32 (TF32) Transition forward of the generic B200 path (sm100/generic/tg_fused_f32.cu, d = 64 / 128, inference): against
fp64 and the GEMM chain it replaces (FUSED_F32 = False), on ragged row counts, both expansion factors, and bitwise under CUDA-graph replay."""
import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.transition import Transition

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="B200 (sm_100) only")


def _module(d, n):
    torch.manual_seed(0)
    m = Transition(d, n=n, implementation=ImplementationType.MINIWORLD).cuda().float().eval()
    with torch.no_grad():
        m.squeeze.weight.normal_(std=(d * n) ** -0.5)
        m.ln_in.weight.uniform_(0.5, 1.5)
        m.ln_in.bias.uniform_(-0.2, 0.2)
    return m


@pytest.mark.parametrize("d,n,shape", [(128, 4, (1, 40, 40)), (128, 2, (2, 33, 30)), (64, 4, (1, 50)), (128, 2, (1, 5)), (64, 2, (1, 200)),
                                       (128, 4, (1, 3))])
def test_fused_f32_matches_fp64_and_the_gemm_chain(d, n, shape, monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a as g

    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        m = _module(d, n)
        ref = Transition(d, n=n, implementation=ImplementationType.PYTORCH).cuda().double()
        ref.load_state_dict(m.state_dict())
        x = torch.randn(*shape, d, device="cuda")
        launched = []
        real = g._fused_f32
        monkeypatch.setattr(g, "_fused_f32", lambda *a: launched.append(real(*a)) or launched[-1])
        with torch.no_grad():
            yr = ref._torch_forward(x.double())
            yf = m(x).double() - x.double()
            assert launched and launched[-1], "the fused fp32 kernel was not taken"
            monkeypatch.setattr(g, "FUSED_F32", False)
            yc = m(x).double() - x.double()
        e_f, e_c = ((yf - yr).norm() / yr.norm()).item(), ((yc - yr).norm() / yr.norm()).item()
        assert e_f <= 1.05 * e_c + 1e-6, (e_f, e_c)
        assert e_f < 1e-3
        assert ((yf - yc).norm() / yc.norm()).item() < 1e-4      # the LayerNorm's summation order is the only difference
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_fused_f32_graph_replay_is_bitwise_the_eager_call():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        m = _module(128, 2)
        x = torch.randn(1, 77, 77, 128, device="cuda")
        with torch.no_grad():
            eager = m(x)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                out = m(x)
            gr.replay()
            torch.cuda.synchronize()
        assert torch.equal(out, eager)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
