"""Single-direction D512 TriMul inference (outgoing and incoming) takes the uni wide
LN/K1/contraction/K3 path and matches the Triton path within the bf16 band, with and without
a sequence mask; the op compiles to one graph and replays correctly under a CUDA graph."""
import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import trimul_h100 as H
from miniworld_engine.kernels.trimul_inproj.cuda import h100_uni_wide_inference as UW
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication.module import (
    TriangleMultiplication,
)

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _hopper():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")


def _module(d, outgoing):
    torch.manual_seed(7)
    m = TriangleMultiplication(d, outgoing=outgoing, implementation=I.MINIWORLD, p_drop=0.0).cuda()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.copy_((1.0 if n.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
                p.data = p.data.to(torch.bfloat16)
    return m.eval()


def _weights(m):
    mats = [m.to_left.weight, m.to_left_gate.weight, m.to_right.weight, m.to_right_gate.weight,
            m.to_gate.weight, m.to_out.weight]
    return [w.to(torch.bfloat16) for w in mats] + [
        p.float().contiguous() for p in (m.ln_pair.weight, m.ln_pair.bias, m.ln_out.weight, m.ln_out.bias)]


def _uni_wide(m, x, mask):
    """What the dispatch runs (``trimul_h100.update_inference`` wiring for single-direction D512)."""
    n = x.shape[1]
    pm = x.new_ones((n, n)) if mask is None else (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()
    return UW.uni_wide_inference(x, _weights(m), pm, m.outgoing)


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_uni_wide_inference_matches_triton(outgoing, masked, monkeypatch):
    _hopper()
    n, d = 384, 512
    m = _module(d, outgoing)
    assert UW.serves(m, torch.empty(1, n, n, d, device="cuda", dtype=torch.bfloat16))
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, n, device="cuda") > 0.1) if masked else None
    with torch.no_grad():
        # When the dispatch is wired, the module itself must take this path.
        if hasattr(H, "_uni_wide_inference_ok"):
            calls = []
            real = UW.uni_wide_inference
            monkeypatch.setattr(UW, "uni_wide_inference", lambda *a: calls.append(1) or real(*a))
            y_mod = m(x, mask=mask)
            assert calls, "default dispatch did not take the uni wide inference path"
        y = _uni_wide(m, x, mask)
        try:
            settings.configure(engine_backend="triton")
            ref = m(x, mask=mask)
        finally:
            settings.configure(engine_backend="auto")
    delta, want = (y - x).float(), (ref - x).float()
    assert float((delta - want).norm() / want.norm()) < 2e-2
    if hasattr(H, "_uni_wide_inference_ok"):
        torch.testing.assert_close(y_mod, y, rtol=0, atol=0)


def test_uni_wide_inference_graph_replay_and_compile():
    _hopper()
    n, d = 384, 512
    m = _module(d, True)
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = _uni_wide(m, x, None)
        # changed-input CUDA-graph replay equals a fresh call
        xs = x.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                _uni_wide(m, xs, None)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = _uni_wide(m, xs, None)
        x2 = torch.randn_like(x)
        xs.copy_(x2)
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, _uni_wide(m, x2, None), rtol=0, atol=0)
        # the opaque op traces into one graph
        compiled = torch.compile(lambda t: _uni_wide(m, t, None), fullgraph=True, dynamic=False)(x)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
