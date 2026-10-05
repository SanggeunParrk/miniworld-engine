"""Bidirectional wide TriMul inference (D256/384/512) takes the fused wide K1/K3 path and
matches the Triton path within the bf16 band, with and without a sequence mask."""
import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_inference as W
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _module(d, cls=BidirectionalTriangleMultiplication):
    torch.manual_seed(7)
    m = cls(d, implementation=I.MINIWORLD, p_drop=0.0).cuda()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.copy_((1.0 if n.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
                p.data = p.data.to(torch.bfloat16)
    return m.eval()


@pytest.mark.parametrize("d", [256, 384, 512])
@pytest.mark.parametrize("masked", [False, True])
def test_wide_inference_matches_triton(d, masked, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    n = 384
    m = _module(d)
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, n, device="cuda") > 0.1) if masked else None
    calls = []
    real = W.wide_inference
    monkeypatch.setattr(W, "wide_inference", lambda *a: calls.append(1) or real(*a))
    with torch.no_grad():
        y = m(x, mask=mask)
        assert calls, "default dispatch did not take the wide K1/K3 inference path"
        try:
            settings.configure(engine_backend="triton")
            ref = m(x, mask=mask)
        finally:
            settings.configure(engine_backend="auto")
    delta, want = (y - x).float(), (ref - x).float()
    assert float((delta - want).norm() / want.norm()) < 2e-2


def test_wide_inference_compiles_to_one_graph():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    n, d = 384, 256
    m = _module(d)
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = m(x)
        compiled = torch.compile(m, fullgraph=True, dynamic=False)(x)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)


@pytest.mark.parametrize("d", [64, 128, 256, 384, 512])
@pytest.mark.parametrize("master", [False, True])
def test_cuda_live_pack_strides_and_graph(d, master, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_pack as P

    monkeypatch.setenv("MINIWORLD_TRIMUL_H100_CUDA_AUX", "1")
    dtype = torch.float32 if master else torch.bfloat16
    weights = [torch.randn(d, d, device="cuda", dtype=dtype).t() for _ in range(4)]
    packed = torch.empty(4 * d, d, device="cuda", dtype=torch.bfloat16)

    def expected():
        gate = torch.cat((weights[1], weights[3])).reshape(-1, 32, d)
        value = torch.cat((weights[0], weights[2])).reshape(-1, 32, d)
        return torch.stack((gate, value), 1).reshape_as(packed).bfloat16()

    with torch.no_grad():
        P.pack_into_cuda(packed, *weights)
        assert torch.equal(packed, expected())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            P.pack_into_cuda(packed, *weights)
        graph.replay()
        first = packed.clone()
        graph.replay()
        assert torch.equal(first, packed)
        weights[0].add_(0.125)
        graph.replay()
        assert torch.equal(packed, expected())
        assert not torch.equal(first, packed)


@pytest.mark.parametrize("d", [64, 128, 256, 384, 512])
@pytest.mark.parametrize("n", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_cuda_inference_required_buckets_pytorch_and_live_graph(d, n, master, bidirectional, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    from miniworld_engine.integrations import trimul_h100 as H
    from miniworld_engine.modules.triangle_multiplication.module import TriangleMultiplication

    if d in (64, 128):
        from miniworld_engine.kernels.trimul_inproj.cuda import _h100_pack as P
        monkeypatch.delenv("MINIWORLD_TRIMUL_H100_CUDA_AUX", raising=False)
        def reject_triton(*args):
            raise AssertionError("promoted inference used Triton weight packing")
        monkeypatch.setattr(P, "pack_into_triton", reject_triton)
    else:
        monkeypatch.setenv("MINIWORLD_TRIMUL_H100_CUDA_AUX", "1")
    cls = BidirectionalTriangleMultiplication if bidirectional else TriangleMultiplication
    m = _module(d, cls)
    if master:
        m.float()
    ref = cls(d, implementation=I.PYTORCH, p_drop=0.0).cuda().float().eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, n, device="cuda") > 0.2
    with torch.no_grad():
        assert H.serves_inference(m, x, bidirectional=bidirectional)
        want = ref(x.float(), mask=mask)
        for _ in range(2):
            m(x, mask=mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = m(x, mask=mask)
        graph.replay()
        first = out.clone()
        graph.replay()
        assert torch.equal(first, out)
        cos = torch.nn.functional.cosine_similarity(first.float().flatten(), want.flatten(), dim=0)
        assert cos.item() >= 0.9999
        x.add_(0.0625)
        mask.logical_not_()
        m.to_left.weight.add_(0.125)
        graph.replay()
        assert torch.equal(out, m(x, mask=mask))
        assert not torch.equal(first, out)
        mask.zero_()
        graph.replay()
        ref.load_state_dict(m.state_dict())
        want = ref(x.float(), mask=mask)
        assert torch.nn.functional.cosine_similarity(out.float().flatten(), want.flatten(), dim=0).item() >= 0.9999
