"""AlphaFold 3's atom block (parallel K/V AdaLN, slid key window) on the H100 mma kernels against the fp64 PyTorch module, inference."""

import pytest
import torch

from miniworld_engine.integrations import local_dit_h100 as h100
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

CASES = [(2, 512, 512), (2, 512, 301), (1, 384, 140), (3, 256, 256), (1, 300, 300), (1, 256, 100), (2, 128, 77)]  # (A, N padded, real atoms)
OPTIONS = [("parallel", "pad"), ("chained", "slide"), ("parallel", "slide")]


def _pair(impl, **options):
    torch.manual_seed(0)
    blk = LocalDiTBlock(128, 128, 16, 4, n=2, cross_attention=True, implementation=impl, **options)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.1)
    return blk


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize(("kv_norm", "key_window"), OPTIONS)
@pytest.mark.parametrize(("a", "n", "real"), CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_matches_the_pytorch_module(kv_norm, key_window, a, n, real, dtype):
    blk = _pair(ImplementationType.TRITON, kv_norm=kv_norm, key_window=key_window).cuda().float()
    ref = _pair(ImplementationType.PYTORCH, kv_norm=kv_norm, key_window=key_window).cuda().double()
    ref.load_state_dict(blk.state_dict())
    g = torch.Generator(device="cuda").manual_seed(1)
    s = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    c = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    z = torch.randn(1, (n + 31) // 32, 32, 128, 16, device="cuda", generator=g)
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    with torch.no_grad():
        expected = ref(s.double(), c.double(), z.double(), mask)
        ins = (s.to(dtype), c.to(dtype), z.to(dtype))
        if dtype == torch.bfloat16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assert h100.matches(blk, *ins, mask)
                actual = blk(*ins, mask)
        else:
            assert h100.matches(blk, *ins, mask)
            actual = blk(*ins, mask)
    tol = 2e-3 if dtype == torch.float32 else 1.5e-2
    assert _rel(actual[:, :, :real], expected[:, :, :real]) < tol


def test_a_gradient_is_not_served():
    blk = _pair(ImplementationType.TRITON, kv_norm="parallel", key_window="slide").cuda().float()
    s = torch.randn(1, 1, 256, 128, device="cuda")
    z = torch.randn(1, 8, 32, 128, 16, device="cuda")
    mask = torch.zeros(1, 256, dtype=torch.bool, device="cuda")
    mask[:, :200] = True
    with torch.no_grad():
        assert h100.matches(blk, s, s, z, mask)
    assert not h100.matches(blk, s, s, z, mask)  # the parameters record a gradient


@pytest.mark.parametrize("real", [256, 200, 100])
def test_the_slid_window_is_captured_in_a_cuda_graph(real):
    """The real atom count stays on the device: the block runs inside a graph capture and replays with another mask."""
    blk = _pair(ImplementationType.TRITON, kv_norm="parallel", key_window="slide").cuda().float().requires_grad_(False)
    g = torch.Generator(device="cuda").manual_seed(3)
    s = torch.randn(2, 1, 256, 128, device="cuda", generator=g)
    c = torch.randn(2, 1, 256, 128, device="cuda", generator=g)
    z = torch.randn(1, 8, 32, 128, 16, device="cuda", generator=g)
    mask = torch.zeros(1, 256, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    with torch.no_grad():
        expected = blk(s, c, z, mask)
        for _ in range(2):  # warm the kernels and the caches outside the capture
            blk(s, c, z, mask)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = blk(s, c, z, mask)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        mask.zero_()
        mask[:, :150] = True  # a different real atom count through the same captured graph
        graph.replay()
        again = blk(s, c, z, mask)
        torch.cuda.synchronize()
    assert torch.equal(out, again)
