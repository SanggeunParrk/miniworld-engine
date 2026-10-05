"""Locally built, unmodified upstream sm80 TriMul units through the public API."""
import os

import pytest
import torch

from miniworld_engine.modules import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("TRIMUL_NATIVE_BUILD_DIR"), reason="needs sm80 native payload")]


@pytest.mark.parametrize("length", [128, 384])
@pytest.mark.parametrize(("direction", "width"), [(d, c) for d in ("outgoing", "incoming")
                                            for c in (64, 128, 256, 384)] + [("bidirectional", 64), ("bidirectional", 128)])
def test_native_module_reference_and_graph(length, direction, width):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires A100")
    torch.manual_seed(110)
    def make(impl):
        kw = {"implementation": impl, "p_drop": 0.0}
        if direction == "bidirectional":
            return BidirectionalTriangleMultiplication(width, **kw)
        return TriangleMultiplication(width, outgoing=direction == "outgoing", **kw)
    mod = make("anthropic").cuda().bfloat16().eval()
    ref = make("pytorch").cuda().float().eval()
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16)
    mask = torch.arange(length, device="cuda")[None] % 7 != 0
    with torch.no_grad():
        for name, p in mod.named_parameters():
            if p.ndim > 1:
                p.normal_(std=p.shape[-1] ** -.5)
            else:
                p.normal_(mean=1 if name.endswith("weight") else 0, std=.05)
        ref.load_state_dict(mod.state_dict())
        expected = ref(x.float(), mask)
        got = mod(x, mask)
        assert torch.isfinite(got).all()
        assert (got.float() - expected).norm() / expected.norm() < .01
        assert "built for sm_80" in mod.native_selection["unit"]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                mod(x, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = mod(x, mask)
        graph.replay()
        torch.testing.assert_close(out, got, rtol=0, atol=0)
        x.mul_(.5)
        graph.replay()
        torch.testing.assert_close(out, mod(x, mask), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match=r"grad|inference|forward"):
        mod(x, mask)
