"""implementation="anthropic" on B200: the release's sm_80 TriMul member built for sm_100a (integrations.anthropic_trimul).

Needs TRIMUL_NATIVE_BUILD_DIR naming a payload build/ with the sm_100a member (``miniworld-engine dev build-anthropic-sm100a``)."""

import os
from pathlib import Path

import pytest
import torch

from miniworld_engine.integrations import anthropic_trimul
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100) required"),
    pytest.mark.skipif(not (Path(os.environ.get(anthropic_trimul.ENV, "/nonexistent")) / anthropic_trimul.SM100_ARCH).is_dir(),
                       reason="TRIMUL_NATIVE_BUILD_DIR with an sm_100a member required"),
]


def _module(kind, width, impl):
    if kind == "bidir":
        return BidirectionalTriangleMultiplication(width, implementation=impl)
    return TriangleMultiplication(width, d_hidden=width, outgoing=kind == "out", implementation=impl)


def _randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("length", [128, 200, 384])
@pytest.mark.parametrize(("kind", "width"), [("bidir", 64), ("bidir", 128), ("out", 64), ("in", 128), ("out", 256), ("in", 384)])
def test_matches_fp32_reference(kind, width, length):
    torch.manual_seed(401)
    ref = _randomize(_module(kind, width, ImplementationType.PYTORCH).cuda()).eval()
    m = _module(kind, width, ImplementationType.ANTHROPIC).cuda().bfloat16().eval()
    m.load_state_dict(ref.state_dict())
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.2
    with torch.no_grad():
        got = m(x, mask)
        want = ref(x.float(), mask)
    assert "sm_100a" in m.native_selection["unit"]
    assert _rel(got, want) < 0.006


def test_repeated_calls_do_not_alias():
    """The output is a fresh tensor each call (the planes / contraction workspace is cached, the result is not)."""
    torch.manual_seed(403)
    m = _randomize(_module("bidir", 128, ImplementationType.ANTHROPIC).cuda().bfloat16()).eval()
    x1 = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn_like(x1)
    with torch.no_grad():
        y1 = m(x1)
        y1c = y1.clone()
        m(x2)
    assert torch.equal(y1, y1c)


def test_cuda_graph_capture():
    torch.manual_seed(405)
    m = _randomize(_module("out", 128, ImplementationType.ANTHROPIC).cuda().bfloat16()).eval()
    x = torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = m(x).clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(x)
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize(("kind", "width", "why"), [("bidir", 256, "no tile row"), ("out", 512, "no tile row")])
def test_refuses_what_the_release_has_no_unit_for(kind, width, why):
    m = _module(kind, width, ImplementationType.ANTHROPIC).cuda().bfloat16().eval()
    x = torch.randn(1, 128, 128, width, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), pytest.raises(anthropic_trimul.PayloadUnavailable, match=why):
        m(x)


def test_refuses_training():
    m = _module("out", 128, ImplementationType.ANTHROPIC).cuda().bfloat16()
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(anthropic_trimul.PayloadUnavailable, match="forward-only"):
        m(x)
