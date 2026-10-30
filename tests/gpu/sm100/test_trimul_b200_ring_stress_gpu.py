"""B200 TriMul D64 / D128 backward (b1s / b1g): the front CTAs hand dP and the per-token LayerNorm sums to the LN CTAs through a global ring
(generic stores -> fence.proxy.async -> flag release / acquire -> bulk copy), each front CTA reusing its 4 slots: many CUDA-graph
replays must give bitwise identical gradients (a hand-over race shows as a changed bit)."""

import pytest
import torch
from portable_helpers import randomize
from settings_helpers import policy_b200_portable_off as policy  # noqa: F401

from miniworld_engine.integrations import trimul_b200
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100) required"),
]


# D128 L256: ~6 tiles per front CTA (the 4-slot ring wraps within a launch); L768: ~59 tiles (it wraps ~15 times);
# D64 B4 (the template stack's batch, sample-major planes): L256 ~23 tiles per front CTA, L768 B1 ~51
@pytest.mark.parametrize(("width", "length", "kind", "batch"), [(128, 256, "bidir", 1), (128, 256, "out", 1), (128, 768, "bidir", 1),
                                                                (64, 256, "bidir", 4), (64, 768, "out", 1)])
def test_ring_replays_bitwise(width, length, kind, batch):
    torch.manual_seed(357)
    impl = ImplementationType.MINIWORLD
    m = (BidirectionalTriangleMultiplication(width, implementation=impl) if kind == "bidir"
         else TriangleMultiplication(width, d_hidden=width, outgoing=True, implementation=impl))
    m = randomize(m.cuda()).train().to(torch.bfloat16)
    m.p_drop = 0.0
    x = torch.randn(batch, length, length, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(batch, length, device="cuda", dtype=torch.bool)
    mask[:, length - length // 8:] = False
    dy = torch.randn_like(x)
    assert trimul_b200.serves_train(m, x, bidirectional=kind == "bidir")
    params = list(m.parameters())

    def step():
        x.grad = None
        for p in params:
            p.grad = None
        m(x, mask).backward(dy)
        return [x.grad.clone(), *(p.grad.clone() for p in params)]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    g.replay()
    ref = [t.clone() for t in out]
    bad = 0
    for _ in range(499):
        g.replay()
        bad += sum(int(not torch.equal(a, b)) for a, b in zip(out, ref, strict=True))
    torch.cuda.synchronize()
    assert bad == 0
