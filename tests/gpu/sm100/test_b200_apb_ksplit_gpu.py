"""AttentionPairBias on B200: the key-split forward cores and the query-split backward dK / dV core against the fp32 module
(every chunk count, padded key tails)."""

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="Blackwell (sm_100) required",
    ),
]

SHAPES = [(8, 384), (12, 384), (16, 384), (24, 384), (16, 512)]  # (heads, d_single)


@pytest.fixture(autouse=True)
def restore_settings():
    prev = settings.current()
    yield
    settings.configure(apb_b200_ksplit=prev.apb_b200_ksplit, apb_b200_qsplit=prev.apb_b200_qsplit)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _module(shape, master=False, seed=0):
    torch.manual_seed(seed)
    m = AttentionPairBias(shape[1], 128, shape[0], implementation=ImplementationType.PYTORCH)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ref = copy.deepcopy(m).cuda()
    m.implementation = ImplementationType.MINIWORLD
    return ref, (m.cuda() if master else m.cuda().to(torch.bfloat16))


def _inputs(B, L, d, seed=1, tail=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(B, L, d, device="cuda", generator=g)
    p = torch.randn(B, L, L, 128, device="cuda", generator=g)
    m = torch.rand(B, L, device="cuda", generator=g) > 0.2
    if tail:
        m[:, int(L * 0.7) :] = False  # padded keys: whole key chunks masked
    m[:, 0] = True
    return s, p, m


def _step(mod, s, p, m, dy):
    """(y, dsingle, dpair, parameter gradients) of one training step; autocast for an fp32-master module."""
    sg = s.detach().clone().requires_grad_()
    pg = p.detach().clone().requires_grad_()
    for q in mod.parameters():
        q.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mod.to_query.weight.dtype == torch.float32):
        y = mod(sg, pg, m)
    y.backward(dy)
    return [y.detach(), sg.grad, pg.grad, *(q.grad for q in mod.parameters())]


def _infer(mod, s, p, m):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mod.to_query.weight.dtype == torch.float32):
        return mod(s, p, m)


def _equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}d{s[1]}")
@pytest.mark.parametrize("L", [208, 384, 768])
def test_ksplit_inference(shape, L):
    """Every chunk count against the fp32 module (the unsplit error + 10 %), finite with fully masked key chunks."""
    ref, ours = _module(shape)
    s, p, m = _inputs(1, L, shape[1])
    sb, pb = s.bfloat16(), p.bfloat16()
    with torch.no_grad():
        want = ref(s, p, m) - s
    nbt = -(-L // 64)
    errs = {}
    for ns in [1] + [n for n in range(2, 17) if nbt % n == 0]:
        settings.configure(apb_b200_ksplit=ns)
        got = _infer(ours, sb, pb, m)
        assert torch.isfinite(got).all(), ns
        errs[ns] = _rel(got.float() - s, want)
    assert max(errs.values()) < 1.1 * errs[1] + 1e-4, errs


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}d{s[1]}")
@pytest.mark.parametrize("L", [384, 768])
def test_ksplit_training(shape, L):
    """The split training forward: output and every gradient against fp32, as the unsplit forward (+ 10 %)."""
    ref, ours = _module(shape)
    s, p, m = _inputs(1, L, shape[1])
    dy = torch.randn_like(s)
    want = _step(ref, s, p, m, dy)
    nbt = L // 64
    worst = {}
    for ns in [1] + [n for n in (2, 3, 4, 6) if nbt % n == 0]:
        settings.configure(apb_b200_ksplit=ns)
        got = _step(ours, s.bfloat16(), p.bfloat16(), m, dy.bfloat16())
        assert all(torch.isfinite(t).all() for t in got), ns
        worst[ns] = [_rel(a.float() - (s if i == 0 else 0), b - (s if i == 0 else 0)) for i, (a, b) in enumerate(zip(got, want))]
    for ns, e in worst.items():
        for i, (x, x1) in enumerate(zip(e, worst[1])):
            assert x < 1.1 * x1 + 2e-4, (ns, i, x, x1)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}d{s[1]}")
@pytest.mark.parametrize("L", [384, 768])
def test_qsplit_training(shape, L):
    """The query-split dK / dV core: every gradient against fp32 as the unsplit core (+ 10 %); the output is untouched."""
    ref, ours = _module(shape)
    s, p, m = _inputs(1, L, shape[1])
    dy = torch.randn_like(s)
    want = _step(ref, s, p, m, dy)
    kt = L // 128
    errs, outs = {}, {}
    for nq in [1] + [n for n in range(2, kt + 1) if kt % n == 0]:
        settings.configure(apb_b200_qsplit=nq)
        got = _step(ours, s.bfloat16(), p.bfloat16(), m, dy.bfloat16())
        assert all(torch.isfinite(t).all() for t in got), nq
        outs[nq] = got[0]
        errs[nq] = [_rel(a.float(), b) for a, b in zip(got[1:], want[1:])]
    for nq, e in errs.items():
        assert torch.equal(outs[nq], outs[1])
        for i, (x, x1) in enumerate(zip(e, errs[1])):
            assert x < 1.1 * x1 + 2e-4, (nq, i, x, x1)
