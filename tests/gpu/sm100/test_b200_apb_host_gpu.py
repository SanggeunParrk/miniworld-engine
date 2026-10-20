"""AttentionPairBias on B200: the C++ host path (``apb_b200::*``) against the Python launch sequence, run-to-run determinism
under a poisoned allocator, CUDA-graph capture, and batched samples against single-sample calls. The accuracy against the fp32
module is ``test_b200_apb_gpu.py``'s (and the key split's ``test_b200_apb_ksplit_gpu.py``); here the paths are held to each other
bit for bit."""

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
    settings.configure(**{k: getattr(prev, k) for k in ("apb_b200_ksplit", "apb_b200_host")})


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
@pytest.mark.parametrize("master", [False, True], ids=["bf16", "master"])
@pytest.mark.parametrize("L", [128, 768])
def test_host_matches_python(shape, master, L):
    """The C++ launch sequences give the Python sequence's bits: inference output, training output and every gradient."""
    _, ours = _module(shape, master)
    s, p, m = _inputs(1, L, shape[1])
    sb, pb = s.bfloat16(), p.bfloat16()
    dy = torch.randn_like(sb)
    res = {}
    for host in (False, True):
        settings.configure(apb_b200_host=host)
        res[host] = (_infer(ours, sb, pb, m), _step(ours, sb, pb, m, dy))
    assert torch.equal(res[False][0], res[True][0])
    assert _equal(res[False][1], res[True][1])
    if master:
        assert all(g.dtype == torch.float32 for g in res[True][1][3:])


def _poison():
    """Fill the caching allocator's free blocks with NaN: a read of memory a step did not write shows up as NaN or a difference."""
    free = torch.cuda.mem_get_info()[0]
    junk = torch.full((min(free // 2, 4 << 30) // 4,), float("nan"), device="cuda")
    del junk


@pytest.mark.parametrize("shape", [(8, 384), (16, 384), (24, 384)], ids=lambda s: f"h{s[0]}d{s[1]}")
@pytest.mark.parametrize("host", [False, True], ids=["python", "host"])
def test_deterministic_poisoned(shape, host):
    """Five training steps and five inference calls, each after poisoning the allocator: bitwise equal."""
    settings.configure(apb_b200_host=host)
    _, ours = _module(shape, master=True)
    s, p, m = _inputs(1, 768, shape[1])
    sb, pb = s.bfloat16(), p.bfloat16()
    dy = torch.randn_like(sb)
    first_t, first_i = None, None
    for _ in range(5):
        _poison()
        t = _step(ours, sb, pb, m, dy)
        _poison()
        i = _infer(ours, sb, pb, m)
        assert all(torch.isfinite(x).all() for x in t) and torch.isfinite(i).all()
        if first_t is None:
            first_t, first_i = t, i
        else:
            assert _equal(first_t, t)
            assert torch.equal(first_i, i)


@pytest.mark.parametrize("shape", [(8, 384), (16, 384)], ids=lambda s: f"h{s[0]}d{s[1]}")
def test_cuda_graph(shape):
    """Inference and a training step captured in CUDA graphs replay the eager results bit for bit (live inputs)."""
    _, ours = _module(shape)
    s, p, m = _inputs(1, 384, shape[1])
    sb = s.bfloat16()
    pb = p.bfloat16()
    dy = torch.randn_like(sb)
    want_i = _infer(ours, sb, pb, m)
    sg = sb.clone().requires_grad_()
    pg = pb.clone().requires_grad_()

    def train():
        y = ours(sg, pg, m)
        y.backward(dy)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _infer(ours, sb, pb, m)
            for q in (sg, pg, *ours.parameters()):
                q.grad = None
            train()
    torch.cuda.current_stream().wait_stream(stream)
    for q in (sg, pg, *ours.parameters()):
        q.grad = None
    gi = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gi):
        out_i = _infer(ours, sb, pb, m)
    gi.replay()
    assert torch.equal(out_i, want_i)
    want_t = _step(ours, sb, pb, m, dy)
    for q in (sg, pg, *ours.parameters()):
        q.grad = None
    gt = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gt):
        out_t = train()
    gt.replay()
    got = [out_t.detach(), sg.grad, pg.grad, *(q.grad for q in ours.parameters())]
    assert _equal(got, want_t)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}d{s[1]}")
@pytest.mark.parametrize("B", [2, 5])
@pytest.mark.parametrize("L", [256, 768])
def test_batched_matches_single(shape, B, L):
    """B samples in one call: inference output, training output, dsingle and dpair bitwise equal to B one-sample calls (each
    sample its own pair and mask); the parameter gradients are the samples' sum, equal up to the fp32 summation order."""
    _, ours = _module(shape, master=True)
    s, p, m = _inputs(B, L, shape[1], tail=False)
    m[1, int(L * 0.6) :] = False  # samples of different lengths
    sb, pb = s.bfloat16(), p.bfloat16()
    dy = torch.randn_like(sb)
    one_i = torch.cat([_infer(ours, sb[b : b + 1], pb[b : b + 1], m[b : b + 1]) for b in range(B)])
    assert torch.equal(_infer(ours, sb, pb, m), one_i)
    got = _step(ours, sb, pb, m, dy)
    singles = [_step(ours, sb[b : b + 1], pb[b : b + 1], m[b : b + 1], dy[b : b + 1]) for b in range(B)]
    for i in range(3):  # y, dsingle, dpair
        assert torch.equal(got[i], torch.cat([t[i] for t in singles])), i
    for i in range(3, len(got)):
        want = sum(t[i] for t in singles)
        assert _rel(got[i], want) < 1e-5, (i, _rel(got[i], want))
