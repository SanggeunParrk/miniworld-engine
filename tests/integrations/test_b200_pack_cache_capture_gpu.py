"""B200 weight-pack caches under CUDA-graph capture (``kernels._capture``): a graph captured after eager warm-up calls must
repack on every replay, so a replay after an in-place weight update (an optimizer step between replays) computes with the
new weights -- as the eager call does. Before the scoping, the integrations whose cache ignored the capture recorded no
packing kernel and kept replaying the weights of capture time."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.modules import (
    AttentionPairBias,
    PairformerBlock,
    PairformerConfig,
)
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
MW = ImplementationType.MINIWORLD
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def _rand(*shape, grad=False):
    return (torch.randn(*shape, device="cuda") * 0.5).to(BF).requires_grad_(grad)


def _randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim >= 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def _token_dit(cls, grad):
    m = _randomize(cls(768, 384, 128, 16, n=2, implementation=MW).cuda().to(BF))
    return m, (_rand(4, 1, 256, 768, grad=grad), _rand(4, 1, 256, 384), _rand(1, 256, 256, 128), None)


def _triattn_wide(grad):
    blk = PairformerBlock(PairformerConfig(d_pair=64, d_hidden_tri_multi=64, d_hidden_tri_attention=32, n_head_tri_attention=4,
                                           p_drop=0.0), implementation=MW).cuda().to(BF)
    return _randomize(blk.tri_atten_starting), (_rand(1, 256, 256, 64, grad=grad), None)


def _local_dit(grad):
    m = _randomize(LocalDiTBlock(128, 128, 16, 4, n=2, cross_attention=True, implementation=MW).cuda().to(BF))
    return m, (_rand(4, 1, 256, 128, grad=grad), _rand(4, 1, 256, 128), _rand(1, 8, 32, 128, 16), None)


def _apb(grad):
    m = _randomize(AttentionPairBias(384, 128, 16, implementation=MW).cuda().to(BF))
    return m, (_rand(1, 256, 384, grad=grad), _rand(1, 256, 256, 128), None)


CASES = {
    "token DiT training": lambda: _token_dit(DiTBlock, True),
    "token DiT inference": lambda: _token_dit(DiTBlock, False),
    "bias-only DiT training": lambda: _token_dit(BiasOnlyDiTBlock, True),
    "bias-only DiT inference": lambda: _token_dit(BiasOnlyDiTBlock, False),
    "TriAttn wide training": lambda: _triattn_wide(True),
    "TriAttn wide inference": lambda: _triattn_wide(False),
    "atom (local) DiT training": lambda: _local_dit(True),
    "atom (local) DiT inference": lambda: _local_dit(False),
    "APB inference": lambda: _apb(False),
}


@pytest.mark.parametrize("name", list(CASES))
def test_replay_uses_updated_weights(name):
    torch.manual_seed(7)
    m, args = CASES[name]()
    m.eval() if "inference" in name else m.train()
    grad = "training" in name
    wrt = [a for a in args if torch.is_tensor(a) and a.requires_grad] + [p for p in m.parameters() if p.requires_grad]
    probe = None

    def step():
        nonlocal probe
        with torch.set_grad_enabled(grad):
            y = m(*args)
            if probe is None:
                probe = torch.randn(y.shape, device=y.device, dtype=torch.float32)
            if not grad:
                return (y,)
            gs = torch.autograd.grad((y.float() * probe).sum(), wrt, allow_unused=True)
            return (y.detach(), *(g for g in gs if g is not None))

    for _ in range(2):                                  # eager calls fill the eager caches
        step()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    g.replay()
    torch.cuda.synchronize()
    before = [t.clone() for t in out]
    with torch.no_grad():                               # an optimizer step: in-place, the versions move
        for p in m.parameters():
            p.add_(0.05 * p.abs().mean().clamp_min(1e-3) * torch.randn_like(p))
    g.replay()
    torch.cuda.synchronize()
    replayed = [t.clone() for t in out]
    eager = [t.clone() for t in step()]
    assert len(replayed) == len(eager)
    for r, e, b in zip(replayed, eager, before, strict=True):
        moved = relative(e, b)
        assert moved > 1e-2, f"{name}: the update does not change the result ({moved:.1e}); the test would prove nothing"
        assert relative(r, e) < 0.05 * moved, f"{name}: replay {relative(r, e):.2e} off the eager result (update moved it {moved:.2e})"


def test_capture_id_scopes():
    """None eager; one id shared by a capture's forked side stream; a new id per capture."""
    assert _capture.capture_id() is None
    ids = []
    for _ in range(2):
        g, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        x = torch.zeros(4, device="cuda")
        with torch.cuda.graph(g):
            a = _capture.capture_id()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                b = _capture.capture_id()
                x.add_(1)
            torch.cuda.current_stream().wait_stream(side)
        ids.append((a, b))
    for a, b in ids:
        assert a == b
        assert a not in (None, 0)
    assert ids[0][0] != ids[1][0]


def test_lookup_scopes_entries():
    store, built = {}, []
    build = lambda: built.append(1) or torch.ones(1, device="cuda")
    _capture.lookup(store, "k", build)
    _capture.lookup(store, "k", build)
    assert len(built) == 1                              # eager: built once
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _capture.lookup(store, "k", build)
        _capture.lookup(store, "k", build)
    assert len(built) == 2                              # the capture builds its own once, never the eager entry
    _capture.lookup(store, "k", build)
    assert len(built) == 2                              # eager again: the eager entry
