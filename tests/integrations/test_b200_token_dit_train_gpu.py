"""The fused token DiT TRAINING block on B200 (integrations/token_dit_train.py) against the fp32 PyTorch DiTBlock: output and
every input / parameter gradient no worse than the engine's own module path in the same bf16 regime (the switch is off in
the second run), and the path actually taken."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5 * 0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def _run(m, single, cond, pair, mask, w):
    ins = [t.detach().clone().to(next(m.parameters()).dtype).requires_grad_() for t in (single, cond, pair)]
    out = m(*ins, mask)
    (out.float() * w).sum().backward()
    res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    res.update({n.removeprefix("_orig_mod."): p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


@pytest.mark.parametrize("qk", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_train_block_matches_the_module_path(qk, masked, monkeypatch):
    torch.manual_seed(7)
    A, L = 4, 256
    ref = _randomize(DiTBlock(use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    eng = eng.to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(8)
    single = torch.randn(A, 1, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = (torch.rand(1, L, device="cuda", generator=g) > 0.15) if masked else None
    w = torch.randn(A, 1, L, 768, device="cuda", generator=g)

    truth = _run(ref, single, cond, pair, mask, w)
    calls = []
    orig = token_dit_train._Block.apply
    monkeypatch.setattr(token_dit_train._Block, "apply", lambda *a: calls.append(1) or orig(*a))
    fused = _run(eng, single, cond, pair, mask, w)
    assert calls, "the bf16 training call did not take the fused path"
    monkeypatch.setenv("MINIWORLD_TOKEN_DIT_TRAIN", "0")
    module = _run(eng, single, cond, pair, mask, w)
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    worst = []
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        worst.append((ef / max(em, 1e-6), k, ef, em))
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"
    print("worst ratios:", sorted(worst, reverse=True)[:4])


def test_train_block_under_torch_compile(monkeypatch):
    """torch.compile keeps the fused forward / backward as opaque ops: the compiled block takes the path and matches eager."""
    torch.manual_seed(9)
    A, L = 2, 256
    eng = _randomize(DiTBlock(use_qk_norm=True, implementation=ImplementationType.MINIWORLD).cuda()).to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(10)
    single = torch.randn(A, 1, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    w = torch.randn(A, 1, L, 768, device="cuda", generator=g)
    eager = _run(eng, single, cond, pair, mask, w)
    calls = []
    orig = token_dit_train._fwd
    monkeypatch.setattr(token_dit_train, "_fwd", lambda *a: calls.append(1) or orig(*a))
    compiled = _run(torch.compile(eng), single, cond, pair, mask, w)
    assert calls, "the compiled block did not take the fused path"
    for k in eager:
        assert _rel(compiled[k], eager[k]) < 1e-3, k
