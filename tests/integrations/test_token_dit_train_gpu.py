"""The fused token DiT TRAINING block on H100 / B200 (integrations/token_dit_train.py) against the fp32 PyTorch DiTBlock (IEEE, TF32
off): output and every input / parameter gradient no worse than the engine's own module path in the same regime -- bf16, or
fp32 (the fused path's TF32 kernels) -- (the switch is off in the second run), and the path actually taken."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
        pytest.skip("Hopper (sm_90) or Blackwell (sm_100) required")
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


def _run(m, single, cond, pair, mask, w, tf32=None):
    if tf32 is not None:
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            return _run(m, single, cond, pair, mask, w)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
    # the block's activation dtype is its narrowest parameter: the norm affines stay fp32 under .to(bf16)
    # (primitives._Fp32ParamsMixin), so the first parameter does not name it
    dtype = min((p.dtype for p in m.parameters()), key=lambda d: d.itemsize)
    ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
    out = m(*ins, mask)
    (out.float() * w).sum().backward()
    res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    res.update({n.removeprefix("_orig_mod."): p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("A", [4, 6])
@pytest.mark.parametrize("qk", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_train_block_matches_the_module_path(dtype, A, qk, masked, monkeypatch):
    torch.manual_seed(7)
    L = 256
    ref = _randomize(DiTBlock(use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    eng = eng.to(dtype)
    g = torch.Generator(device="cuda").manual_seed(8)
    single = torch.randn(A, 1, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = (torch.rand(1, L, device="cuda", generator=g) > 0.15) if masked else None
    w = torch.randn(A, 1, L, 768, device="cuda", generator=g)

    truth = _run(ref, single, cond, pair, mask, w, tf32=False)
    calls = []
    orig = token_dit_train._Block.apply
    monkeypatch.setattr(token_dit_train._Block, "apply", lambda *a: calls.append(1) or orig(*a))
    # the same regime for both: fp32 is the TF32 recipe (the fused path forces TF32 on its GEMMs; the module path follows
    # allow_tf32, so it is set for its run too)
    tf32 = True if dtype is torch.float32 else None
    fused = _run(eng, single, cond, pair, mask, w, tf32=tf32)
    assert calls, f"the {dtype} training call did not take the fused path"
    monkeypatch.setenv("MINIWORLD_TOKEN_DIT_TRAIN", "0")
    module = _run(eng, single, cond, pair, mask, w, tf32=tf32)
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    worst = []
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        worst.append((ef / max(em, 1e-6), k, ef, em))
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"
    print("worst ratios:", sorted(worst, reverse=True)[:4])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_train_block_under_torch_compile(dtype, monkeypatch):
    """torch.compile keeps the fused forward / backward as opaque ops: the compiled block takes the path and matches eager."""
    torch.manual_seed(9)
    A, L = 2, 256
    eng = _randomize(DiTBlock(use_qk_norm=True, implementation=ImplementationType.MINIWORLD).cuda()).to(dtype)
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


@pytest.mark.parametrize("A", [4, 6])
@pytest.mark.parametrize("qk", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_fp32_train_block_matches_the_module_path(qk, masked, A, monkeypatch):
    """The fp32 block on H100 (fp32 operands, TF32 attention core): output and every gradient no worse than the engine's
    own fp32 module path, against the fp32 PyTorch block (TF32 off everywhere, as torch defaults)."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the fp32 block is H100-only")
    torch.manual_seed(7)
    L = 256
    ref = _randomize(DiTBlock(use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
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
    assert calls, "the fp32 training call did not take the fused path"
    monkeypatch.setenv("MINIWORLD_TOKEN_DIT_TRAIN", "0")
    module = _run(eng, single, cond, pair, mask, w)
    worst = []
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        worst.append((ef, k, em))
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 2e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"
    print("worst fused errors:", sorted(worst, reverse=True)[:4])
