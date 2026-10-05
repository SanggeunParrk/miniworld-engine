"""The fused token DiT TRAINING block on A100 (integrations/token_dit_train.py: cuBLAS GEMMs, the CUDA row kernels and the sm_80 attention core
``kernels/augmented_attention/cuda/sm80``) against the fp32 PyTorch DiTBlock (IEEE, TF32 off): the output and every input / parameter gradient no
worse than the engine's own module path in the same regime (the switch is off in the second run), and the path actually taken."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere A100 (sm_80) required")
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
    dtype = min((p.dtype for p in m.parameters()), key=lambda d: d.itemsize)
    ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
    out = m(*ins, mask)
    (out.float() * w).sum().backward()
    res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    res.update({n.removeprefix("_orig_mod."): p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


def _inputs(a, length, masked, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(a, 1, length, 768, device="cuda", generator=g)
    cond = torch.randn(a, 1, length, 384, device="cuda", generator=g)
    pair = torch.randn(1, length, length, 128, device="cuda", generator=g)
    mask = (torch.rand(1, length, device="cuda", generator=g) > 0.15) if masked else None
    w = torch.randn(a, 1, length, 768, device="cuda", generator=g)
    return single, cond, pair, mask, w


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(("a", "length"), [(4, 256), (6, 128), (8, 384)])
def test_train_block_matches_the_module_path(a, length, masked, monkeypatch):
    torch.manual_seed(7)
    ref = _randomize(DiTBlock(implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    eng = eng.to(torch.bfloat16)
    single, cond, pair, mask, w = _inputs(a, length, masked, 8)
    truth = _run(ref, single, cond, pair, mask, w, tf32=False)
    calls = []
    orig = token_dit_train._Block.apply
    monkeypatch.setattr(token_dit_train._Block, "apply", lambda *args: calls.append(1) or orig(*args))
    fused = _run(eng, single, cond, pair, mask, w)
    assert calls, "the bf16 training call did not take the fused path"
    monkeypatch.setenv("MINIWORLD_TOKEN_DIT_TRAIN", "0")
    module = _run(eng, single, cond, pair, mask, w)
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"


def test_train_block_under_torch_compile(monkeypatch):
    """torch.compile keeps the fused forward / backward as opaque ops: the compiled block takes the path and matches eager."""
    torch.manual_seed(9)
    eng = _randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda()).to(torch.bfloat16)
    single, cond, pair, mask, w = _inputs(2, 256, True, 10)
    eager = _run(eng, single, cond, pair, mask, w)
    calls = []
    orig = token_dit_train._fwd
    monkeypatch.setattr(token_dit_train, "_fwd", lambda *args: calls.append(1) or orig(*args))
    compiled = _run(torch.compile(eng), single, cond, pair, mask, w)
    assert calls, "the compiled block did not take the fused path"
    for k in eager:
        assert _rel(compiled[k], eager[k]) < 1e-3, k


def test_a_replay_of_the_step_is_bit_identical():
    """The attention backward sums its bias-gradient partials in a fixed order and the weight gradients are cuBLAS GEMMs: the same inputs give the same
    bits -- except the two conditioning-LayerNorm weight gradients, which the shared row kernel ``unfold_lnw_k`` (B200's, also built for sm_80)
    accumulates with fp32 atomics: they agree to fp32 rounding."""
    torch.manual_seed(11)
    eng = _randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda()).to(torch.bfloat16)
    single, cond, pair, mask, w = _inputs(4, 256, True, 12)
    first = _run(eng, single, cond, pair, mask, w)
    second = _run(eng, single, cond, pair, mask, w)
    atomic = {"attention.ada_ln_in.ln_cond.weight", "transition.ada_ln_in.ln_cond.weight"}
    for k in first:
        if k in atomic:
            assert _rel(first[k], second[k]) < 1e-5, k
        else:
            assert torch.equal(first[k], second[k]), k


@pytest.mark.parametrize("compiled", [False, True])
def test_shared_conditioning_reduces_sample_gradients(compiled):
    torch.manual_seed(31)
    eng = _randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda()).bfloat16()
    single, cond, pair, mask, w = _inputs(4, 128, True, 32)
    shared = cond[:1].clone()
    assert token_dit_train.serves(eng, single.bfloat16(), shared.bfloat16(), pair.bfloat16(), mask)
    expanded = _run(eng, single, shared.expand_as(cond).contiguous(), pair, mask, w)
    model = torch.compile(eng, fullgraph=True) if compiled else eng
    actual = _run(model, single, shared, pair, mask, w)
    for name in expanded:
        expected = expanded[name]
        if name == "dcond":
            expected = expected.sum(0, keepdim=True).bfloat16().float()
        assert _rel(actual[name], expected) < 1e-3, name


def test_the_gate_serves_only_what_the_block_was_built_for(monkeypatch):
    """Everything the A100 block cannot take keeps the module path: fp32 operands (the TF32 kernels are B200), QK-norm, another head layout, an odd
    sample count, a length off the 128-row tiles, no autograd, the switches."""
    eng = _randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda()).to(torch.bfloat16)
    x = torch.randn(4, 1, 256, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    c = torch.randn(4, 1, 256, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16)
    assert token_dit_train.serves(eng, x, c, p, None)
    assert not token_dit_train.serves(eng, x.detach().float().requires_grad_(), c.float(), p.float(), None)
    assert not token_dit_train.serves(eng, x[:3], c[:3], p, None)
    assert not token_dit_train.serves(eng, x[:, :, :200], c[:, :, :200], p[:, :200, :200], None)
    assert not token_dit_train.serves(eng, x, c[:, :, :, :256], p, None)
    with torch.no_grad():
        assert not token_dit_train.serves(eng, x, c, p, None), "no autograd"
    qk = DiTBlock(use_qk_norm=True, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    wide = DiTBlock(n_head=24, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    assert not token_dit_train.serves(qk, x, c, p, None)
    assert not token_dit_train.serves(wide, x, c, p, None)
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80", "0")
    assert not token_dit_train.serves(eng, x, c, p, None)
    monkeypatch.delenv("MINIWORLD_AUGATTN_SM80")
    monkeypatch.setenv("MINIWORLD_TOKEN_DIT_TRAIN", "0")
    assert not token_dit_train.serves(eng, x, c, p, None)
