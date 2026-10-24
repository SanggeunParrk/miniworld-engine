"""A token DiT block over B independent proteins (``single [1, B, L, d]``, ``pair [B, L, L, dp]``, a key mask of its own per protein)
at La-Proteina's widths -- d_cond 256 or 128, pair 256, transition x4, 12 heads x 64 -- runs the fused H100 training block with an
attention bias of its own per sample and agrees with the fp32 PyTorch DiTBlock: output and every input / parameter gradient no worse
than the engine's own bf16 module path in the same regime. The AF3 widths (cond 384, pair 128, x2, 16 heads x 48) batched the same way
are the regression cases."""
import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


@pytest.fixture(autouse=True)
def policy():
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


def _run(m, single, cond, pair, mask, w, dtype, tf32, fused):
    """Forward + backward of one call; the gradients of the inputs and of every parameter."""
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old = settings.configure(token_dit_train=fused)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
        out = m(ins[0], ins[1], ins[2], mask)
        (out.float() * w).sum().backward()
        res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(),
               "dpair": ins[2].grad.float()}
        res.update({n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
        m.zero_grad(set_to_none=True)
        return res
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        settings.configure(**vars(old))


@pytest.mark.parametrize(("B", "L", "masked", "qk", "d_cond", "d_pair", "heads", "n"), [
    (3, 256, True, True, 256, 256, 12, 4),      # the flow model's trunk
    (4, 384, True, True, 256, 256, 12, 4),
    (3, 200, True, True, 128, 256, 12, 4),      # the autoencoder's trunk, L not a multiple of the 128-row tile
    (2, 256, False, False, 256, 256, 12, 4),
    (3, 256, True, True, 384, 128, 16, 2),      # AF3 widths, batched
    (4, 128, True, False, 384, 128, 12, 2),
    # one width at a time away from the AF3 block, to localize a disagreement
    (3, 256, True, True, 384, 128, 16, 4),      # n only
    (3, 256, True, True, 256, 128, 16, 2),      # d_cond only
    (3, 256, True, True, 384, 256, 16, 2),      # pair channels only
    (3, 256, True, True, 384, 128, 12, 2),      # heads only
])
def test_batched_proteins_block_matches_the_module_path(B, L, masked, qk, d_cond, d_pair, heads, n):
    torch.manual_seed(7)
    kw = dict(d_single=768, d_cond=d_cond, d_pair=d_pair, n_head=heads, n=n, use_qk_norm=qk)
    ref = _randomize(DiTBlock(**kw, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(**kw, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    eng = eng.to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(8)
    single = torch.randn(1, B, L, 768, device="cuda", generator=g)
    cond = torch.randn(1, B, L, d_cond, device="cuda", generator=g)
    pair = torch.randn(B, L, L, d_pair, device="cuda", generator=g)
    if masked:  # each protein its own length: a valid prefix, the rest padding
        lengths = torch.randint(L // 2, L + 1, (B,), device="cuda", generator=g)
        mask = torch.arange(L, device="cuda")[None] < lengths[:, None]
    else:
        mask = None
    w = torch.randn(1, B, L, 768, device="cuda", generator=g)
    if mask is not None:
        w = w * mask[None, :, :, None]  # padded rows are not part of the loss

    truth = _run(ref, single, cond, pair, mask, w, torch.float32, tf32=False, fused=False)
    module = _run(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True, fused=False)
    calls = []
    real = token_dit_train.block_psb
    token_dit_train.block_psb = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        fused = _run(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True, fused=True)
    finally:
        token_dit_train.block_psb = real
    assert calls, "the call did not take the fused per-sample-bias block"
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize(("B", "L", "d_cond", "heads"), [(3, 200, 128, 12), (4, 256, 256, 12)])
def test_batched_proteins_block_is_bitwise_deterministic(B, L, d_cond, heads):
    """Two runs of the same forward + backward (masked, padded length, La-Proteina's widths) agree bit for bit: output and every gradient."""
    torch.manual_seed(11)
    kw = dict(d_single=768, d_cond=d_cond, d_pair=256, n_head=heads, n=4, use_qk_norm=True)
    eng = _randomize(DiTBlock(**kw, implementation=ImplementationType.MINIWORLD).cuda()).to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(12)
    single = torch.randn(1, B, L, 768, device="cuda", generator=g)
    cond = torch.randn(1, B, L, d_cond, device="cuda", generator=g)
    pair = torch.randn(B, L, L, 256, device="cuda", generator=g)
    mask = torch.arange(L, device="cuda")[None] < torch.randint(L // 2, L + 1, (B,), device="cuda", generator=g)[:, None]
    w = torch.randn(1, B, L, 768, device="cuda", generator=g) * mask[None, :, :, None]
    first = _run(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True, fused=True)
    second = _run(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True, fused=True)
    assert set(first) == set(second)
    for k in first:
        assert torch.equal(first[k], second[k]), f"{k}: two identical runs differ"
