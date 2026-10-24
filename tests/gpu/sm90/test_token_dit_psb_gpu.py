"""The fused token DiT training block with an attention bias of its own per sample (``token_dit_train.block_psb``) against the
fp32 PyTorch DiTBlock fed the same per-sample pair (IEEE, TF32 off, samples folded into the batch): output and every input /
parameter gradient no worse than the engine's own bf16 module path in the same regime."""
import math

import pytest
import torch
import torch.nn.functional as F

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
LOG2E = 1.0 / math.log(2.0)


@pytest.fixture(autouse=True)
def policy():
    old = settings.configure(force_triton_kernels=False)
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


def _collect(m, out, ins, w):
    (out.float() * w).sum().backward()
    res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(),
           "dpair": ins[2].grad.float()}
    res.update({n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


def _run_folded(m, single, cond, pair, mask, w, dtype, tf32):
    """The module path with the A samples folded into the batch, so each reads its own pair."""
    A, _, L, _ = single.shape
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
        key = None if mask is None else mask.expand(A, L)
        out = m(ins[0].reshape(1, A, L, -1), ins[1].reshape(1, A, L, -1), ins[2], key)
        res = _collect(m, out.reshape(A, 1, L, -1), ins, w)
        return res
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _run_psb(m, single, cond, pair, mask, w):
    """The fused block; the bias is read off the pair by torch ops, so the bias producer's gradients arrive by autograd."""
    A, _, L, _ = single.shape
    ins = [t.detach().clone().to(torch.bfloat16).requires_grad_() for t in (single, cond, pair)]
    a = m.attention
    bias = F.linear(F.layer_norm(ins[2].float(), (128,), a.ln_pair.weight.float(), None, a.ln_pair.eps), a.to_bias.weight.float())
    bias = (bias * LOG2E).permute(3, 0, 1, 2).contiguous().to(torch.bfloat16)        # [H, A, L, L]
    assert token_dit_train.serves_psb(m, ins[0], ins[1], bias, mask)
    out = token_dit_train.block_psb(m, ins[0], ins[1], bias, mask)
    return _collect(m, out, ins, w)


@pytest.mark.parametrize(("A", "L", "masked", "qk", "d_cond"), [
    (3, 256, False, True, 384),
    (4, 256, True, False, 384),
    (3, 384, True, True, 384),
    (4, 128, False, True, 384),
    (3, 200, True, True, 384),      # not a multiple of the 128-row tile: padded inside the block
    (4, 96, False, True, 768),
    (3, 256, True, True, 768),      # a token DiT conditioned at its own width
    (4, 384, False, False, 768),
])
def test_per_sample_bias_block_matches_the_module_path(A, L, masked, qk, d_cond):
    torch.manual_seed(7)
    ref = _randomize(DiTBlock(d_cond=d_cond, use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(d_cond=d_cond, use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    eng = eng.to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(8)
    single = torch.randn(A, 1, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, d_cond, device="cuda", generator=g)
    pair = torch.randn(A, L, L, 128, device="cuda", generator=g)
    mask = (torch.rand(1, L, device="cuda", generator=g) > 0.15) if masked else None
    w = torch.randn(A, 1, L, 768, device="cuda", generator=g)

    truth = _run_folded(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    fused = _run_psb(eng, single, cond, pair, mask, w)
    module = _run_folded(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True)
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize(("A", "L", "blocks"), [(3, 256, 4), (4, 384, 12), (2, 128, 3), (2, 200, 3)])
def test_pair_biases_match_the_per_block_layernorm_and_projection(A, L, blocks):
    """One LayerNorm and one GEMM leave every block's bias; value and gradients against each block's own torch LN + projection."""
    from miniworld_engine.integrations.token_dit_pair_bias import pair_biases

    torch.manual_seed(5)
    g = torch.Generator(device="cuda").manual_seed(6)
    pair = torch.randn(A, L, L, 128, device="cuda", generator=g)
    lns = [(1 + 0.2 * torch.randn(128, device="cuda", generator=g)) for _ in range(blocks)]
    wbs = [0.2 * torch.randn(16, 128, device="cuda", generator=g) for _ in range(blocks)]
    w = [torch.randn(16, A, L, L, device="cuda", generator=g) for _ in range(blocks)]

    def truth(pair, lns, wbs):
        out = []
        for ln, wb in zip(lns, wbs, strict=True):
            b = F.linear(F.layer_norm(pair, (128,), ln, None, 1e-5), wb) * LOG2E       # [A, L, L, H]
            out.append(b.permute(3, 0, 1, 2))
        return out

    leaves = lambda: [t.detach().clone().requires_grad_() for t in (pair, *lns, *wbs)]
    ref = leaves()
    sum((o * wi).sum() for o, wi in zip(truth(ref[0], ref[1 : 1 + blocks], ref[1 + blocks :]), w, strict=True)).backward()
    got = leaves()
    outs = pair_biases(got[0], got[1 : 1 + blocks], got[1 + blocks :])
    sum((o.float() * wi).sum() for o, wi in zip(outs, w, strict=True)).backward()
    expect = truth(pair, lns, wbs)
    for b in range(blocks):
        assert _rel(outs[b].float(), expect[b]) < 1e-2, f"bias {b}"
    for name, x, y in zip(["dpair", *[f"dln{b}" for b in range(blocks)], *[f"dwb{b}" for b in range(blocks)]], got, ref, strict=True):
        assert _rel(x.grad, y.grad) < 2e-2, name
