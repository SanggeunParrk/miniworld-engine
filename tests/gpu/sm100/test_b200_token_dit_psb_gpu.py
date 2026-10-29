"""The B200 fused token DiT training block with an attention bias of its own per sample (``token_dit_train.block_psb`` /
``block_batched``: the AttentionPairBias cores attn_fwd2 -DQPAIR, attn_dkv / attn_dqb -DBPS, built for the token head layouts)
against the fp32 PyTorch DiTBlock fed the same per-sample pair (IEEE, TF32 off, samples folded into the batch): output and every
input / parameter gradient no worse than 1.5x the bf16 PyTorch module's own error in the same regime (+3e-3), as
tests/gpu/sm90/test_token_dit_psb_gpu.py on H100."""
import copy
import math

import pytest
import torch
import torch.nn.functional as F
from portable_helpers import randomize_half_std as _randomize
from settings_helpers import policy_portable_off as policy  # noqa: F401

from miniworld_engine.integrations import token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="Blackwell (sm_100) required"
    ),
]
LOG2E = 1.0 / math.log(2.0)


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


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
        return _collect(m, out.reshape(A, 1, L, -1), ins, w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _run_psb(m, single, cond, pair, mask, w):
    """``block_psb`` on the H100 contract ([H, A, L, L] bf16 log2 units); the bias is read off the pair by torch ops, so the bias
    producer's gradients arrive by autograd."""
    A, _, L, _ = single.shape
    ins = [t.detach().clone().to(torch.bfloat16).requires_grad_() for t in (single, cond, pair)]
    a = m.attention
    bias = F.linear(F.layer_norm(ins[2].float(), (128,), a.ln_pair.weight.float(), None, a.ln_pair.eps), a.to_bias.weight.float())
    bias = (bias * LOG2E).permute(3, 0, 1, 2).contiguous().to(torch.bfloat16)        # [H, A, L, L]
    assert token_dit_train.serves_psb(m, ins[0], ins[1], bias, mask)
    out = token_dit_train.block_psb(m, ins[0], ins[1], bias, mask)
    return _collect(m, out, ins, w)


def _blocks(qk, heads, d, seed=7):
    torch.manual_seed(seed)
    ref = _randomize(DiTBlock(d_single=d, n_head=heads, use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(d_single=d, n_head=heads, use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    return ref, eng.to(torch.bfloat16), copy.deepcopy(ref).to(torch.bfloat16)


def _data(A, L, d, masked, per_sample_mask=False, seed=8):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(A, 1, L, d, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(A, L, L, 128, device="cuda", generator=g)
    mask = None
    if masked:
        mask = torch.rand(A if per_sample_mask else 1, L, device="cuda", generator=g) > 0.15
        mask[:, 0] = True
    w = torch.randn(A, 1, L, d, device="cuda", generator=g)
    return single, cond, pair, mask, w


def _check(fused, truth, module):
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs bf16 module {em:.2e}"


@pytest.mark.parametrize(("A", "L", "masked", "qk", "heads", "d"), [
    (3, 256, False, True, 16, 768),
    (4, 256, True, False, 16, 768),
    (3, 384, True, True, 16, 768),
    (4, 128, False, True, 16, 768),
    (3, 200, True, True, 16, 768),      # not a multiple of the 128-row tile: padded inside the block
    (2, 96, False, False, 16, 768),
    (3, 256, True, False, 24, 768),     # 24 x 32
    (3, 256, True, True, 12, 768),      # 12 x 64
    (2, 256, True, False, 16, 1024),    # 16 x 64
])
def test_b200_per_sample_bias_block_matches_the_module(A, L, masked, qk, heads, d):
    ref, eng, ref_bf = _blocks(qk, heads, d)
    single, cond, pair, mask, w = _data(A, L, d, masked)
    truth = _run_folded(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    fused = _run_psb(eng, single, cond, pair, mask, w)
    module = _run_folded(ref_bf, single, cond, pair, mask, w, torch.bfloat16, tf32=True)
    _check(fused, truth, module)


@pytest.mark.parametrize(("B", "L", "masked"), [(3, 256, True), (2, 200, True), (4, 128, False)])
def test_b200_batched_proteins_route_to_the_fused_block(B, L, masked, monkeypatch):
    """``DiTBlock`` over B independent proteins ([1, B, L, d], pair [B, L, L, 128], a key mask of its own per protein) takes
    ``block_batched`` on B200, and matches the fp32 module."""
    calls = {"n": 0}
    inner = token_dit_train.block_batched

    def counted(*a, **k):
        calls["n"] += 1
        return inner(*a, **k)

    monkeypatch.setattr(token_dit_train, "block_batched", counted)
    ref, eng, ref_bf = _blocks(True, 16, 768, seed=11)
    single, cond, pair, mask, w = _data(B, L, 768, masked, per_sample_mask=True, seed=12)
    truth = _run_folded(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    fused = _run_folded(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True)
    assert calls["n"] == 1, "the fused batched block did not serve this call"
    module = _run_folded(ref_bf, single, cond, pair, mask, w, torch.bfloat16, tf32=True)
    _check(fused, truth, module)


def test_b200_per_sample_bias_fully_masked_sample_stays_finite():
    """A sample whose keys are all masked (finite NEG fill): its output and every gradient stay finite."""
    _, eng, _ = _blocks(False, 16, 768, seed=13)
    single, cond, pair, mask, w = _data(2, 128, 768, True, per_sample_mask=True, seed=14)
    mask[1] = False
    fused = _run_psb(eng, single, cond, pair, mask, w)
    for k, v in fused.items():
        assert torch.isfinite(v).all(), k
