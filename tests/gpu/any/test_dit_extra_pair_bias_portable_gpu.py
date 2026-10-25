"""``DiTBlock(extra_pair_bias=)`` off the Hopper bf16 CUDA path: an additive [L, L] logit term for every head, on top of the pair's own bias.
It rides the attention's bias input, so the composed path serves it on every card (portable attention kernels / PyTorch statements), in bf16
and fp32, inference and training (gradient to the activations, the weights and the term itself), with key masks. Checked against the block's
fp64 PyTorch statements (bf16 error held to the bf16 PyTorch block's own), against the Hopper CUDA path where there is one, and by the
softmax shift property (a constant term changes nothing).
"""
from __future__ import annotations

import copy

import pytest
import torch
from portable_helpers import portable_on, rel, report

from miniworld_engine.integrations import token_dit
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType as I

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF, F32 = "cuda", torch.bfloat16, torch.float32

WIDTHS = [(768, 384, 128, 16), (128, 128, 64, 4), (256, 128, 64, 8)]


@pytest.fixture(autouse=True)
def portable():
    with portable_on():
        yield


def _block(width, implementation, seed=5):
    d_single, d_cond, d_pair, n_head = width
    torch.manual_seed(seed)
    return DiTBlock(d_single, d_cond, d_pair, n_head, implementation=implementation).to(DEV)


def _randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else
                    1 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
    return m


def _data(width, A, B, L):
    d_single, d_cond, d_pair, _ = width
    g = torch.Generator(device=DEV).manual_seed(2)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)   # noqa: E731
    mask = torch.rand(B, L, device=DEV, generator=g) > 0.1
    return r(A, B, L, d_single), r(A, B, L, d_cond), r(B, L, L, d_pair), mask, 2 * r(L, L), r(A, B, L, d_single)


def _run(block, data, dtype, training):
    single, cond, pair, mask, extra, dy = data
    single, cond, pair, extra = (t.to(dtype).clone().requires_grad_(training) for t in (single, cond, pair, extra))
    with torch.enable_grad() if training else torch.no_grad():
        out = block(single, cond, pair, mask, extra_pair_bias=extra)
        if not training:
            return out, ()
        return out, torch.autograd.grad(out, [single, cond, pair, extra, *block.parameters()], dy.to(dtype), allow_unused=True)


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(("width", "A", "B", "L"), [(WIDTHS[0], 2, 1, 192), (WIDTHS[1], 1, 2, 130), (WIDTHS[2], 2, 1, 160)],
                         ids=["768h16-A2B1", "128h4-A1B2-ragged", "256h8-A2B1"])
def test_extra_pair_bias_matches_pytorch(width, dtype, training, A, B, L):
    base = _randomize(_block(width, I.PYTORCH))
    gold = copy.deepcopy(base).double()
    ref = _block(width, I.PYTORCH)
    ref.load_state_dict(base.state_dict())
    ref = ref.to(dtype)
    eng = _block(width, I.MINIWORLD)
    eng.load_state_dict(base.state_dict())
    eng = eng.to(dtype)
    data = _data(width, A, B, L)
    out_gold, g_gold = _run(gold, tuple(t.double() if t.dtype.is_floating_point else t for t in data), torch.float64, training)
    out_ref, g_ref = _run(ref, data, dtype, training)
    out, g_eng = _run(eng, data, dtype, training)
    assert torch.isfinite(out).all()
    report(f"dit_extra_pair_bias w{width[0]}h{width[3]} {str(dtype)[6:]} {'train' if training else 'infer'} A{A}B{B}L{L}", out, out_ref, out_gold, g_eng, g_ref, g_gold)
    floor = 3e-3 if dtype is BF else 6e-3      # fp32 engine = TF32 tensor cores in the portable kernels; the fp32 PyTorch block is full fp32
    factor = 1.5 if dtype is BF else 1.0
    assert rel(out, out_gold) <= max(factor * rel(out_ref, out_gold), floor), (rel(out, out_gold), rel(out_ref, out_gold))
    for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_ref, strict=True)):
        if r is None:
            continue
        assert rel(a, r) <= max(factor * rel(c, r), 2 * floor), (i, rel(a, r), rel(c, r))
    if training:
        assert g_eng[3] is not None and g_eng[3].abs().max() > 0, "the term itself must receive a gradient"


@pytest.mark.parametrize("implementation", [I.PYTORCH, I.MINIWORLD])
def test_a_constant_term_changes_nothing(implementation):
    """softmax(z + c) == softmax(z): the term is on the logits, before the softmax, for every head."""
    width = WIDTHS[1]
    m = _randomize(_block(width, implementation)).to(BF).eval()
    single, cond, pair, mask, _, _ = _data(width, 2, 1, 160)
    args = (single.to(BF), cond.to(BF), pair.to(BF), mask)
    with torch.no_grad():
        base = m(*args)
        shifted = m(*args, extra_pair_bias=torch.full((160, 160), 3.0, device=DEV, dtype=BF))
        zeros = m(*args, extra_pair_bias=torch.zeros(160, 160, device=DEV, dtype=BF))
        varying = m(*args, extra_pair_bias=4 * torch.randn(160, 160, device=DEV, dtype=BF))
    assert rel(shifted, base) < 1e-2          # bf16 rounds the shifted logits differently
    assert rel(zeros, base) < 1e-6
    assert rel(varying, base) > 1e-2, "a varying term must move the output"


def test_a_wrong_shape_is_an_error_not_a_dropped_term():
    m = _block(WIDTHS[2], I.MINIWORLD).to(BF)
    single, cond, pair, mask, _, _ = _data(WIDTHS[2], 1, 1, 64)
    with pytest.raises(ValueError, match="extra_pair_bias"):
        m(single.to(BF), cond.to(BF), pair.to(BF), mask, extra_pair_bias=torch.zeros(63, 64, device=DEV, dtype=BF))


def test_matches_the_hopper_cuda_path():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 only")
    width = WIDTHS[0]
    gold = _randomize(_block(width, I.PYTORCH))
    single, cond, pair, mask, extra, _ = _data(width, 3, 1, 256)
    a = tuple(t.to(BF) for t in (single, cond, pair)) + (mask,)
    ref16 = _block(width, I.PYTORCH)
    ref16.load_state_dict(gold.state_dict())
    ref16 = ref16.to(BF)
    with torch.no_grad():
        want = gold(single, cond, pair, mask, extra_pair_bias=extra)
        r16 = ref16(*a, extra_pair_bias=extra.to(BF))
    with portable_on(False):
        cuda_block = _block(width, I.MINIWORLD)
        cuda_block.load_state_dict(gold.state_dict())
        cuda_block = cuda_block.to(BF)
        assert token_dit.serves_extra_bias(a[0], extra.to(BF)), "the CUDA path must be the one under test"
        with torch.no_grad():
            c = cuda_block(*a, extra_pair_bias=extra.to(BF))
    portable_block = _block(width, I.MINIWORLD)
    portable_block.load_state_dict(gold.state_dict())
    portable_block = portable_block.to(BF)
    with torch.no_grad():
        p = portable_block(*a, extra_pair_bias=extra.to(BF))
    bound = max(1.5 * rel(r16, want), 3e-3)
    assert rel(c, want) <= bound
    assert rel(p, want) <= bound
