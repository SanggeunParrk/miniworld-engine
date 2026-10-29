"""``DiTBlock(extra_pair_bias=)`` on the B200 hand-CUDA paths: inference (bf16, fp32/TF32; the [L, L] term added to the hoisted log2-unit
pair bias, cached with it per pair / mask / extra) and training (bf16, fp32/TF32; added to the fused block's natural-unit bias, the
term's gradient the attention's dbias summed over the heads). FoldForge's structural-token family hands every token DiT block this
term ([A, 1, L, 768], cond 384, pair [1, L, L, 128], 16 heads, any L).

Output and every gradient are held to 1.5 x the same-dtype PyTorch error against fp64 (fp32: the engine runs TF32 tensor cores, the
PyTorch block full fp32, so the bound has an absolute floor), as tests/gpu/sm90/test_dit_extra_pair_bias_h100_gpu.py. The routing is
asserted: the CUDA block must be the one that ran, not a silent fallback to the composed statements."""

from __future__ import annotations

import copy

import pytest
import torch
from portable_helpers import portable_on, rel, report
from portable_helpers import randomize_normalised as _randomize
from portable_helpers import run_extra_pair_bias as _run

from miniworld_engine.integrations import token_dit, token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType as I

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="Blackwell (sm_100) required"
    ),
]
DEV, BF, F32 = "cuda", torch.bfloat16, torch.float32
WIDTH = (768, 384, 128, 16)


def _block(implementation, seed=5):
    torch.manual_seed(seed)
    return DiTBlock(*WIDTH, implementation=implementation).to(DEV)


def _data(A, L, masked, seed=2):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)   # noqa: E731
    mask = (torch.rand(1, L, device=DEV, generator=g) > 0.1) if masked else None
    return r(A, 1, L, WIDTH[0]), r(A, 1, L, WIDTH[1]), r(1, L, L, WIDTH[2]), mask, 2 * r(L, L), r(A, 1, L, WIDTH[0])


@pytest.fixture
def routes(monkeypatch):
    """Counts the calls of the CUDA blocks (inference runner, training block)."""
    n = {"infer": 0, "train": 0}
    infer, train = token_dit.update, token_dit_train.block

    def update(*a, **k):
        n["infer"] += 1
        return infer(*a, **k)

    def block(*a, **k):
        n["train"] += 1
        return train(*a, **k)

    monkeypatch.setattr(token_dit, "update", update)
    monkeypatch.setattr(token_dit_train, "block", block)
    return n


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    ("A", "L", "masked"),
    [(2, 256, False), (2, 256, True), (3, 130, True), (5, 203, True)],
    ids=["A2L256", "A2L256-mask", "A3L130-ragged", "A5L203-ragged"],
)
def test_b200_extra_pair_bias_cuda_matches_pytorch(dtype, training, A, L, masked, routes):
    base = _randomize(_block(I.PYTORCH))
    gold = copy.deepcopy(base).double()
    ref = _block(I.PYTORCH)
    ref.load_state_dict(base.state_dict())
    ref = ref.to(dtype)
    eng = _block(I.MINIWORLD)
    eng.load_state_dict(base.state_dict())
    eng = eng.to(dtype)
    data = _data(A, L, masked)
    out_gold, g_gold = _run(gold, tuple(t.double() if t is not None and t.dtype.is_floating_point else t for t in data), torch.float64, training)
    out_ref, g_ref = _run(ref, data, dtype, training)
    with portable_on(False):
        out, g_eng = _run(eng, data, dtype, training)
    # B200: inference pads L to a multiple of 8, training to 128 (and A to an even count)
    assert routes["train" if training else "infer"] == 1, f"the CUDA block did not serve this call: {routes}"
    assert torch.isfinite(out).all()
    report(f"dit_xbias_b200 {str(dtype)[6:]} {'train' if training else 'infer'} A{A}L{L}m{int(masked)}", out, out_ref, out_gold, g_eng, g_ref, g_gold)
    floor = 3e-3 if dtype is BF else 6e-3      # fp32 engine = TF32 tensor cores, the fp32 PyTorch block is full fp32: the bound has a floor
    factor = 1.5 if dtype is BF else 1.0
    assert rel(out, out_gold) <= max(factor * rel(out_ref, out_gold), floor), (rel(out, out_gold), rel(out_ref, out_gold))
    for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_ref, strict=True)):
        if r is None:
            continue
        assert rel(a, r) <= max(factor * rel(c, r), 2 * floor), (i, rel(a, r), rel(c, r))
    if training:
        assert g_eng[3] is not None and g_eng[3].abs().max() > 0, "the term itself must receive a gradient"


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_b200_extra_pair_bias_inference_cache_follows_the_term(dtype):
    """The hoisted bias is cached per pair: a different extra tensor, or the same one changed in place, must not be served the
    cached bias of the previous term (each call equals a fresh block's call on that term)."""
    base = _randomize(_block(I.MINIWORLD)).to(dtype)
    single, cond, pair, mask, extra, _ = (t if t is None else t.to(dtype) for t in _data(2, 200, True, seed=7))
    extra2 = (extra * -0.5).contiguous()

    def fresh(e):
        b = copy.deepcopy(base)
        with torch.no_grad():
            return b(single, cond, pair, mask, extra_pair_bias=e.clone())

    with portable_on(False), torch.no_grad():
        y1 = base(single, cond, pair, mask, extra_pair_bias=extra)
        y2 = base(single, cond, pair, mask, extra_pair_bias=extra2)
        extra.mul_(-0.5)  # in place: a version bump on the same tensor
        y3 = base(single, cond, pair, mask, extra_pair_bias=extra)
        y0 = base(single, cond, pair, mask)
        assert torch.equal(y2, fresh(extra2))
        assert torch.equal(y3, fresh(extra2))
        assert not torch.equal(y1, y2) and not torch.equal(y0, y2)


def test_b200_extra_pair_bias_training_is_bitwise_repeatable():
    """Output, the input gradients and the term's own gradient (a sum over the heads of dbias, no atomics) repeat bitwise. The B200
    block's two ``ada_ln_in.ln_cond.weight`` gradients are not bitwise run to run with or without the term (measured 10-29 at
    a014b518, A2 L256 bf16): they are left out here."""
    eng = _randomize(_block(I.MINIWORLD)).to(BF)
    data = _data(2, 256, True)
    with portable_on(False):
        a = _run(eng, data, BF, True)
        b = _run(eng, data, BF, True)
    assert torch.equal(a[0], b[0])
    names = ["single", "cond", "pair", "extra"] + [n for n, _ in eng.named_parameters()]
    for name, x, y in zip(names, a[1], b[1], strict=True):
        if name.endswith("ada_ln_in.ln_cond.weight"):
            continue
        assert (x is None and y is None) or torch.equal(x, y), name


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("L", [1, 5, 13, 203])
def test_b200_token_dit_inference_any_length(dtype, L, routes):
    """B200 token DiT inference at any L >= 1 (the host pads to a multiple of 8, the padded keys masked out), with and without the
    term: within the same-dtype PyTorch error against fp64, as above."""
    base = _randomize(_block(I.PYTORCH))
    gold = copy.deepcopy(base).double()
    ref = copy.deepcopy(base).to(dtype)
    eng = _block(I.MINIWORLD)
    eng.load_state_dict(base.state_dict())
    eng = eng.to(dtype)
    for use_extra in (False, True):
        data = list(_data(3, L, L > 1, seed=9))
        if not use_extra:
            data[4] = None
        cast = lambda t, dt: None if t is None else (t.to(dt) if t.dtype.is_floating_point else t)   # noqa: E731
        with torch.no_grad():
            args = lambda dt: [cast(t, dt) for t in data[:4]]   # noqa: E731
            kw = lambda dt: {} if data[4] is None else {"extra_pair_bias": cast(data[4], dt)}   # noqa: E731
            out_gold = gold(*args(torch.float64), **kw(torch.float64))
            out_ref = ref(*args(dtype), **kw(dtype))
            with portable_on(False):
                out = eng(*args(dtype), **kw(dtype))
        assert torch.isfinite(out).all()
        floor = 3e-3 if dtype is BF else 6e-3
        factor = 1.5 if dtype is BF else 1.0
        assert rel(out, out_gold) <= max(factor * rel(out_ref, out_gold), floor), (use_extra, rel(out, out_gold), rel(out_ref, out_gold))
    assert routes["infer"] == 2, f"the CUDA block did not serve these calls: {routes}"
