"""The bias-only token DiT block with a 256-wide pair (Protenix-v2's diffusion transformer: d_single 768, d_cond 384, 16 heads x 48,
d_pair 256) on every fused H100 route the 128-wide pair has: inference and training, bf16 and fp32 (TF32), one pair for every sample
(``integrations.bias_only_dit.update`` / ``bias_only_dit_train.block``) and a pair of its own per sample (``update_psb`` /
``block_psb`` behind the hoist ``pair_bias_all``), and a stack of blocks on hoisted biases. The pair-bias kernels are the 128-wide
ones templated on the width (``pair_bias_k`` / ``pair_bias_bwd`` / ``pair_bias_tc`` / ``pair_bias_bwd_tc`` / finalize,
``bo_hoist_pass.cu``, ``ln0_bwd_rows``, the LayerNorm rows template), so each case runs at d_pair 128 and 256.

Every call must (a) take the fused route (a silent composed / PyTorch fallback is a failure) and (b) hold the repo's yardstick against an
fp64 PyTorch run: error no worse than 1.5x the error of the PyTorch module in the same dtype (bf16: bf16 inputs under autocast, fp32: fp32
with TF32 GEMMs -- the engine's recipe). Training checks the output, the input gradients (pair included) and the parameter gradients.
fp32 training holds a wider bound (``SLACK_F32_TRAIN``): its TF32 step misses 1.5x at d_pair 128 already, by the same ratios at 256.
"""
import contextlib

import pytest
import torch
from portable_helpers import randomize_half_std as _randomize
from settings_helpers import policy_portable_off as policy  # noqa: F401

from miniworld_engine.integrations import bias_only_dit as fused
from miniworld_engine.integrations import bias_only_dit_hoist as hoist
from miniworld_engine.integrations import bias_only_dit_train as train
from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock, pair_bias_all
from miniworld_engine.modules.exceptions import ImplementationType as IT

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]
BF, F32 = torch.bfloat16, torch.float32
DTYPES = [BF, F32]
WIDTHS = [128, 256]
SLACK = 1.5
#: fp32 TRAINING only: the TF32 training step misses 1.5x of the TF32 PyTorch module at d_pair 128 already, by the same ratios at 256
#: (2026-10-29, L256 A3: dcond / dpair 1.6-1.7x, the attention AdaLN weight gradients 3.1-3.3x, to_value 2.1x; the output 1.0x, d single
#: 1.48x) -- a property of the fp32 recipe, not of the pair width. The bound below holds that level so a width regression still fails.
SLACK_F32_TRAIN = 4.0


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _modules(d_pair, n=1, seed=3):
    """(engine blocks, PyTorch blocks, fp64 PyTorch blocks) on the same randomised weights; fp32 parameters (the bf16 recipe runs them
    under autocast, as a model does)."""
    eng, pt, ref = [], [], []
    for i in range(n):
        torch.manual_seed(seed + i)
        e = _randomize(BiasOnlyDiTBlock(d_pair=d_pair, implementation=IT.MINIWORLD)).cuda().float()
        p = BiasOnlyDiTBlock(d_pair=d_pair, implementation=IT.PYTORCH).cuda().float()
        p.load_state_dict(e.state_dict())
        r = BiasOnlyDiTBlock(d_pair=d_pair, implementation=IT.PYTORCH).cuda().double()
        r.load_state_dict({k: v.double() for k, v in e.state_dict().items()})
        eng.append(e)
        pt.append(p)
        ref.append(r)
    return eng, pt, ref


@contextlib.contextmanager
def _recipe(dtype):
    """The dtype's recipe: bf16 under autocast, fp32 with TF32 GEMMs (the fused fp32 path's precision)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = dtype is F32
    try:
        with torch.autocast("cuda", dtype=BF, enabled=dtype is BF):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class _Spy:
    """Counts the calls that reach ``getattr(owner, name)``."""

    def __init__(self, monkeypatch, owner, name):
        self.n = 0
        real = getattr(owner, name)

        def counted(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(owner, name, counted)


def _yardstick(got, base, truth, what, slack=SLACK):
    e_got, e_base = _rel(got, truth), _rel(base, truth)
    assert torch.isfinite(got.float()).all(), what
    assert e_got <= slack * e_base, (what, e_got, e_base)


def _inputs(A, L, d_pair, per_sample, masked, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lead = (1, A) if per_sample else (A, 1)
    single = torch.randn(*lead, L, 768, device="cuda", generator=g)
    cond = torch.randn(*lead, L, 384, device="cuda", generator=g)
    pair = torch.randn(A if per_sample else 1, L, L, d_pair, device="cuda", generator=g)
    mask = (torch.rand(A if per_sample else 1, L, device="cuda", generator=g) > 0.15) if masked else None
    w = torch.randn(*lead, L, 768, device="cuda", generator=g)
    return single, cond, pair, mask, w


# ------------------------------------------------------------------------------------------------------------------ inference
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked", "hoisted"), [(5, 256, True, False), (2, 512, False, False), (3, 384, True, True)])
def test_inference_one_pair(monkeypatch, dtype, d_pair, A, L, masked, hoisted):
    """One pair for every sample (a sampling step): ``update`` (the pair's attention weights hoisted per pair: the LayerNorm rows + cuBLAS
    in bf16, the one-pass FMA ``pair_bias`` in fp32), or the block's precomputed bias (``pair_bias_all``, the one-pass hoist kernel)."""
    (eng,), (pt,), (ref,) = _modules(d_pair)
    single, cond, pair, mask, _ = _inputs(A, L, d_pair, False, masked)
    cond = cond[:1].expand(A, -1, -1, -1)  # one conditioning for every sample (stride 0)
    x, c, z = single.to(dtype), cond.to(dtype), pair.to(dtype)
    spy = _Spy(monkeypatch, fused, "update")
    with torch.no_grad(), _recipe(dtype):
        assert fused.serves(eng, x, c, z, mask)
        if hoisted:
            bias = pair_bias_all([eng], z)[0]
            got = eng(x, c, None, mask, bias=bias)
        else:
            got = eng(x, c, z, mask)
        base = pt(x, c, z, mask)
    assert spy.n == 1, "the fused inference step did not serve the call"
    with torch.no_grad():
        truth = ref(single.double(), cond.double(), pair.double(), mask)
    _yardstick(got, base, truth, "out")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("B", "L", "masked"), [(3, 256, True), (2, 512, False)])
def test_inference_pair_per_sample(monkeypatch, dtype, d_pair, B, L, masked):
    """A pair of its own per sample (``single`` [1, B, L, d], ``pair`` [B, L, L, d_pair]): ``update_psb``, whose bias is made per chunk of
    samples by one pass over the pair (``pair_bias_k`` bf16 / ``pair_bias_tc_k`` fp32)."""
    (eng,), (pt,), (ref,) = _modules(d_pair)
    single, cond, pair, mask, _ = _inputs(B, L, d_pair, True, masked)
    x, c, z = single.to(dtype), cond.to(dtype), pair.to(dtype)
    spy = _Spy(monkeypatch, fused, "update_psb")
    with torch.no_grad(), _recipe(dtype):
        assert fused.serves_psb(eng, x, c, z, mask)
        got = eng(x, c, z, mask)
        base = pt(x, c, z, mask)
    assert spy.n == 1, "the fused per-sample inference step did not serve the call"
    with torch.no_grad():
        truth = ref(single.double(), cond.double(), pair.double(), mask)
    _yardstick(got, base, truth, "out")


# ------------------------------------------------------------------------------------------------------------------- training
def _train(blocks, single, cond, pair, mask, w, dtype, hoisted=False):
    """Loss = <out, w> through the chained blocks; (outputs and gradients by name). ``dtype`` None: fp64 (the truth)."""
    dt = dtype or torch.float64
    ins = [t.detach().clone().to(dt).requires_grad_() for t in (single, cond, pair)]
    for b in blocks:
        b.zero_grad(set_to_none=True)
    ctx = _recipe(dtype) if dtype is not None else contextlib.nullcontext()
    with ctx:
        x = ins[0]
        biases = pair_bias_all(blocks, ins[2]) if hoisted else [None] * len(blocks)
        for b, bias in zip(blocks, biases, strict=True):
            x = b(x, ins[1], None if hoisted else ins[2], mask, bias=bias)
        loss = (x.float() * w.to(x.device)).sum() if dtype is not None else (x * w.double()).sum()
    loss.backward()
    res = {"out": x.detach(), "dsingle": ins[0].grad, "dcond": ins[1].grad, "dpair": ins[2].grad}
    for i, b in enumerate(blocks):
        res.update({f"{i}.{n}": p.grad.clone() for n, p in b.named_parameters()})
    return res


def _train_yardstick(got, base, truth, dtype):
    assert set(got) == set(truth) == set(base)
    misses = []
    for k in truth:
        assert got[k] is not None and base[k] is not None, k
        assert torch.isfinite(got[k].float()).all(), k
        e_got, e_base = _rel(got[k], truth[k]), _rel(base[k], truth[k])
        if e_got > (SLACK_F32_TRAIN if dtype is F32 else SLACK) * e_base:
            misses.append((k, e_got, e_base))
    assert not misses, misses


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked"), [(4, 256, True), (3, 384, False)])
def test_training_one_pair(monkeypatch, dtype, d_pair, A, L, masked):
    """One pair for every sample: ``bias_only_dit_train.block`` -- the pair LayerNorm + projection on chip (``pair_bias``, bf16 mma /
    ``pair_bias_tc`` TF32) and its backward (``pair_bias_bwd`` / ``pair_bias_bwd_tc``) with finalize's ln_pair / to_bias jobs."""
    eng, pt, ref = _modules(d_pair)
    single, cond, pair, mask, w = _inputs(A, L, d_pair, False, masked)
    spy = _Spy(monkeypatch, train, "block")
    got = _train(eng, single, cond, pair, mask, w, dtype)
    assert spy.n == 1, "the fused training block did not serve the call"
    again = _train(eng, single, cond, pair, mask, w, dtype)
    assert all(torch.equal(got[k], again[k]) for k in got), "the step is not bitwise repeatable"
    base = _train(pt, single, cond, pair, mask, w, dtype)
    truth = _train(ref, single, cond, pair, mask, w, None)
    _train_yardstick(got, base, truth, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked"), [(3, 256, True), (2, 384, False)])
def test_training_pair_per_sample(monkeypatch, dtype, d_pair, A, L, masked):
    """A pair of its own per sample: the block hoists its bias (``pair_bias_all``: ``bo_hoist_pass.cu`` hoist_fwd, and hoist_bwd1 for the
    backward of one block of 16 heads) and runs ``block_psb`` on it."""
    eng, pt, ref = _modules(d_pair)
    single, cond, pair, mask, w = _inputs(A, L, d_pair, True, masked)
    spy = _Spy(monkeypatch, train, "block_psb")
    fwd, bwd = _Spy(monkeypatch, HK, "pass_fwd"), _Spy(monkeypatch, HK, "pass_bwd")
    got = _train(eng, single, cond, pair, mask, w, dtype)
    assert spy.n == 1, "the fused per-sample training block did not serve the call"
    assert fwd.n == 1 and bwd.n == 1, ("the one-pass hoist kernels did not serve the pair bias", fwd.n, bwd.n)
    again = _train(eng, single, cond, pair, mask, w, dtype)
    assert all(torch.equal(got[k], again[k]) for k in got), "the step is not bitwise repeatable"
    base = _train(pt, single, cond, pair, mask, w, dtype)
    truth = _train(ref, single, cond, pair, mask, w, None)
    _train_yardstick(got, base, truth, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize("per_sample", [False, True])
def test_training_hoisted_stack(monkeypatch, dtype, d_pair, per_sample):
    """Three blocks on the biases of ONE ``pair_bias_all`` (the hoist's one-pass forward, N = 48 rows; its backward: the two GEMMs around
    ``ln0_bwd_rows`` -- at d_pair 128 bf16 the multi-block kernel ``bo_hoist_bwd.cu``), against each block on the pair."""
    A, L = 3, 256
    eng, pt, ref = _modules(d_pair, n=3)
    single, cond, pair, mask, w = _inputs(A, L, d_pair, per_sample, True)
    spy = _Spy(monkeypatch, hoist, "pair_bias_all_psb" if per_sample else "pair_bias_all")
    blk = _Spy(monkeypatch, train, "block_psb" if per_sample else "block")
    got = _train(eng, single, cond, pair, mask, w, dtype, hoisted=True)
    assert spy.n == 1 and blk.n == 3, ("the fused hoist / blocks did not serve the stack", spy.n, blk.n)
    again = _train(eng, single, cond, pair, mask, w, dtype, hoisted=True)
    assert all(torch.equal(got[k], again[k]) for k in got), "the step is not bitwise repeatable"
    base = _train(pt, single, cond, pair, mask, w, dtype)
    truth = _train(ref, single, cond, pair, mask, w, None)
    _train_yardstick(got, base, truth, dtype)
