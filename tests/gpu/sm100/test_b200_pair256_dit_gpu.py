"""B200: the token DiT and the bias-only DiT block with a 256-wide pair (Protenix-v2's diffusion transformer: d_single 768, d_cond 384,
16 heads x 48, d_pair 256) on the fused sm_100 routes the 128-wide pair takes -- inference and training, bf16 and fp32 (TF32). Before,
B200 ran these blocks' composed path at d_pair 256 (the token DiT: the module composition; the bias-only DiT: the reference composition
with the portable pair-bias kernels). Only the pair bias reads the pair, and its kernels are the 128-wide ones templated on the width
(the LayerNorm rows template + cuBLAS, ``pair_ln`` / ``pair_bias`` / their backwards, ``pair_bias_k`` / ``pair_bias_tc_k``,
``ln0_bwd_rows``), so every case runs at d_pair 128 and 256.

Every call must (a) take the fused route (a silent composed / PyTorch fallback is a failure) and (b) hold the yardstick of the H100
pair-256 tests (``tests/gpu/sm90/test_*_pair256_gpu.py``) against an fp64 PyTorch run: error no worse than 1.5x the PyTorch module's in
the same dtype (bf16 under autocast; fp32 with TF32 GEMMs), training every gradient as well (fp32 training: ``F32_TRAIN_ABS``)."""
import contextlib

import pytest
import torch
from portable_helpers import randomize_half_std as _randomize
from settings_helpers import policy_portable_off as policy  # noqa: F401

from miniworld_engine.integrations import bias_only_dit as fused
from miniworld_engine.integrations import bias_only_dit_hoist as hoist
from miniworld_engine.integrations import bias_only_dit_train as train
from miniworld_engine.integrations import token_dit as TD
from miniworld_engine.integrations import token_dit_train as TDT
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock, pair_bias_all
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType as IT

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100) required"
    ),
]
BF, F32 = torch.bfloat16, torch.float32
DTYPES = [BF, F32]
WIDTHS = [128, 256]
SLACK = 1.5
#: fp32 TRAINING: the yardstick of the B200 TF32 training test (``test_b200_bias_only_dit_tf32_gpu.py``: 1.5x the TF32 PyTorch module's
#: error + 1e-3). The ratio alone misses at d_pair 128 already (A4 L256: the attention AdaLN weight gradients 9.7e-4 against 1.8e-4) --
#: a property of the fp32 recipe, not of the pair width.
F32_TRAIN_ABS = 1e-3
#: ... or under the TF32 kernel bound of the B200 hoist test (``test_b200_bias_only_dit_hoist_gpu.py``: 3e-3 against fp64): the sm_100a
#: one-pass hoist at d_pair 128 rounds LN0 to TF32 for dW' / d pair (d pair 1.66e-3 against the TF32 module's 4.0e-4 there)
F32_TRAIN_BOUND = 3e-3


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@contextlib.contextmanager
def _recipe(dtype):
    """bf16 under autocast, fp32 with TF32 GEMMs (the fused fp32 path's precision)."""
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


def _modules(cls, n=1, seed=3, **kw):
    """(engine blocks, PyTorch blocks, fp64 PyTorch blocks) on the same randomised weights, fp32 parameters."""
    eng, pt, ref = [], [], []
    for i in range(n):
        torch.manual_seed(seed + i)
        e = _randomize(cls(**kw, implementation=IT.MINIWORLD)).cuda().float()
        p = cls(**kw, implementation=IT.PYTORCH).cuda().float()
        p.load_state_dict(e.state_dict())
        r = cls(**kw, implementation=IT.PYTORCH).cuda().double()
        r.load_state_dict({k: v.double() for k, v in e.state_dict().items()})
        eng.append(e)
        pt.append(p)
        ref.append(r)
    return eng, pt, ref


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
            x = b(x, ins[1], None if hoisted else ins[2], mask, bias=bias) if hoisted else b(x, ins[1], ins[2], mask)
        loss = (x.float() * w.to(x.device)).sum() if dtype is not None else (x * w.double()).sum()
    loss.backward()
    res = {"out": x.detach(), "dsingle": ins[0].grad, "dcond": ins[1].grad, "dpair": ins[2].grad}
    for i, b in enumerate(blocks):
        res.update({f"{i}.{n}": p.grad.clone() for n, p in b.named_parameters() if p.grad is not None})
    return res


def _train_yardstick(got, base, truth, dtype):
    assert set(got) == set(truth) == set(base), sorted(set(got) ^ set(truth))
    misses = []
    for k in truth:
        assert torch.isfinite(got[k].float()).all(), k
        e_got, e_base = _rel(got[k], truth[k]), _rel(base[k], truth[k])
        ok = e_got <= SLACK * e_base + (F32_TRAIN_ABS if dtype is F32 else 0.0) or (dtype is F32 and e_got < F32_TRAIN_BOUND)
        if not ok:
            misses.append((k, e_got, e_base))
    assert not misses, misses


# ======================================================================================================== token DiT (DiTBlock)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked"), [(5, 384, True), (3, 200, False)])
def test_token_dit_inference(monkeypatch, dtype, d_pair, A, L, masked):
    """``token_dit.update`` (the hoisted pair bias: the LayerNorm row template at the pair's width + one cuBLAS GEMM; then the d_pair 128
    step) serves the call; L 200 pads to a multiple of 8."""
    (eng,), (pt,), (ref,) = _modules(DiTBlock, d_pair=d_pair)
    eng.eval()
    single, cond, pair, mask, _ = _inputs(A, L, d_pair, False, masked)
    cond = cond[:1].expand(A, -1, -1, -1)
    x, c, z = single.to(dtype), cond.to(dtype), pair.to(dtype)  # fp32 parameters (bf16: fp32 masters, as a model holds them)
    spy = _Spy(monkeypatch, TD, "update")
    with torch.no_grad():
        got = eng(x, c, z, mask)
        with _recipe(dtype):
            base = pt(x, c, z, mask)
        truth = ref(single.double(), cond.double(), pair.double(), mask)
    assert spy.n == 1, "the fused token DiT step did not serve the call"
    _yardstick(got, base, truth, "out")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked"), [(4, 256, True), (3, 200, False)])
def test_token_dit_training(monkeypatch, dtype, d_pair, A, L, masked):
    """``token_dit_train._Block`` serves the call (bf16: the APB pair kernels; fp32: the one-pass ``pair_bias``) -- except bf16 at pair 256
    with a conditioning per sample (as here), which B200 trains on the module composition (faster there; a shared conditioning takes
    the fused block: ``test_token_dit_training_shared_cond``); every gradient no worse than the engine's own module path against the
    fp32 PyTorch block (IEEE): the yardstick of ``tests/gpu/sm90/test_token_dit_pair256_gpu.py``."""
    from settings_helpers import override_setting

    torch.manual_seed(7)
    ref = _randomize(DiTBlock(d_pair=d_pair, implementation=IT.PYTORCH).cuda())
    eng = DiTBlock(d_pair=d_pair, implementation=IT.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    eng = eng.to(dtype)
    single, cond, pair, mask, w = _inputs(A, L, d_pair, False, masked, seed=8)

    def run(m, tf32):
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            pdt = min((p.dtype for p in m.parameters()), key=lambda d: d.itemsize)
            ins = [t.detach().clone().to(pdt).requires_grad_() for t in (single, cond, pair)]
            out = m(*ins, mask)
            (out.float() * w).sum().backward()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
        res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
        res.update({n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
        m.zero_grad(set_to_none=True)
        return res

    truth = run(ref, False)
    spy = _Spy(monkeypatch, TDT._Block, "apply")
    got = run(eng, dtype is F32)
    composed = dtype is BF and d_pair == 256   # B200 bf16 pair 256, a conditioning per sample: the module composition (serves)
    assert spy.n == (0 if composed else 1), f"the {dtype} pair-{d_pair} training call took the wrong route ({spy.n} fused calls)"
    override_setting(monkeypatch, "token_dit_train", "0")
    module = run(eng, dtype is F32)
    assert set(got) == set(truth), sorted(set(truth) ^ set(got))
    for k in truth:
        ef, em = _rel(got[k], truth[k]), _rel(module[k], truth[k])
        assert torch.isfinite(got[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked", "extra"), [(4, 256, True, False), (6, 200, False, True), (48, 128, True, False)])
def test_token_dit_training_shared_cond(monkeypatch, d_pair, A, L, masked, extra):
    """bf16 training with ONE conditioning shared by the samples (expanded over them with stride 0, as a sampling step or a shared-cond
    training step hands it): the fused block computes the conditioning once (``token_dit_train._shared_cond``: [L, 6 D] tables read at row
    r % L, their row gradients summed over the samples before the conditioning GEMMs) -- at pair 256 too, where a conditioning per sample
    trains on the module composition. Every output and gradient is no worse than the same call with the conditioning materialized per
    sample (the fused per-row path; pair 256: the composition) against the fp32 PyTorch block (IEEE); the shared conditioning's gradient
    is the per-sample gradients' sum."""
    torch.manual_seed(7)
    ref = _randomize(DiTBlock(d_pair=d_pair, implementation=IT.PYTORCH).cuda())
    eng = DiTBlock(d_pair=d_pair, implementation=IT.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    single, cond, pair, mask, w = _inputs(A, L, d_pair, False, masked, seed=8)
    c0 = cond[:1].clone()
    ex = (0.5 * torch.randn(L, L, device="cuda")) if extra else None
    shared_calls = []
    real = TDT._shared_cond
    monkeypatch.setattr(TDT, "_shared_cond", lambda *a: shared_calls.append(real(*a)) or shared_calls[-1])

    def run(m, shared, dtype):
        ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, c0, pair)]
        xin = None if ex is None else ex.detach().clone().to(dtype).requires_grad_()
        c = ins[1].expand(A, -1, -1, -1) if shared else ins[1].expand(A, -1, -1, -1).contiguous()
        with torch.autocast("cuda", dtype=BF, enabled=dtype is BF):
            out = m(ins[0], c, ins[2], mask, **({} if xin is None else {"extra_pair_bias": xin}))
        (out.float() * w).sum().backward()
        res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
        if xin is not None:
            res["dextra"] = xin.grad.float()
        res.update({n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
        m.zero_grad(set_to_none=True)
        return res

    truth = run(ref, False, F32)
    spy = _Spy(monkeypatch, TDT._Block, "apply")
    got = run(eng, True, BF)
    assert spy.n == 1 and shared_calls[-1] is True, "the shared-conditioning fused block did not serve the call"
    per_row = run(eng, False, BF)
    assert shared_calls[-1] is False
    assert set(got) == set(truth) == set(per_row), sorted(set(truth) ^ set(got))
    for k in truth:
        ef, em = _rel(got[k], truth[k]), _rel(per_row[k], truth[k])
        assert torch.isfinite(got[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: shared conditioning {ef:.2e} vs per-row conditioning {em:.2e}"


# ================================================================================================ bias-only DiT (BiasOnlyDiTBlock)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked", "hoisted"), [(5, 256, True, False), (2, 512, False, False), (3, 384, True, True)])
def test_bias_only_inference_one_pair(monkeypatch, dtype, d_pair, A, L, masked, hoisted):
    """One pair for every sample: ``bias_only_dit.update`` (the pair's attention weights hoisted per pair: the LayerNorm rows + cuBLAS in
    bf16, the one-pass FMA ``pair_bias`` in fp32), or the block's precomputed bias (``pair_bias_all``)."""
    (eng,), (pt,), (ref,) = _modules(BiasOnlyDiTBlock, d_pair=d_pair)
    single, cond, pair, mask, _ = _inputs(A, L, d_pair, False, masked)
    cond = cond[:1].expand(A, -1, -1, -1)
    x, c, z = single.to(dtype), cond.to(dtype), pair.to(dtype)
    spy = _Spy(monkeypatch, fused, "update")
    with torch.no_grad(), _recipe(dtype):
        assert fused.serves(eng, x, c, z, mask)
        if hoisted:
            got = eng(x, c, None, mask, bias=pair_bias_all([eng], z)[0])
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
def test_bias_only_inference_pair_per_sample(monkeypatch, dtype, d_pair, B, L, masked):
    """A pair of its own per sample: ``update_psb``, the bias made by one pass over the pair (``pair_bias_k`` bf16: the sm_100a hoist at
    128; ``pair_bias_tc_k`` fp32)."""
    (eng,), (pt,), (ref,) = _modules(BiasOnlyDiTBlock, d_pair=d_pair)
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
@pytest.mark.parametrize(("A", "L", "masked"), [(4, 256, True), (3, 384, False)])
def test_bias_only_training_one_pair(monkeypatch, dtype, d_pair, A, L, masked):
    """One pair for every sample: ``bias_only_dit_train.block`` -- ``pair_bias`` (bf16 mma) / ``pair_bias_tc`` (TF32) and their backwards
    (at 256 the per-step pair_bias_bwd + finalize; pair_bias_bwd_fin is 128 wide)."""
    eng, pt, ref = _modules(BiasOnlyDiTBlock, d_pair=d_pair)
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
def test_bias_only_training_pair_per_sample(monkeypatch, dtype, d_pair, A, L, masked):
    """A pair of its own per sample: the block hoists its bias through the engine hoist (``pair_bias_all_psb``; at 256 the LayerNorm rows
    + cuBLAS forward, ``ln0_bwd_rows`` between two cuBLAS GEMMs backward) and runs ``block_psb`` on it."""
    eng, pt, ref = _modules(BiasOnlyDiTBlock, d_pair=d_pair)
    single, cond, pair, mask, w = _inputs(A, L, d_pair, True, masked)
    spy = _Spy(monkeypatch, train, "block_psb")
    hs = _Spy(monkeypatch, hoist, "pair_bias_all_psb")
    got = _train(eng, single, cond, pair, mask, w, dtype)
    assert spy.n == 1, "the fused per-sample training block did not serve the call"
    assert hs.n == 1, "the engine hoist did not make the per-sample bias"
    again = _train(eng, single, cond, pair, mask, w, dtype)
    assert all(torch.equal(got[k], again[k]) for k in got), "the step is not bitwise repeatable"
    base = _train(pt, single, cond, pair, mask, w, dtype)
    truth = _train(ref, single, cond, pair, mask, w, None)
    _train_yardstick(got, base, truth, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_pair", WIDTHS)
def test_bias_only_training_hoisted_stack(monkeypatch, dtype, d_pair):
    """Three blocks on the biases of ONE ``pair_bias_all``, against each block on the pair."""
    A, L = 3, 256
    eng, pt, ref = _modules(BiasOnlyDiTBlock, n=3, d_pair=d_pair)
    single, cond, pair, mask, w = _inputs(A, L, d_pair, False, True)
    spy = _Spy(monkeypatch, hoist, "pair_bias_all")
    blk = _Spy(monkeypatch, train, "block")
    got = _train(eng, single, cond, pair, mask, w, dtype, hoisted=True)
    assert spy.n == 1 and blk.n == 3, ("the fused hoist / blocks did not serve the stack", spy.n, blk.n)
    base = _train(pt, single, cond, pair, mask, w, dtype)
    truth = _train(ref, single, cond, pair, mask, w, None)
    _train_yardstick(got, base, truth, dtype)
