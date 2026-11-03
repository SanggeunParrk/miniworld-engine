"""H100: OuterProductMean's ``count_offset`` (AlphaFold 3 / Protenix: divisor ``count + 1e-3``) on the
hand-CUDA paths, inference and training, against the fp64 PyTorch statements (``_variant_reference``), the engine's bf16 error held to the bf16
statements' own. Only registry-declared rows (``registry_module.csv``): MiniWorld (64 / 128), Protenix-v2 / ESMFold2 (128 / 256), OpenDDE
(128 / 384), RoseTTAFold3 (64 / 128 with projection biases), at their lengths. A served call must not warn ``KernelRefusedWarning``; the default
call (no option) stays bit-identical to the clamped-count kernel (``test_opm_default_divisor_is_unchanged``)."""

import copy
import warnings

import pytest
import torch

from miniworld_engine.kernels.integrations import opm_train
from miniworld_engine.kernels.outer_product_mean import composed
from miniworld_engine.modules.dispatch import KernelRefusedWarning
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.outer_product import OuterProductMean

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

#: (d_msa, d_pair, projection_bias): the declared rows
ROWS = {
    "miniworld": (64, 128, False),
    "protenix_v2": (128, 256, False),
    "opendde": (128, 384, False),
    "rf3": (64, 128, True),
}
OPTIONS = {"count_offset": {"count_offset": 1e-3}}


def _build(row, option, before, train):
    d_msa, d_pair, bias = ROWS[row]
    torch.manual_seed(0)
    m = OuterProductMean(d_msa, d_pair, 32, projection_bias=bias, normalize_before_proj=before,
                         implementation=ImplementationType.MINIWORLD, **OPTIONS[option])
    m = (m.cuda().train() if train else m.cuda().eval())
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


def _spy(monkeypatch):
    calls = []
    for mod, name in ((opm_train, "outer_product_mean"), (composed, "biased_wide"), (composed, "biased_on_fused"), (composed, "inference")):
        fn = getattr(mod, name)
        monkeypatch.setattr(mod, name, lambda *a, _f=fn, _n=name, **k: (calls.append(_n), _f(*a, **k))[1])
    return calls


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("before", [True, False], ids=["before", "after"])
@pytest.mark.parametrize("option", list(OPTIONS))
@pytest.mark.parametrize("row", list(ROWS))
@pytest.mark.parametrize(("s", "n"), [(512, 128), (1024, 256), (64, 128)])
def test_opm_divisor_inference(row, option, before, s, n, monkeypatch):
    m = _build(row, option, before, train=False)
    d_msa = ROWS[row][0]
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(1, s, n, d_msa, device="cuda", generator=g)
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    mask[..., n - 3:] = False                      # a few tokens no row covers: count 0, the divisor is the offset / S
    pair = torch.randn(1, n, n, ROWS[row][1], device="cuda", generator=g)
    gold = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    calls = _spy(monkeypatch)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        want = gold._variant_reference(msa.double(), mask, None, pair.double())
        base = ref16._variant_reference(msa.bfloat16(), mask, None, pair.bfloat16())
        got = eng(msa.bfloat16(), mask, residual=pair.bfloat16())
    assert not any(issubclass(w.category, KernelRefusedWarning) for w in caught), [str(w.message) for w in caught]
    assert calls, f"{row} {option}: no CUDA path served the call"
    assert torch.isfinite(got).all()
    assert _rel(got, want) <= max(1.5 * _rel(base, want), 1e-2), f"engine {_rel(got, want):.2e} vs bf16 statements {_rel(base, want):.2e}"


@pytest.mark.parametrize("before", [True, False], ids=["before", "after"])
@pytest.mark.parametrize("option", list(OPTIONS))
@pytest.mark.parametrize("row", list(ROWS))
@pytest.mark.parametrize(("s", "n", "residual"), [(512, 128, True), (1024, 256, False)])
def test_opm_divisor_training(row, option, before, s, n, residual, monkeypatch):
    """Output and every gradient (msa, residual, all parameters) against the fp64 statements; the backward reads the divisor off the saved table."""
    m = _build(row, option, before, train=True)
    d_msa, d_pair, _ = ROWS[row]
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(1, s, n, d_msa, device="cuda", generator=g)
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    pair = torch.randn(1, n, n, d_pair, device="cuda", generator=g)
    dz = torch.randn(1, n, n, d_pair, device="cuda", generator=g)
    gold = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    calls = _spy(monkeypatch)

    def go(mod, dtype, fn):
        x = msa.to(dtype).clone().requires_grad_()
        r = pair.to(dtype).clone().requires_grad_() if residual else None
        y = fn(mod, x, r)
        grads = torch.autograd.grad(y, [x, *([r] if residual else []), *mod.parameters()], dz.to(dtype))
        return y.detach(), grads

    ref_fn = lambda mod, x, r: mod._variant_reference(x, mask, None, r)
    eng_fn = lambda mod, x, r: mod(x, mask, residual=r)
    yg, gg = go(gold, torch.float64, ref_fn)
    yb, gb = go(ref16, torch.bfloat16, ref_fn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ye, ge = go(eng, torch.bfloat16, eng_fn)
    assert not any(issubclass(w.category, KernelRefusedWarning) for w in caught), [str(w.message) for w in caught]
    assert calls, f"{row} {option}: no CUDA path served the training call"
    assert _rel(ye, yg) <= max(1.5 * _rel(yb, yg), 1e-2), f"y: engine {_rel(ye, yg):.2e} vs bf16 statements {_rel(yb, yg):.2e}"
    names = ["msa", *(["residual"] if residual else []), *(n_ for n_, _ in m.named_parameters())]
    for nm, a, b, c in zip(names, ge, gb, gg, strict=True):
        assert _rel(a, c) <= max(1.5 * _rel(b, c), 2e-2), f"{nm}: engine {_rel(a, c):.2e} vs bf16 statements {_rel(b, c):.2e}"


@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
def test_opm_default_divisor_is_unchanged(train):
    """No option: the kernels' divisor scalars are (1, 0), i.e. fmaxf(count, 1) + 0, the old clamp (by construction; the pre-existing OPM GPU tests
    pin the numbers). Two instances with the options off give bit-equal outputs, and both differ from the ``count_offset`` one on a call that has
    zero-count pairs."""
    n, s = 128, 512
    a = _build("miniworld", "count_offset", True, train=False)
    a.count_offset = None
    b = copy.deepcopy(a)
    c = copy.deepcopy(a)
    c.count_offset = 1e-3
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(1, s, n, 64, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    mask[..., :2] = False
    a, b, c = (x.bfloat16() for x in (a, b, c))
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        ya, yb, yc = a(msa, mask), b(msa, mask), c(msa, mask)
    assert torch.equal(ya, yb)
    assert not torch.equal(ya, yc)
