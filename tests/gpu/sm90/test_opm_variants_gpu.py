"""RoseTTAFold3's OuterProductMean (biased left / right projections) and Chai-1's (8 groups, product LayerNorm) on the H100 composed path,
inference, against the fp64 torch statements: the engine's bf16 error held to the bf16 statements' own."""

import copy

import pytest
import torch

from miniworld_engine.kernels.outer_product_mean import composed
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

VARIANTS = {
    "rf3_after": {"d_pair": 128, "d_hidden": 32, "projection_bias": True, "normalize_before_proj": False},
    "rf3_before": {"d_pair": 128, "d_hidden": 32, "projection_bias": True, "normalize_before_proj": True},
    "chai1": {"d_pair": 256, "d_hidden": 64, "groups": 8},
}


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize(("b", "s", "n"), [(1, 512, 128), (1, 1024, 256), (2, 256, 192)])
def test_opm_variant(name, b, s, n):
    options = dict(VARIANTS[name])
    torch.manual_seed(0)
    m = OuterProductMean(64, options.pop("d_pair"), options.pop("d_hidden"), implementation=ImplementationType.MINIWORLD, **options)
    m = m.cuda().eval()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(b, s, n, 64, device="cuda", generator=g)
    mask = torch.rand(b, s, n, device="cuda", generator=g) > 0.2
    gold = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    with torch.no_grad():
        assert composed.serves(eng, msa.bfloat16(), mask)
        want = gold._variant_reference(msa.double(), mask, None, None)
        base = ref16._variant_reference(msa.bfloat16(), mask, None, None)
        got = eng(msa.bfloat16(), mask)

    def rel(x):
        return ((x.double() - want).norm() / want.norm()).item()

    assert torch.isfinite(got).all()
    assert rel(got) <= max(1.5 * rel(base), 1e-2), f"{name}: engine {rel(got):.2e} vs bf16 statements {rel(base):.2e}"


@pytest.mark.parametrize("name", ["rf3_after", "rf3_before"])
@pytest.mark.parametrize(("s", "n", "residual"), [(512, 128, True), (1024, 256, False), (256, 192, True)])
def test_rf3_training_matches_fp64_statements(name, s, n, residual, monkeypatch):
    """RoseTTAFold3's biased projections in a grad-enabled call run the wide fused composition (the projection biases in the prologue,
    their gradients off dA / dB): outputs and every gradient (msa, residual, all parameters) against the fp64 statements, the
    engine's bf16 error held to the bf16 statements' own."""
    from miniworld_engine.integrations import opm_train

    options = dict(VARIANTS[name])
    torch.manual_seed(0)
    m = OuterProductMean(64, options.pop("d_pair"), options.pop("d_hidden"), implementation=ImplementationType.MINIWORLD, **options)
    m = m.cuda().train()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(1, s, n, 64, device="cuda", generator=g)
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    pair = torch.randn(1, n, n, 128, device="cuda", generator=g)
    dz = torch.randn(1, n, n, 128, device="cuda", generator=g)
    gold = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    calls = []
    run = opm_train.outer_product_mean
    monkeypatch.setattr(opm_train, "outer_product_mean", lambda *a, **k: (calls.append(1), run(*a, **k))[1])

    def go(mod, dtype, fn):
        x = msa.to(dtype).clone().requires_grad_()
        r = pair.to(dtype).clone().requires_grad_() if residual else None
        y = fn(mod, x, r)
        leaves = [x, *([r] if residual else []), *mod.parameters()]
        grads = torch.autograd.grad(y, leaves, dz.to(dtype))
        return y.detach(), grads

    ref_fn = lambda mod, x, r: mod._variant_reference(x, mask, None, r)   # noqa: E731
    eng_fn = lambda mod, x, r: mod(x, mask, residual=r)                   # noqa: E731
    yg, gg = go(gold, torch.float64, ref_fn)
    yb, gb = go(ref16, torch.bfloat16, ref_fn)
    ye, ge = go(eng, torch.bfloat16, eng_fn)
    assert calls, "the fused training composition did not serve the call"

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()

    assert rel(ye, yg) <= max(1.5 * rel(yb, yg), 1e-2), f"y: engine {rel(ye, yg):.2e} vs bf16 statements {rel(yb, yg):.2e}"
    names = ["msa", *(["residual"] if residual else []), *(n_ for n_, _ in m.named_parameters())]
    for nm, a, b, c in zip(names, ge, gb, gg, strict=True):
        assert rel(a, c) <= max(1.5 * rel(b, c), 2e-2), f"{nm}: engine {rel(a, c):.2e} vs bf16 statements {rel(b, c):.2e}"


@pytest.mark.parametrize(("s", "n", "masked"), [(512, 128, True), (256, 192, True), (512, 256, False)])
def test_chai1_fused_passes_match_fp64_statements(s, n, masked, monkeypatch):
    """Chai-1's grouped OPM on the fused H100 passes (kernels.outer_product_mean.chai): output and every gradient (msa, residual, all parameters)
    against the fp64 statements, the engine's bf16 error held to the bf16 statements' own; inference and training."""
    from miniworld_engine.kernels.outer_product_mean import chai

    torch.manual_seed(0)
    m = OuterProductMean(64, 256, 64, groups=8, implementation=ImplementationType.MINIWORLD).cuda().train()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(1, s, n, 64, device="cuda", generator=g)
    mask = (torch.rand(1, s, n, device="cuda", generator=g) > 0.2) if masked else None
    pair = torch.randn(1, n, n, 256, device="cuda", generator=g)
    dz = torch.randn(1, n, n, 256, device="cuda", generator=g)
    gold = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    calls = []
    fwd = chai.forward
    monkeypatch.setattr(chai, "forward", lambda *a, **k: (calls.append(1), fwd(*a, **k))[1])

    def go(mod, dtype, fn):
        x = msa.to(dtype).clone().requires_grad_()
        r = pair.to(dtype).clone().requires_grad_()
        y = fn(mod, x, r)
        grads = torch.autograd.grad(y, [x, r, *mod.parameters()], dz.to(dtype))
        return y.detach(), grads

    ref_fn = lambda mod, x, r: mod._variant_reference(x, mask, None, r)   # noqa: E731
    eng_fn = lambda mod, x, r: mod(x, mask, residual=r)                   # noqa: E731
    yg, gg = go(gold, torch.float64, ref_fn)
    yb, gb = go(ref16, torch.bfloat16, ref_fn)
    ye, ge = go(eng, torch.bfloat16, eng_fn)
    assert calls, "the fused Chai-1 passes did not serve the call"

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()

    assert rel(ye, yg) <= max(1.5 * rel(yb, yg), 1e-2), f"y: engine {rel(ye, yg):.2e} vs bf16 statements {rel(yb, yg):.2e}"
    names = ["msa", "residual", *(n_ for n_, _ in m.named_parameters())]
    for nm, a, b, c in zip(names, ge, gb, gg, strict=True):
        assert rel(a, c) <= max(1.5 * rel(b, c), 2e-2), f"{nm}: engine {rel(a, c):.2e} vs bf16 statements {rel(b, c):.2e}"
    with torch.no_grad():                                                  # inference
        yi = eng(msa.bfloat16(), mask, residual=pair.bfloat16())
        assert rel(yi, yg) <= max(1.5 * rel(yb, yg), 1e-2)
