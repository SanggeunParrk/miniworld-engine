"""The H100 inference OuterProductMean as ONE kernel (``opm_fused_infer``: the grouped outer product and the c_hidden^2 -> c_z projection, the products
never written to HBM), MiniWorld / RF3 widths. Against the fp64 statements (the engine's bf16 error held to 1.5x the bf16 statements' own), against
the unfused composition (cuBLAS + epilogue kernel), bitwise run to run, under a CUDA graph and ``torch.compile``.

Needs an H100 (sm_90a); skipped elsewhere.
"""

import copy

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.integrations import opm_train
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

OPTIONS = {
    "miniworld": {},
    "esmfold2_after": {"normalize_before_proj": False},
    "rf3_before": {"projection_bias": True},
    "rf3_after": {"projection_bias": True, "normalize_before_proj": False},
}


def build(options, dtype=torch.bfloat16):
    torch.manual_seed(0)
    m = OuterProductMean(64, 128, 32, implementation=ImplementationType.MINIWORLD, **options).cuda().eval()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m.to(dtype)


def inputs(s, n, *, empty_tokens=True, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    msa = torch.randn(1, s, n, 64, device="cuda", generator=g)
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    if empty_tokens:   # a token with no valid MSA row has count 0 -> clamped to 1; two tokens with disjoint rows have count 0 as well
        mask[:, :, 3] = False
        mask[:, : s // 2, 5] = False
        mask[:, s // 2 :, 6] = False
    return msa, mask


def spy_fused(monkeypatch):
    calls = []
    real = opm_train._fused_weight
    monkeypatch.setattr(opm_train, "_fused_weight", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    return calls


def rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("name", list(OPTIONS))
@pytest.mark.parametrize(("s", "n"), [(256, 64), (512, 192), (1024, 128)])
@pytest.mark.parametrize("residual", [False, True])
def test_fused_matches_fp64_statements(name, s, n, residual, monkeypatch):
    calls = spy_fused(monkeypatch)
    m = build(OPTIONS[name], torch.float32)
    msa, mask = inputs(s, n)
    g = torch.Generator(device="cuda").manual_seed(2)
    pair = torch.randn(1, n, n, 128, device="cuda", generator=g)
    gold, ref16, eng = copy.deepcopy(m).double(), copy.deepcopy(m).bfloat16(), copy.deepcopy(m).bfloat16()
    with torch.no_grad():
        want = gold._variant_reference(msa.double(), mask, None, pair.double() if residual else None)
        base = ref16._variant_reference(msa.bfloat16(), mask, None, pair.bfloat16() if residual else None)
        got = eng(msa.bfloat16(), mask, residual=pair.bfloat16() if residual else None)
    assert calls, "the fused kernel did not serve the call"
    assert torch.isfinite(got).all()
    assert rel(got, want) <= 1.5 * rel(base, want), f"engine {rel(got, want):.2e} vs bf16 statements {rel(base, want):.2e}"


@pytest.mark.parametrize("name", list(OPTIONS))
@pytest.mark.parametrize(("s", "n"), [(512, 128), (1024, 256)])
def test_fused_matches_the_unfused_composition(name, s, n, monkeypatch):
    """Same products, same bf16 rounding of them: only the order of the fp32 sums of the projection differs."""
    m = build(OPTIONS[name])
    msa, mask = inputs(s, n)
    pair = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16)
    calls = spy_fused(monkeypatch)
    with torch.no_grad():
        fused = m(msa.bfloat16(), mask, residual=pair)
        assert calls
        calls.clear()
        override_setting(monkeypatch, "force_portable_kernels", True)
        unfused = m(msa.bfloat16(), mask, residual=pair)
        assert not calls
    d = (fused.float() - unfused.float()).abs()
    assert d.max().item() <= 2e-2 * unfused.float().abs().max().item(), d.max().item()
    assert rel(fused, unfused.double()) < 5e-3


def test_fused_is_deterministic_and_graph_safe():
    m = build({})
    msa, mask = inputs(512, 128)
    msa = msa.bfloat16()
    with torch.no_grad():
        a = m(msa, mask)
        b = m(msa, mask)
        assert torch.equal(a, b)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            m(msa, mask)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out = m(msa, mask)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, a)
        msa2 = msa.clone()
        msa2.mul_(0.5)   # static inputs are refilled in place between replays
        msa.copy_(msa2)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, m(msa, mask))


def test_fused_under_torch_compile(monkeypatch):
    m = build({"projection_bias": True})
    msa, mask = inputs(256, 64)
    msa = msa.bfloat16()
    with torch.no_grad():
        eager = m(msa, mask)
        compiled = torch.compile(m, fullgraph=False)(msa, mask)
    assert torch.equal(eager, compiled)


def test_weights_updated_in_place_are_seen():
    """The permuted weight and the packed forms are cached while the weights are unchanged; an in-place update must be picked up."""
    m = build({})
    msa, mask = inputs(256, 64)
    msa = msa.bfloat16()
    with torch.no_grad():
        before = m(msa, mask)
        m.to_out.weight.mul_(2.0)
        m.to_out.bias.add_(0.5)
        after = m(msa, mask)
        ref = copy.deepcopy(m)
        assert not torch.equal(before, after)
        want = ref._variant_reference(msa, mask, None, None)
    assert rel(after, want.double()) < 2e-2


def test_refusals_run_the_unfused_composition(monkeypatch):
    """N not a multiple of 16 is padded by the caller to 64; c_z other than 128 has no fused kernel and keeps the epilogue kernel."""
    calls = spy_fused(monkeypatch)
    torch.manual_seed(0)
    m = OuterProductMean(128, 256, 32, implementation=ImplementationType.MINIWORLD).cuda().bfloat16().eval()
    msa = torch.randn(1, 256, 64, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, 256, 64, device="cuda") > 0.2
    with torch.no_grad():
        out = m(msa, mask)
    assert out.shape == (1, 64, 64, 256) and torch.isfinite(out).all()
    assert not calls
