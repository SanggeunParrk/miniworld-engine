"""B200 OuterProductMean: the fused inference kernel (``opm_fused_infer``, csrc/sm100/opm_sm100.cu: the grouped outer product and the
c_hidden^2 -> c_z projection in one kernel on CTA pairs, the products never written to HBM) and the ``normalize_before_proj=False`` order
(ESMFold2's, and AF3's as released: the count divides the projection and its bias) on the B200 path, inference and training.

Against the fp64 statements (the engine's bf16 error held to 1.5x the bf16 statements' own), against the unfused composition
(cuBLAS + the epilogue kernel), with masks (empty tokens, disjoint tokens), the pair residual, ragged N / S (the wrapper pads N to 64,
S to 256), bitwise run to run, in a CUDA graph.  Needs a B200 (sm_100a); skipped elsewhere.
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
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="B200 (sm_100a) required",
    ),
]

OPTIONS = {
    "af3": {},
    "after": {"normalize_before_proj": False},
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


def inputs(s, n, *, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    msa = torch.randn(1, s, n, 64, device="cuda", generator=g)
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    # a token with no valid MSA row has count 0 -> clamped to 1; two tokens with disjoint rows have count 0 as well
    mask[:, :, 3] = False
    mask[:, : s // 2, 5] = False
    mask[:, s // 2 :, 6] = False
    return msa, mask


@pytest.fixture
def fused_everywhere(monkeypatch):
    """Route every (N, S) cell to the fused kernel and count its calls (`_fused_weight` runs only on that path)."""
    calls = []
    real = opm_train._fused_weight
    monkeypatch.setattr(opm_train, "_fused_weight", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    monkeypatch.setattr(opm_train, "_fused_sm100_enabled", lambda s, n: True)
    return calls


def rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("name", list(OPTIONS))
@pytest.mark.parametrize(("s", "n"), [(256, 64), (200, 100), (1024, 128), (512, 320)])
@pytest.mark.parametrize("residual", [False, True])
def test_fused_matches_fp64_statements(name, s, n, residual, fused_everywhere):
    m = build(OPTIONS[name], torch.float32)
    msa, mask = inputs(s, n)
    g = torch.Generator(device="cuda").manual_seed(2)
    pair = torch.randn(1, n, n, 128, device="cuda", generator=g)
    gold, ref16, eng = copy.deepcopy(m).double(), copy.deepcopy(m).bfloat16(), copy.deepcopy(m).bfloat16()
    with torch.no_grad():
        want = gold._variant_reference(msa.double(), mask, None, pair.double() if residual else None)
        base = ref16._variant_reference(msa.bfloat16(), mask, None, pair.bfloat16() if residual else None)
        got = eng(msa.bfloat16(), mask, residual=pair.bfloat16() if residual else None)
    assert fused_everywhere, "the fused kernel did not serve the call"
    assert torch.isfinite(got).all()
    assert rel(got, want) <= 1.5 * rel(base, want), f"engine {rel(got, want):.2e} vs bf16 statements {rel(base, want):.2e}"


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize(("s", "n"), [(512, 128), (1024, 256)])
def test_fused_matches_the_unfused_composition(after, s, n, monkeypatch):
    """Same bf16 products: only the order of the fp32 sums of the projection differs."""
    m = build({"normalize_before_proj": not after})
    msa, mask = inputs(s, n)
    pair = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16)
    calls = []
    real = opm_train._fused_weight
    monkeypatch.setattr(opm_train, "_fused_weight", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    with torch.no_grad():
        monkeypatch.setattr(opm_train, "_fused_sm100_enabled", lambda s_, n_: True)
        fused = m(msa.bfloat16(), mask, residual=pair)
        assert calls
        calls.clear()
        monkeypatch.setattr(opm_train, "_fused_sm100_enabled", lambda s_, n_: False)
        unfused = m(msa.bfloat16(), mask, residual=pair)
        assert not calls
    d = (fused.float() - unfused.float()).abs()
    assert d.max().item() <= 2e-2 * unfused.float().abs().max().item(), d.max().item()
    assert rel(fused, unfused.double()) < 5e-3


def test_fused_is_deterministic_and_graph_safe(fused_everywhere):
    m = build({})
    msa, mask = inputs(512, 192)
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
        msa.mul_(0.5)   # static inputs are refilled in place between replays
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, m(msa, mask))


@pytest.mark.parametrize(("s", "n"), [(256, 128), (200, 100)])
def test_after_order_training_matches_the_statements(s, n, monkeypatch):
    """normalize_before_proj=False on the B200 training path (forward epilogue + dbias = sum dz / n) against the PyTorch statements."""
    override_setting(monkeypatch, "opm_train", None)
    m = build({"normalize_before_proj": False})
    ref = copy.deepcopy(m)
    msa, mask = inputs(s, n)
    msa = msa.bfloat16()
    gz = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    pair = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16)
    served = opm_train.STATS["served"]
    x = msa.clone().requires_grad_(True)
    out = m(x, mask, residual=pair)
    assert opm_train.STATS["served"] > served, "the B200 path did not serve the call"
    grads = torch.autograd.grad(out, (x, *m.parameters()), gz)
    xr = msa.clone().requires_grad_(True)
    want = ref._variant_reference(xr, mask, None, pair)
    wgrads = torch.autograd.grad(want, (xr, *ref.parameters()), gz)
    assert rel(out.float() - pair.float(), (want.float() - pair.float()).double()) < 1e-2
    for (name, _), a, b in zip([("msa", None), *m.named_parameters()], grads, wgrads, strict=True):
        assert rel(a.float(), b.double()) < 2e-2, name
