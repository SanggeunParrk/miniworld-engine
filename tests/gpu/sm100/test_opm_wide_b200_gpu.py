"""B200 OuterProductMean at the FoldForge widths on the sm_100a kernels (``csrc/sm100/opm_wide_sm100.cu`` through ``integrations.opm_train``):
d_msa 64 / 128 / 256 x d_pair 128 / 256 / 384 / 512 (Protenix-v2 / ESMFold2 (128, 256), OpenDDE (128, 384), IntelliFold-v2 (256, 512)), either
normalisation order, RoseTTAFold3's
projection biases, inference AND training.

Against the fp64 statements (``_variant_reference``): the engine's bf16 error held to 1.5x the bf16 statements' own (outputs) / 2x
(gradients, whose bf16 statements round fewer intermediates), with masks (an empty token, disjoint tokens), the pair residual, ragged
N / S (padded by the wrapper), bitwise run to run and inside a CUDA graph.  The projection biases are the inference op's own: the
low-level entry (``opm_train.outer_product_mean``) must apply them without any caller adding bias terms.  Needs a B200.
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

# name: (d_msa, d_pair, options)
WIDTHS = {
    "rf3_before": (64, 128, {"projection_bias": True}),
    "rf3_after": (64, 128, {"projection_bias": True, "normalize_before_proj": False}),
    "protenix2": (128, 256, {}),
    "esmfold2_after": (128, 256, {"normalize_before_proj": False}),
    "opendde_after": (128, 384, {"normalize_before_proj": False}),
    "opendde_before": (128, 384, {}),
    "rf3_wide": (128, 256, {"projection_bias": True, "normalize_before_proj": False}),
    "w64_256": (64, 256, {}),
    "w64_384_bias": (64, 384, {"projection_bias": True}),
    "w128_128": (128, 128, {}),
    "intellifold2": (256, 512, {}),                  # IntelliFold-v2: d_msa 256 (a CTA pair per prologue-backward tile), c_z 512
    "w256_after_bias": (256, 128, {"projection_bias": True, "normalize_before_proj": False}),
}


def build(d_msa, d_pair, options, dtype=torch.float32):
    torch.manual_seed(0)
    m = OuterProductMean(d_msa, d_pair, 32, implementation=ImplementationType.MINIWORLD, **options).cuda()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m.to(dtype)


def inputs(s, n, d_msa, *, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    msa = torch.randn(1, s, n, d_msa, device="cuda", generator=g)
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    mask[:, :, 3] = False                    # no valid row: count clamped to 1, and the biases must not leak into it
    mask[:, : s // 2, 5] = False
    mask[:, s // 2:, 6] = False              # tokens 5 and 6 share no row
    return msa, mask


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.fixture
def served(monkeypatch):
    override_setting(monkeypatch, "opm_train", None)
    start = opm_train.STATS["served"]
    return lambda: opm_train.STATS["served"] - start


SHAPES = [(256, 128), (200, 100), (512, 192)]


@pytest.mark.parametrize("name", list(WIDTHS))
@pytest.mark.parametrize(("s", "n"), SHAPES)
@pytest.mark.parametrize("residual", [False, True])
def test_inference_matches_fp64_statements(name, s, n, residual, served):
    d_msa, d_pair, options = WIDTHS[name]
    m = build(d_msa, d_pair, options)
    msa, mask = inputs(s, n, d_msa)
    pair = torch.randn(1, n, n, d_pair, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    gold, ref16, eng = copy.deepcopy(m).double(), copy.deepcopy(m).bfloat16(), copy.deepcopy(m).bfloat16()
    with torch.no_grad():
        want = gold._variant_reference(msa.double(), mask, None, pair.double() if residual else None)
        base = ref16._variant_reference(msa.bfloat16(), mask, None, pair.bfloat16() if residual else None)
        got = eng(msa.bfloat16(), mask, residual=pair.bfloat16() if residual else None)
    assert served() > 0, "the sm_100a kernels did not serve the call"
    assert torch.isfinite(got).all()
    assert rel(got, want) <= max(1.5 * rel(base, want), 1e-3), f"{name}: engine {rel(got, want):.2e} vs bf16 statements {rel(base, want):.2e}"


@pytest.mark.parametrize("name", list(WIDTHS))
@pytest.mark.parametrize(("s", "n"), SHAPES)
@pytest.mark.parametrize("master", ["bf16", "fp32"])
def test_training_matches_fp64_statements(name, s, n, master, served):
    """Forward and every gradient (msa, LayerNorm, both projections and their biases, the out-projection and its bias, the residual)."""
    d_msa, d_pair, options = WIDTHS[name]
    m = build(d_msa, d_pair, options)
    msa, mask = inputs(s, n, d_msa)
    g = torch.Generator(device="cuda").manual_seed(3)
    pair = torch.randn(1, n, n, d_pair, device="cuda", generator=g)
    gz = torch.randn(1, n, n, d_pair, device="cuda", generator=g) * 0.1
    gold, ref16 = copy.deepcopy(m).double(), copy.deepcopy(m).bfloat16()
    eng = copy.deepcopy(m) if master == "fp32" else copy.deepcopy(m).bfloat16()

    def run(mod, x, p, statements, autocast=False):
        x = x.clone().requires_grad_(True)
        p = p.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = mod._variant_reference(x, mask, None, p) if statements else mod(x, mask, residual=p)
        grads = torch.autograd.grad(out, (x, p, *mod.parameters()), gz.to(out.dtype))
        return out, grads

    want, wgrads = run(gold, msa.double(), pair.double(), True)
    base, bgrads = run(ref16, msa.bfloat16(), pair.bfloat16(), True)
    got, ggrads = run(eng, msa.bfloat16(), pair.bfloat16(), False, autocast=master == "fp32")
    assert served() > 0, "the sm_100a kernels did not serve the call"
    assert rel(got.float() - pair.bfloat16().float(), want - pair.double()) <= max(1.5 * rel(base.double() - pair.bfloat16().double(), want - pair.double()), 1e-3)
    names = ["msa", "pair", *(n_ for n_, _ in eng.named_parameters())]
    dtypes = [torch.bfloat16, torch.bfloat16, *(p.dtype for p in eng.parameters())]
    for nm, dt, a, b, w in zip(names, dtypes, ggrads, bgrads, wgrads, strict=True):
        assert a.dtype == dt, nm                 # each gradient in its parameter's dtype
        assert torch.isfinite(a).all(), nm
        e, r = rel(a, w), rel(b, w)
        assert e <= max(2.0 * r, 1e-2), f"{name} d{nm}: engine {e:.2e} vs bf16 statements {r:.2e}"


@pytest.mark.parametrize("name", ["rf3_before", "opendde_after", "rf3_wide", "intellifold2"])
def test_training_is_deterministic_and_graph_safe(name, served):
    d_msa, d_pair, options = WIDTHS[name]
    m = build(d_msa, d_pair, options, torch.bfloat16)
    msa, mask = inputs(512, 128, d_msa)
    msa = msa.bfloat16()
    pair = torch.randn(1, 128, 128, d_pair, device="cuda", dtype=torch.bfloat16)
    gz = torch.randn_like(pair) * 0.1
    x = msa.clone().requires_grad_(True)

    def step():
        out = m(x, mask, residual=pair)
        return (out, *torch.autograd.grad(out, (x, *m.parameters()), gz))

    first, second = step(), step()
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    torch.autograd.graph.set_override_stale_capture_stream(True)   # the eager steps above made the parameters' accumulators
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(captured, first, strict=True):
        assert torch.equal(a, b)
    assert served() > 0


@pytest.mark.parametrize(("d_msa", "d_pair", "after"), [(64, 128, False), (64, 128, True), (128, 384, True)])
def test_the_inference_op_applies_the_projection_biases_itself(d_msa, d_pair, after, monkeypatch):
    """The low-level entry (no module routing, no bias terms added by a caller) against the fp64 statements WITH the biases; the same call
    with the biases zeroed must differ, so the comparison sees them."""
    m = build(d_msa, d_pair, {"projection_bias": True, "normalize_before_proj": not after})
    with torch.no_grad():
        m.to_left.bias.mul_(10.0)
        m.to_right.bias.mul_(10.0)             # large enough that dropping them is far outside bf16 noise
    msa, mask = inputs(256, 128, d_msa)
    calls = []
    real = opm_train._forward_sm100
    monkeypatch.setattr(opm_train, "_forward_sm100", lambda *a, **k: (calls.append(k.get("pbias") is not None), real(*a, **k))[1])
    gold, eng = copy.deepcopy(m).double(), copy.deepcopy(m).bfloat16()
    with torch.no_grad():
        want = gold._variant_reference(msa.double(), mask, None, None)
        got = opm_train.outer_product_mean(eng, msa.bfloat16(), mask)
        eng.to_left.bias.zero_()
        eng.to_right.bias.zero_()
        unbiased = opm_train.outer_product_mean(eng, msa.bfloat16(), mask)
    assert calls == [True, True], calls    # the sm_100a op, handed the biases (zeroed the second time)
    assert rel(got, want) < 1e-2, rel(got, want)
    assert rel(unbiased, want) > 10 * rel(got, want)
