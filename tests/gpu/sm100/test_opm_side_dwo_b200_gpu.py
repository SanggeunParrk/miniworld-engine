"""B200 OuterProductMean training: dWo on a side stream beside the dA / dB GEMMs.

From L128 the training backward runs dWo (``opm_dwo`` / ``opm_wdwo`` with a CTA budget, ``opm_train._DWO_SIDE_CTAS``) on a side stream while
cuBLAS computes dA and dB on the remaining SMs (``opm_train._sm_carveout``).  Held here: the fp64 statements' accuracy bars (as
``test_opm_wide_b200_gpu``), every gradient but dWo bitwise equal to the serial order and dWo equal up to its fp32 split order, bitwise run
to run and inside a CUDA graph.  Needs a B200.
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

WIDTHS = {"miniworld": (64, 128, {}), "protenix2": (128, 256, {}), "rf3_after": (64, 128, {"projection_bias": True, "normalize_before_proj": False})}


def build(d_msa, d_pair, options):
    torch.manual_seed(0)
    m = OuterProductMean(d_msa, d_pair, 32, implementation=ImplementationType.MINIWORLD, **options).cuda()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


def inputs(s, n, d_msa, d_pair):
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(1, s, n, d_msa, device="cuda", generator=g)
    mask = torch.rand(1, s, n, device="cuda", generator=g) > 0.2
    mask[:, :, 3] = False
    pair = torch.randn(1, n, n, d_pair, device="cuda", generator=g)
    gz = torch.randn(1, n, n, d_pair, device="cuda", generator=g) * 0.1
    return msa, mask, pair, gz


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.fixture
def served(monkeypatch):
    override_setting(monkeypatch, "opm_train", None)
    start = opm_train.STATS["served"]
    return lambda: opm_train.STATS["served"] - start


def side_count(monkeypatch):
    calls = []
    real = opm_train._side_stream
    monkeypatch.setattr(opm_train, "_side_stream", lambda d: (calls.append(1), real(d))[1])
    return calls


def engine_step(m, msa, mask, pair, gz, master):
    eng = m if master == "fp32" else m.bfloat16()
    x = msa.bfloat16().clone().requires_grad_(True)
    p = pair.bfloat16().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=master == "fp32"):
        out = eng(x, mask, residual=p)
    return (out, *torch.autograd.grad(out, (x, p, *eng.parameters()), gz.bfloat16()))


@pytest.mark.parametrize("name", list(WIDTHS))
@pytest.mark.parametrize("n", [256, 320])
@pytest.mark.parametrize("master", ["bf16", "fp32"])
def test_side_stream_training_matches_fp64_statements(name, n, master, served, monkeypatch):
    d_msa, d_pair, options = WIDTHS[name]
    if d_pair not in opm_train._DWO_SIDE_CTAS:
        pytest.skip(f"c_z {d_pair} keeps dWo after the GEMMs")
    calls = side_count(monkeypatch)
    m = build(d_msa, d_pair, options)
    msa, mask, pair, gz = inputs(256, n, d_msa, d_pair)
    gold, ref16 = copy.deepcopy(m).double(), copy.deepcopy(m).bfloat16()

    def statements(mod, x, p):
        x = x.clone().requires_grad_(True)
        p = p.clone().requires_grad_(True)
        out = mod._variant_reference(x, mask, None, p)
        return (out, *torch.autograd.grad(out, (x, p, *mod.parameters()), gz.to(out.dtype)))

    want = statements(gold, msa.double(), pair.double())
    base = statements(ref16, msa.bfloat16(), pair.bfloat16())
    got = engine_step(copy.deepcopy(m), msa, mask, pair, gz, master)
    assert served() > 0 and calls, "the side-stream dWo did not serve the call"
    p16 = pair.bfloat16()
    assert rel(got[0].float() - p16.float(), want[0] - pair.double()) <= max(1.5 * rel(base[0].double() - p16.double(), want[0] - pair.double()), 1e-3)
    names = ["msa", "pair", *(n_ for n_, _ in m.named_parameters())]
    for nm, a, b, w in zip(names, got[1:], base[1:], want[1:], strict=True):
        assert torch.isfinite(a).all(), nm
        e, r = rel(a, w), rel(b, w)
        assert e <= max(2.0 * r, 1e-2), f"{name} d{nm}: engine {e:.2e} vs bf16 statements {r:.2e}"


@pytest.mark.parametrize("name", list(WIDTHS))
@pytest.mark.parametrize("master", ["bf16", "fp32"])
def test_side_stream_equals_the_serial_order(name, master, served, monkeypatch):
    """Every gradient but dWo bitwise as with dWo after the GEMMs on every SM; dWo up to the fp32 order of its K splits."""
    d_msa, d_pair, options = WIDTHS[name]
    if d_pair not in opm_train._DWO_SIDE_CTAS:
        pytest.skip(f"c_z {d_pair} keeps dWo after the GEMMs")
    m = build(d_msa, d_pair, options)
    msa, mask, pair, gz = inputs(256, 384, d_msa, d_pair)
    calls = side_count(monkeypatch)
    side = engine_step(copy.deepcopy(m), msa, mask, pair, gz, master)
    assert calls
    monkeypatch.setattr(opm_train, "_DWO_SIDE_CTAS", {})
    serial = engine_step(copy.deepcopy(m), msa, mask, pair, gz, master)
    names = ["out", "msa", "pair", *(n_ for n_, _ in m.named_parameters())]
    for nm, a, b in zip(names, side, serial, strict=True):
        if nm == "to_out.weight":
            # fp32: 4 K splits instead of 37 -- the split order, and the tensor core's fp32 accumulation over 9x longer chains per CTA (it
            # shrinks magnitudes a little: rel 3.6e-5 / 3.7e-5 measured at L384, 1.6e-4 of the largest entry at L768; the bf16 statements'
            # own error is ~1e-3); bf16 parameters: dWo also rounded to bf16 after the sum
            assert rel(a, b) <= (2e-4 if master == "fp32" else 4e-3), nm
        else:
            assert torch.equal(a, b), nm


@pytest.mark.parametrize("master", ["bf16", "fp32"])
def test_side_stream_is_deterministic_and_graph_safe(master, served, monkeypatch):
    calls = side_count(monkeypatch)
    m = build(64, 128, {})
    msa, mask, pair, gz = inputs(256, 384, 64, 128)
    eng = m if master == "fp32" else m.bfloat16()
    x = msa.bfloat16().clone().requires_grad_(True)
    p16, g16 = pair.bfloat16(), gz.bfloat16()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=master == "fp32"):
            out = eng(x, mask, residual=p16)
        return (out, *torch.autograd.grad(out, (x, *eng.parameters()), g16))

    first, second = step(), step()
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(captured, first, strict=True):
        assert torch.equal(a, b)
    assert calls and served() > 0

