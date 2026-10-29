"""Chai-1's grouped OuterProductMean (8 groups of 8, product LayerNorm) on the B200 kernels (``kernels.outer_product_mean.chai`` ->
``opm_wide_sm100.cu``: tcgen05 prologue / epilogue / prologue backward, the product-norm backward), inference and training.

Output and every gradient (msa, residual, all parameters) against the fp64 statements, the engine's bf16 error held to 1.5x the bf16
statements' own (2e-2 floor, the H100 test's bar), with and without a mask, ragged length / depth (padded by the wrapper), c_z 256 (Chai-1)
and 128, bf16 and fp32 master parameters; bitwise run to run and inside a CUDA graph.  Needs a B200.
"""

import copy

import pytest
import torch

from miniworld_engine.kernels.outer_product_mean import chai
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


def build(d_pair=256):
    torch.manual_seed(0)
    m = OuterProductMean(64, d_pair, 64, groups=8, implementation=ImplementationType.MINIWORLD).cuda().train()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.fixture
def served(monkeypatch):
    calls = []
    fwd = chai.forward
    monkeypatch.setattr(chai, "forward", lambda *a, **k: (calls.append(1), fwd(*a, **k))[1])
    return calls


@pytest.mark.parametrize(("s", "n", "masked"), [(512, 128, True), (256, 192, True), (512, 256, False), (200, 100, True), (130, 64, False)])
@pytest.mark.parametrize("d_pair", [256, 128])
@pytest.mark.parametrize("master", ["bf16", "fp32"])
def test_chai1_matches_fp64_statements(s, n, masked, d_pair, master, served):
    m = build(d_pair)
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(1, s, n, 64, device="cuda", generator=g)
    mask = (torch.rand(1, s, n, device="cuda", generator=g) > 0.2) if masked else None
    if masked:
        mask[:, :, 3] = False                    # a token with no valid row: its products are all zero
    pair = torch.randn(1, n, n, d_pair, device="cuda", generator=g)
    dz = torch.randn(1, n, n, d_pair, device="cuda", generator=g)
    gold = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = copy.deepcopy(m) if master == "fp32" else copy.deepcopy(m).to(torch.bfloat16)

    def go(mod, dtype, fn, autocast=False):
        x = msa.to(dtype).clone().requires_grad_()
        r = pair.to(dtype).clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            y = fn(mod, x, r)
        grads = torch.autograd.grad(y, [x, r, *mod.parameters()], dz.to(y.dtype))
        return y.detach(), grads

    ref_fn = lambda mod, x, r: mod._variant_reference(x, mask, None, r)
    eng_fn = lambda mod, x, r: mod(x, mask, residual=r)
    yg, gg = go(gold, torch.float64, ref_fn)
    yb, gb = go(ref16, torch.bfloat16, ref_fn)
    ye, ge = go(eng, torch.bfloat16, eng_fn, autocast=master == "fp32")
    assert served, "the Chai-1 kernels did not serve the call"
    assert rel(ye, yg) <= max(1.5 * rel(yb, yg), 1e-2), f"y: engine {rel(ye, yg):.2e} vs bf16 statements {rel(yb, yg):.2e}"
    names = ["msa", "residual", *(n_ for n_, _ in m.named_parameters())]
    dtypes = [torch.bfloat16, torch.bfloat16, *(p.dtype for p in eng.parameters())]
    for nm, dt, a, b, c in zip(names, dtypes, ge, gb, gg, strict=True):
        assert a.dtype == dt, nm
        assert torch.isfinite(a).all(), nm
        assert rel(a, c) <= max(1.5 * rel(b, c), 2e-2), f"{nm}: engine {rel(a, c):.2e} vs bf16 statements {rel(b, c):.2e}"
    served.clear()
    with torch.no_grad():                                                  # inference
        yi = eng(msa.bfloat16(), mask, residual=pair.bfloat16())
    assert served
    assert rel(yi, yg) <= max(1.5 * rel(yb, yg), 1e-2)


def test_chai1_training_is_deterministic_and_graph_safe(served):
    m = build().to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(4)
    msa = torch.randn(1, 512, 128, 64, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, 512, 128, device="cuda", generator=g) > 0.2
    pair = torch.randn(1, 128, 128, 256, device="cuda", generator=g).bfloat16()
    dz = torch.randn_like(pair) * 0.1
    x = msa.clone().requires_grad_(True)

    def step():
        out = m(x, mask, residual=pair)
        return (out, *torch.autograd.grad(out, (x, *m.parameters()), dz))

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
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(captured, first, strict=True):
        assert torch.equal(a, b)
    assert served
