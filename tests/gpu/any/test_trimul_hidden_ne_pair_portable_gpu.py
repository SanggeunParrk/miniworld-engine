"""TriMul with ``d_hidden != d_pair`` on the portable kernels (Protenix / FoldForge template TriMul: pair 64, hidden 128), forward and
gradients, unidirectional (outgoing / incoming) and bidirectional, against the module's fp64 PyTorch statements. The bf16 engine error is
held to the bf16 PyTorch statements' own (x 1.5, floor 3e-3). Masks (a different one per sample) and a fixed drop_row scale are in
every training case. ``portable_on`` steps the H100 CUDA kernels aside; the refusal must not fire for a served width.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from portable_helpers import portable_on, rel, report

from miniworld_engine.modules import BidirectionalTriangleMultiplication, TriangleMultiplication
from miniworld_engine.modules.dispatch import KernelRefusedWarning
from miniworld_engine.modules.exceptions import ImplementationType as I

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF = "cuda", torch.bfloat16

WIDTHS = [(64, 128), (64, 192), (128, 64), (128, 256), (64, 32), (32, 128)]   # (d_pair, d_hidden)
KINDS = ["out", "in", "bidir"]


@pytest.fixture(autouse=True)
def portable():
    with portable_on():
        yield


def _build(kind, d_pair, d_hidden, implementation):
    torch.manual_seed(0)
    if kind == "bidir":
        m = BidirectionalTriangleMultiplication(d_pair, d_hidden=d_hidden, p_drop=0.25, implementation=implementation)
    else:
        m = TriangleMultiplication(d_pair, d_hidden=d_hidden, outgoing=kind == "out", p_drop=0.25, implementation=implementation)
    m = m.to(DEV)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if name.endswith("ln_pair.weight") or name.endswith("ln_out.weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.1 * torch.randn_like(p))
    return m


def _fix_dropout(m, ds):
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)


def _spy(monkeypatch, cls):
    calls = []
    fn = cls._forward_triton

    def wrapped(self, *a, **k):
        calls.append(1)
        return fn(self, *a, **k)

    monkeypatch.setattr(cls, "_forward_triton", wrapped)
    return calls


# the corners of the served box (contract.hidden_refusal): the widest pair, the widest hidden, the narrowest of each
CORNERS = [(256, 512, "out"), (256, 512, "bidir"), (16, 512, "in"), (256, 16, "out"), (16, 16, "bidir")]


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize(("widths", "kind", "b", "n"),
                         [(w, k, b, n) for w in WIDTHS for k in KINDS for b, n in ((1, 160), (2, 200))]
                         + [((a, h), k, 1, 128) for a, h, k in CORNERS],
                         ids=[f"D{w[0]}H{w[1]}-{k}-B{b}L{n}" for w in WIDTHS for k in KINDS for b, n in ((1, 160), (2, 200))]
                         + [f"corner-D{a}H{h}-{k}" for a, h, k in CORNERS])
def test_hidden_ne_pair_matches_pytorch(monkeypatch, widths, kind, training, b, n):
    d_pair, d_hidden = widths
    if kind == "bidir" and training and d_hidden == 32:
        pytest.skip("covered by the unidirectional cases")
    m = _build(kind, d_pair, d_hidden, I.MINIWORLD)
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(b, n, n, d_pair, device=DEV, generator=g)
    mask = torch.rand(b, n, device=DEV, generator=g) > 0.12
    ds = (torch.rand(b, 1, n, d_pair, device=DEV, generator=g) > 0.25).float() / 0.75
    dy = torch.randn(b, n, n, d_pair, device=DEV, generator=g)

    def reference(dtype):
        r = _build(kind, d_pair, d_hidden, I.PYTORCH)
        r.load_state_dict(m.state_dict())
        r = r.to(dtype)
        _fix_dropout(r, ds)
        return r.train(training)

    def run(module, dtype):
        xx = x.to(dtype).requires_grad_(training)
        if training:
            out = module(xx, mask)
            params = list(module.parameters())
            return out, torch.autograd.grad(out, [xx, *params], dy.to(dtype))
        with torch.no_grad():
            return module(xx, mask), ()

    gold = reference(torch.float64)
    ref16 = reference(BF)
    out_gold, g_gold = run(gold, torch.float64)
    out_16, g_16 = run(ref16, BF)
    eng = m.to(BF).train(training)
    _fix_dropout(eng, ds)
    calls = _spy(monkeypatch, type(eng))
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelRefusedWarning)
        out, g_eng = run(eng, BF)
    assert len(calls) >= 1, "the portable path did not run"
    assert torch.isfinite(out).all()
    report(f"trimul_hidden_ne_pair {kind} D{d_pair}H{d_hidden} {'train' if training else 'infer'} B{b}L{n}", out, out_16, out_gold, g_eng, g_16, g_gold)
    assert rel(out, out_gold) <= max(1.5 * rel(out_16, out_gold), 3e-3), (rel(out, out_gold), rel(out_16, out_gold))
    for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_16, strict=True)):
        assert rel(a, r) <= max(1.5 * rel(c, r), 3e-3), (i, rel(a, r), rel(c, r))
        assert a.dtype == c.dtype


@pytest.mark.parametrize("kind", KINDS)
def test_out_of_range_widths_refuse_with_a_warning(kind):
    """A width the portable front does not serve warns once (KernelRefusedWarning) and runs the PyTorch statements; the answer is right."""
    from miniworld_engine.modules import dispatch

    d_pair, d_hidden = 64, 1024
    m = _build(kind, d_pair, d_hidden, I.MINIWORLD).to(BF).eval()
    r = _build(kind, d_pair, d_hidden, I.PYTORCH).to(BF).eval()
    r.load_state_dict(m.state_dict())
    x = torch.randn(1, 64, 64, d_pair, device=DEV, dtype=BF)
    dispatch._refused_warned.clear()
    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        got = m(x)
    assert any(issubclass(w.category, KernelRefusedWarning) for w in caught)
    assert rel(got, r(x)) < 1e-2
