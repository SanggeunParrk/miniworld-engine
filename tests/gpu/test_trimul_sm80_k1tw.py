"""A100 (sm_80) TriMul inference fronts: the D256 ``k1tw_kernel`` (one CTA per token tile, LayerNorm_in statistics in-kernel) and the transposed
outgoing planes (plane (r, c) = token (c, r), so the contraction runs as A^T B).

* kernel level: k1tw's transposed planes are bit for bit the transpose of its plain planes (the same statistics and MMAs per token, only the
  store position moves), its statistics are the same bits either way, and both match ``ln_stats_rows`` + ``k1w`` (the front it replaces) to
  bf16 rounding;
* module level: the D64 outgoing output is bit for bit the same with and without transposed planes (hidden 64 / 128, the RoseTTAFold3 variant),
  and the D256 output with k1tw stays as close to the PyTorch reference as the replaced front and is deterministic, both directions, padded
  inputs (the measurers' masks), L % 128 == 0 (transposed) and not (plain planes)."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TriangleMultiplication, TriangleMultiplicationRF3

pytestmark = [pytest.mark.gpu, pytest.mark.a100]


def _module(cls, d_pair: int, outgoing: bool, d_hidden: int | None = None):
    kw = {"d_pair": d_pair, "outgoing": outgoing}
    if d_hidden is not None:
        kw["d_hidden"] = d_hidden
    m = cls(**kw).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    return m


def _run(kern, inputs):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert kern.unsupported(**inputs) is None
        return kern(**inputs).clone()


@pytest.mark.parametrize("length", [512, 640])
def test_k1tw_planes(length: int) -> None:
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    ext = sm80_wide._ext()
    m = _module(TriangleMultiplication, 256, True)
    ws = [m.to_left.weight, m.to_left_gate.weight, m.to_right.weight, m.to_right_gate.weight, m.to_gate.weight, m.to_out.weight]
    pk = sm80_wide._pack(*(w.detach() for w in ws), m.ln_pair.weight, m.ln_pair.bias, m.ln_out.weight, m.ln_out.bias, adt=torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(3)
    n, hs, eps = length, pk["hs"], m.ln_pair.eps
    z = (torch.randn((n * n, 256), generator=gen, device="cuda") * 1.5 + 0.3).bfloat16()
    mask = torch.ones(n, dtype=torch.uint8, device="cuda")
    mask[n - n // 8 :] = 0
    planes = {tr: z.new_empty((2 * hs, n * n)) for tr in (False, True)}
    st = {tr: torch.empty((n * n, 2), dtype=torch.float32, device="cuda") for tr in (False, True)}
    for tr in (False, True):
        ext.k1tw(z, pk["w1"], pk["vs"], pk["vb"], st[tr], mask, planes[tr], n, eps, tr)
    assert torch.equal(planes[True].view(2 * hs, n, n), planes[False].view(2 * hs, n, n).transpose(1, 2))
    assert torch.equal(st[True], st[False])
    # the replaced front: ln_stats_rows + k1w
    st_r, planes_r = torch.empty_like(st[False]), torch.empty_like(planes[False])
    ext.stats_rows(z, st_r, eps)
    ext.k1w_row_mask(z, pk["w1"], pk["vs"], pk["vb"], st_r, mask, planes_r, n, eps, False, False)
    torch.testing.assert_close(st[False], st_r, rtol=1e-5, atol=1e-6)
    a, b = planes[False].float(), planes_r.float()
    assert torch.equal(a == 0, b == 0)  # the masked pairs are exact zeros in both
    assert bool(((a - b).abs() <= 2.0**-7 * torch.maximum(a.abs(), b.abs()) + 1e-6).all())  # at most one bf16 rounding step apart


@pytest.mark.parametrize("length", [1024])
@pytest.mark.parametrize(
    ("cls", "d_hidden"), [(TriangleMultiplication, None), (TriangleMultiplication, 128), (TriangleMultiplicationRF3, None)]
)
def test_d64_transposed_bitwise(cls, d_hidden, length: int, monkeypatch) -> None:
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    base = _module(cls, 64, True, d_hidden)
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = base.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    kern = copy.deepcopy(base).set_backend("cuda")
    outs = []
    for flag in (False, True):
        monkeypatch.setattr(sm80_wide, "_TRANSPOSED", flag)
        outs.append(_run(kern, inputs))
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("length", [528, 640])
@pytest.mark.parametrize("outgoing", [True, False])
def test_d256_k1tw_module(outgoing: bool, length: int, monkeypatch) -> None:
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    base = _module(TriangleMultiplication, 256, outgoing)
    gen = torch.Generator(device="cuda").manual_seed(2)
    inputs = base.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    ref = _run(copy.deepcopy(base).set_backend("pytorch"), inputs).float()
    kern = copy.deepcopy(base).set_backend("cuda")
    errs = {}
    for flag in (False, True):
        monkeypatch.setattr(sm80_wide, "_K1TW", flag)
        monkeypatch.setattr(sm80_wide, "_TRANSPOSED", flag)
        out = _run(kern, inputs)
        if flag:
            assert torch.equal(out, _run(kern, inputs))  # deterministic
        errs[flag] = (out.float() - ref).abs().max().item()
    assert errs[True] <= 1.25 * errs[False] + 1e-3, errs
