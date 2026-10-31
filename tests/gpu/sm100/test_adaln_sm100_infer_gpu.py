"""B200: the AdaLN inference route ``kernels/adaln/cuda/sm100`` -- the row pass ``ln_cond_xstats`` + the tcgen05 GEMM ``adaln_gemm_sm100`` with the
gate and LN(x) in its epilogue, and at the atom width the one-kernel ``adaln_fused_sm100`` (both routes tested there) -- at every output-tile width, bf16 and fp32 (TF32), against an fp64 reference: within 1.5x the
error of the same-dtype PyTorch statements, bitwise repeatable, and the fp32 ``rna`` output on the TF32 grid. Row counts off the 128-row tile and
large enough for a second wave of the persistent grid."""
from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 only"),
]

SHAPES = [(333, 768, 384), (640, 768, 768), (1000, 128, 128), (40000, 768, 384), (40000, 128, 128)]


@pytest.fixture(autouse=True)
def _tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


def _ref(x, c, wc, ws, bs, wb, eps=1e-5):
    xn = torch.nn.functional.layer_norm(x, x.shape[-1:], eps=eps)
    cn = torch.nn.functional.layer_norm(c, c.shape[-1:], eps=eps) * wc
    return torch.sigmoid(cn @ ws.t() + bs) * xn + cn @ wb.t()


def _err(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"m{m}-d{d}-c{c}" for m, d, c in SHAPES])
@pytest.mark.parametrize("bn", [128, 64, 32])
@pytest.mark.parametrize("fused", [None, False], ids=["route", "two_launch"])
def test_matches_the_reference_at_every_tile_width(dtype, shape, bn, fused):
    from miniworld_engine.kernels.adaln.cuda import sm100

    if bn == 32 and dtype is torch.bfloat16:
        pytest.skip("bf16 output groups are 64 columns")
    m, d, dc = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, d, device="cuda", generator=g).to(dtype)
    c = (torch.randn(m, dc, device="cuda", generator=g) * 2 + 0.5).to(dtype)
    wc = 1 + 0.3 * torch.randn(dc, device="cuda", generator=g)
    ws = 0.05 * torch.randn(d, dc, device="cuda", generator=g)
    bs = 0.5 * torch.randn(d, device="cuda", generator=g)
    wb = 0.05 * torch.randn(d, dc, device="cuda", generator=g)
    if fused is False and shape[2] not in sm100.FUSED_KC:
        pytest.skip("the two-launch route is the only one at this width")
    sm100.BN_FORCE, sm100.FUSED = bn, fused
    try:
        assert sm100.supported(x, c) or (dtype is torch.float32 and dc in sm100.FUSED_KC)   # fp32 atoms: the module keeps the row kernels
        with torch.no_grad():
            y = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5)
            again = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5)
            yr = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5, rna=True)
    finally:
        sm100.BN_FORCE, sm100.FUSED = None, None
    assert y.dtype == dtype and y.shape == x.shape
    assert torch.equal(y, again)
    ref = _ref(x.double(), c.double(), wc.double(), ws.double(), bs.double(), wb.double())
    with torch.no_grad():
        pt = _ref(x.float(), c.float(), wc, ws, bs, wb) if dtype is torch.float32 else _ref(x, c, wc.to(dtype), ws.to(dtype), bs.to(dtype), wb.to(dtype))
    assert _err(y, ref) <= 1.5 * _err(pt.to(dtype), ref) + 1e-6, (_err(y, ref), _err(pt.to(dtype), ref))
    if dtype is torch.float32:
        bits = yr.view(torch.int32) & 0x1FFF
        assert int(bits.abs().max()) == 0, "rna output off the TF32 grid"
        assert _err(yr, ref) <= 1.5 * _err(pt, ref) + 2e-4
