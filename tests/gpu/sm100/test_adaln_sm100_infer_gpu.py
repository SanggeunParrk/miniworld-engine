"""B200: the AdaLN inference route ``kernels/adaln/cuda/sm100`` -- the row pass ``ln_cond_xstats`` + the tcgen05 GEMM ``adaln_gemm_sm100`` with the
gate and LN(x) in its epilogue (both staging depths), and at the atom width the one-kernel ``adaln_fused_sm100`` (both routes tested there) -- at every output-tile width, bf16 and fp32 (TF32), against an fp64 reference: within 1.5x the
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
@pytest.mark.parametrize("bn", [192, 128, 64, 32])
@pytest.mark.parametrize("fused", [None, False], ids=["route", "two_launch"])
@pytest.mark.parametrize("stg", [2, 1], ids=["stage2", "stage1"])
@pytest.mark.parametrize("fold", [None, False], ids=["fold", "nofold"])
def test_matches_the_reference_at_every_tile_width(dtype, shape, bn, fused, stg, fold):
    from miniworld_engine.kernels.adaln.cuda import sm100

    m, d, dc = shape
    if bn == 192 and (dtype is torch.bfloat16 or d % 192):
        pytest.skip("the 192-column tile is an fp32 (TF32) width of d % 192 == 0")
    if bn == 192 and fused is None and dc in (128,):
        pytest.skip("the one-kernel route does not take 192 columns")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, d, device="cuda", generator=g).to(dtype)
    c = (torch.randn(m, dc, device="cuda", generator=g) * 2 + 0.5).to(dtype)
    wc = 1 + 0.3 * torch.randn(dc, device="cuda", generator=g)
    ws = 0.05 * torch.randn(d, dc, device="cuda", generator=g)
    bs = 0.5 * torch.randn(d, device="cuda", generator=g)
    wb = 0.05 * torch.randn(d, dc, device="cuda", generator=g)
    if fused is False and shape[2] not in sm100.FUSED_KC:
        pytest.skip("the two-launch route is the only one at this width")
    if stg == 1 and (fused is None and shape[2] in sm100.FUSED_KC and dtype is torch.bfloat16):
        pytest.skip("the one-kernel route has no staging option")
    if fold is False and (dtype is torch.float32 or dc in sm100.FUSED_KC):
        pytest.skip("the fold is the bf16 token-width route only")
    saved, saved_route = dict(sm100.STAGING), sm100.F32_ROUTE
    sm100.FOLD = fold
    # fp32 calls take the cuBLAS route at the atom width in production; the tcgen05 kernels are pinned here so that every width of them stays tested
    sm100.BN_FORCE, sm100.FUSED, sm100.STAGING, sm100.F32_ROUTE = bn, fused, {False: stg, True: stg}, "tcgen05"
    try:
        assert sm100.supported(x, c) or (dtype is torch.float32 and dc in sm100.FUSED_KC)   # fp32 atoms: the module keeps the row kernels
        with torch.no_grad():
            y = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5)
            again = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5)
            yr = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5, rna=True)
    finally:
        sm100.BN_FORCE, sm100.FUSED, sm100.STAGING, sm100.F32_ROUTE, sm100.FOLD = None, None, saved, saved_route, None
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", [(5000, 128, 128), (3000, 768, 384)], ids=["atom", "token"])
@pytest.mark.parametrize("fused", [None, False], ids=["route", "two_launch"])
@pytest.mark.parametrize("f32_route", ["cublas", "tcgen05"])
def test_a_large_mean_keeps_the_statistics(dtype, shape, fused, f32_route):
    """x and cond with mean 1e3 and standard deviation 1: a one-pass E[v^2] - E[v]^2 variance in fp32 loses ~6 % of it (1e6 * 6e-8); the
    two-pass statistics keep the result within 1.5x the error of the same-dtype PyTorch statements."""
    from miniworld_engine.kernels.adaln.cuda import sm100

    m, d, dc = shape
    if fused is False and dc not in sm100.FUSED_KC:
        pytest.skip("the two-launch route is the only one at this width")
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (1e3 + torch.randn(m, d, device="cuda", generator=g)).to(dtype)
    c = (1e3 + torch.randn(m, dc, device="cuda", generator=g)).to(dtype)
    wc = 1 + 0.3 * torch.randn(dc, device="cuda", generator=g)
    ws = 0.05 * torch.randn(d, dc, device="cuda", generator=g)
    bs = 0.5 * torch.randn(d, device="cuda", generator=g)
    wb = 0.05 * torch.randn(d, dc, device="cuda", generator=g)
    if dtype is torch.bfloat16 and f32_route != "cublas":
        pytest.skip("fp32 only")
    saved_route = sm100.F32_ROUTE
    sm100.FUSED, sm100.F32_ROUTE = fused, f32_route
    try:
        with torch.no_grad():
            y = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5)
    finally:
        sm100.FUSED, sm100.F32_ROUTE = None, saved_route
    ref = _ref(x.double(), c.double(), wc.double(), ws.double(), bs.double(), wb.double())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
        pt = _ref(x, c, wc, ws, bs, wb).to(dtype)   # the same-dtype PyTorch statements (bf16: autocast, LayerNorm in fp32, bf16 GEMMs)
    assert _err(y, ref) <= 1.5 * _err(pt, ref) + 1e-4, (_err(y, ref), _err(pt, ref))


@pytest.mark.parametrize("m", [333, 5000, 40000])
@pytest.mark.parametrize("mean", [0.0, 1e3], ids=["centred", "large_mean"])
def test_the_pair_kernel_matches_the_reference(m, mean):
    """fp32 at the atom width: the CTA-pair kernel (``adaln_fused2_sm100.cu``, cta_group::2) against an fp64 reference, within 1.5x the
    error of the fp32 PyTorch statements, bitwise repeatable, the ``rna`` output on the TF32 grid; rows off the 256-row pair tile and
    enough for a second tile on some pairs."""
    from miniworld_engine.kernels.adaln.cuda import sm100

    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(m, 128, device="cuda", generator=g) + mean
    c = torch.randn(m, 128, device="cuda", generator=g) * 2 + 0.5 + mean
    wc = 1 + 0.3 * torch.randn(128, device="cuda", generator=g)
    ws = 0.05 * torch.randn(128, 128, device="cuda", generator=g)
    bs = 0.5 * torch.randn(128, device="cuda", generator=g)
    wb = 0.05 * torch.randn(128, 128, device="cuda", generator=g)
    saved = sm100.FUSED2
    sm100.FUSED2 = True
    try:
        with torch.no_grad():
            y = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5)
            again = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5)
            yr = sm100.adaln_inference(x, c, wc, ws, bs, wb, 1e-5, 1e-5, rna=True)
    finally:
        sm100.FUSED2 = saved
    assert torch.equal(y, again)
    ref = _ref(x.double(), c.double(), wc.double(), ws.double(), bs.double(), wb.double())
    with torch.no_grad():
        pt = _ref(x, c, wc, ws, bs, wb)
    assert _err(y, ref) <= 1.5 * _err(pt, ref) + 1e-6, (_err(y, ref), _err(pt, ref))
    assert int((yr.view(torch.int32) & 0x1FFF).abs().max()) == 0, "rna output off the TF32 grid"
    assert _err(yr, ref) <= 1.5 * _err(pt, ref) + 2e-4
