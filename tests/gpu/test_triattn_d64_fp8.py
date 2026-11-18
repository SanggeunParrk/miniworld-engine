"""EXPERIMENTAL e4m3 PV of the 64-channel-head wgmma core (``triattn_d64_sm90.cu`` -DTE_F8PV=1; Chai-1's trunk / confidence triangle attention,
``h100_sm90.FP8_PV_D64``, docs/fp8.md "Chai-1 TriAttn d64"): the core against the fp16 core, the module route against the fp16 route, and the switch off = default bytes."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine import fp8
from miniworld_engine.kernels.triangle_attention.cuda import h100_dh as dh
from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
from miniworld_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _init(m, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif n.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) core")
@pytest.mark.parametrize("variant", ["random", "vconst", "uniform"])
def test_core_matches_fp16_core(variant: str) -> None:
    torch.manual_seed(0)
    B, S, H, D = 1, 256, 4, 64
    proj = (0.5 * torch.randn(B * S * S, 4 * H * D, device="cuda")).half()
    sbias = (0.5 * torch.randn(B, H, S, S, device="cuda")).half()
    v = proj.view(B, S, S, 4, H, D)[:, :, :, 2]
    if variant == "vconst":
        v.fill_(1.0)
    if variant == "uniform":
        proj.view(B, S, S, 4, H, D)[:, :, :, 0:2].zero_()
        sbias.zero_()
    gb = torch.zeros(0, device="cuda")
    ref = dh.ext_d64().triattn_d64_forward(proj, sbias, gb, H, D**-0.5, None).float()
    vt, vs = hs.quantize_v(proj.view(B, S, S, 4, H, D)[:, :, :, 2])
    out = dh.ext_d64_f8().triattn_d64_forward(proj, sbias, gb, H, D**-0.5, None, vt, vs).float()
    err = (out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert err < 0.08, err  # e4m3 P and V: a few percent of the output (docs/fp8.md)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) core")
@pytest.mark.parametrize("model", ["chai1", "chai1_confidence"])
@pytest.mark.parametrize("starting", [True, False])
def test_module_route_close_to_fp16_and_off_is_default(model: str, starting: bool) -> None:
    L = 256
    m = TriangleAttention(256, 4, model=model, starting=starting, p_drop=0.0).cuda().eval()
    _init(m, 1)
    m = m.to(torch.bfloat16)
    m.set_backend("cuda")
    g = torch.Generator(device="cpu").manual_seed(5)
    x = torch.randn(1, L, L, 256, generator=g).cuda().to(torch.bfloat16)
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[:, L - L // 8 :] = False
    old = hs.FP8_PV_D64_MIN_S
    try:
        fp8.enable_triattn(frozenset(), False)
        with torch.no_grad():
            y0 = m(x, mask)
        fp8.enable_triattn(frozenset({"d64"}), False)
        hs.FP8_PV_D64_MIN_S = 0
        hs.STATS["pv_d64"] = 0
        with torch.no_grad():
            m(x, mask)  # the site's calibration call
            y8 = m(x, mask)
        assert hs.STATS["pv_d64"] >= 1  # the route ran (static scale)
        fp8.enable_triattn(frozenset(), False)
        with torch.no_grad():
            y1 = m(x, mask)
    finally:
        hs.FP8_PV_D64_MIN_S = old
        fp8.enable_triattn(frozenset(), False)
    assert torch.equal(y0, y1)  # switch off: the default route's bytes
    upd = (y0.float() - x.float()).pow(2).mean().sqrt()
    assert 0 < (y8.float() - y0.float()).pow(2).mean().sqrt() / upd < 0.1


def test_chai1_is_in_the_triattn_table() -> None:
    kinds, qk = fp8.TRIATTN_MODELS["chai1"]
    assert "d64" in kinds and not qk
    assert hs.FP8_PV_D64_MIN_S >= 0
