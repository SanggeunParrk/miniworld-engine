"""EXPERIMENTAL e4m3 placements of the H100 bf16 token DiT inference block (``conditioned_transition.cuda.token_dit_fp8``, docs/fp8.md
"Token DiT"): the e4m3 kernels against torch references on the same e4m3 operands, the opt-in route's error against the bf16 route, and
the default route untouched by the switch being importable."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _F():
    from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8

    return token_dit_fp8


def _master(seed: int = 0) -> typing.Any:
    module = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(seed))
    out: typing.Any = C.cast(module, "master")
    return out.eval().set_backend("cuda")


def _run(m, inputs, places) -> torch.Tensor:
    F = _F()
    old, old_min = F.PLACES, F.MIN_ROWS
    F.PLACES, F.MIN_ROWS = frozenset(places), 0
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(**inputs)
    finally:
        F.PLACES, F.MIN_ROWS = old, old_min


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("M,ht", [(1920, 64), (3840, 128), (640, 64)])
def test_gemm_swiglu_f8_matches_reference(M: int, ht: int) -> None:
    F = _F()
    g = torch.Generator(device="cuda").manual_seed(M)
    K, H = 768, 1536
    x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
    w = (torch.randn(2 * H, K, device="cuda", generator=g) / K**0.5).bfloat16()
    x8, sx = F.rows_q8(x)
    w8, sw = F.quant_weight(w)
    h = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    F.ext().gemm_swiglu_f8(x8, w8, h, sx, sw, torch.cuda.get_device_properties(0).multi_processor_count, ht, False)
    # w8 rows are packed per 2 ht: tile t = [a rows ht t .. | b rows ht t ..]
    ab = (x8.float() * sx) @ (w8.float() * sw[:, None]).t()
    ab = ab.view(M, H // ht, 2, ht)
    ref = (torch.nn.functional.silu(ab[:, :, 0]) * ab[:, :, 1]).reshape(M, H)
    err = (h.float() - ref).abs().max() / ref.abs().max()
    assert err < 1e-2, err


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_quant_rows_scale_and_bytes() -> None:
    F = _F()
    x = torch.randn(1024, 768, device="cuda").bfloat16() * torch.logspace(-3, 3, 1024, device="cuda")[:, None].bfloat16()
    x8, s = F.rows_q8(x)
    ref_s = x.float().abs().amax(1, keepdim=True) / 448.0
    torch.testing.assert_close(s, ref_s, rtol=1e-6, atol=0)
    assert x8.float().abs().amax() <= 448.0
    deq = x8.float() * s
    # e4m3 rounding: half a step of 3 mantissa bits (normals), half the subnormal step 2^-9 (times the row scale) below
    assert ((deq - x.float()).abs() <= 0.0625 * x.float().abs() * 1.0001 + s * 2**-10).all()


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("places", [("qkvg",), ("out",), ("expand",), ("squeeze",), ("qkvg", "expand", "out", "squeeze")])
def test_fp8_route_close_to_bf16(places) -> None:
    m = _master()
    g = torch.Generator(device="cuda").manual_seed(1)
    inputs = m.make_inputs(length=384, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    y0 = _run(m, inputs, ())
    y8 = _run(m, inputs, places)
    assert torch.isfinite(y8).all()
    x = inputs["single"].float()
    upd = (y0.float() - x).pow(2).mean().sqrt()
    rel = (y8.float() - y0.float()).pow(2).mean().sqrt() / upd
    assert 0 < rel < 0.12, rel  # e4m3 operands: a few percent of the update (docs/fp8.md); 0 would mean the route did not run


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_switch_off_is_default_route() -> None:
    m = _master(3)
    g = torch.Generator(device="cuda").manual_seed(2)
    inputs = m.make_inputs(length=256, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    y0 = _run(m, inputs, ())
    _run(m, inputs, ("qkvg", "expand"))
    y1 = _run(m, inputs, ())
    assert torch.equal(y0, y1)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_below_min_rows_is_default_route() -> None:
    F = _F()
    m = _master(4)
    g = torch.Generator(device="cuda").manual_seed(2)
    inputs = m.make_inputs(length=256, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    y0 = _run(m, inputs, ())
    old, old_min = F.PLACES, F.MIN_ROWS
    F.PLACES, F.MIN_ROWS = frozenset(("qkvg", "expand")), 1920
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y1 = m(**inputs)
    finally:
        F.PLACES, F.MIN_ROWS = old, old_min
    assert torch.equal(y0, y1)


# ---------------------------------------------------------------------------------------------------------- Boltz-2 (up-gate transition)
def _master_boltz2(seed: int = 0) -> typing.Any:
    from miniworld_engine.modules.token_dit import TokenDiTBoltz2

    module = TokenDiTBoltz2(d_single=768, d_cond=768, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(seed))
    out: typing.Any = C.cast(module, "master")
    return out.eval().set_backend("cuda")


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("M", [640, 1920, 2560])
def test_gemm_swiglu_f8_up_matches_reference(M: int) -> None:
    """The up-gate expand (h = silu(a) b u, [Wa 64 | Wb 64 | Wu 64] tiles) against torch on the same e4m3 operands."""
    F = _F()
    g = torch.Generator(device="cuda").manual_seed(M)
    K, H, ht = 768, 1536, 64
    x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
    w = (torch.randn(3 * H, K, device="cuda", generator=g) / K**0.5).bfloat16()
    x8, sx = F.rows_q8(x)
    w8, sw = F.quant_weight(w)  # row-wise scales: the packed order is whatever the rows are
    h = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    F.ext().gemm_swiglu_f8(x8, w8, h, sx, sw, torch.cuda.get_device_properties(0).multi_processor_count, ht, False, True)
    abu = ((x8.float() * sx) @ (w8.float() * sw[:, None]).t()).view(M, H // ht, 3, ht)
    ref = (torch.nn.functional.silu(abu[:, :, 0]) * abu[:, :, 1] * abu[:, :, 2]).reshape(M, H)
    err = (h.float() - ref).abs().max() / ref.abs().max()
    assert err < 1e-2, err


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_boltz2_expand_route_runs_and_is_close() -> None:
    """Boltz-2's block takes the e4m3 expand (it was refused: ``plain`` is False with the up-gate): output differs from the bf16 route by
    a few percent of the update; against the PyTorch fp64 reference (random weights) the e4m3 expand costs about what it costs the AF3-family blocks (docs/fp8.md "Token DiT": expand 5.4 % of the update) and the bf16 route stays under 1 %."""
    m = _master_boltz2()
    g = torch.Generator(device="cuda").manual_seed(1)
    inputs = m.make_inputs(length=384, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    y0 = _run(m, inputs, ())
    y8 = _run(m, inputs, ("expand",))
    assert torch.isfinite(y8).all()
    x = inputs["single"].float()
    upd = (y0.float() - x).pow(2).mean().sqrt()
    rel = (y8.float() - y0.float()).pow(2).mean().sqrt() / upd
    assert 0 < rel < 0.12, rel  # 0 would mean the route did not run
    # fp64 PyTorch reference of the same weights and inputs
    ref = _master_boltz2()
    ref.load_state_dict(m.state_dict())
    ref64 = ref.float().double().set_backend("pytorch")
    with torch.no_grad():
        y64 = ref64(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()})
    e0 = (y0.double() - y64).pow(2).mean().sqrt()
    e8 = (y8.double() - y64).pow(2).mean().sqrt()
    print(f"boltz2 token DiT vs fp64 (rms, update rms {upd.item():.4f}): bf16 {e0.item():.5f} ({e0.item() / upd.item():.2%}), expand e4m3 {e8.item():.5f} ({e8.item() / upd.item():.2%})")
    assert e0 < 0.015 * upd.double() and e8 < 0.10 * upd.double(), (e0.item(), e8.item())


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_boltz2_switch_off_is_default_route() -> None:
    m = _master_boltz2(3)
    g = torch.Generator(device="cuda").manual_seed(2)
    inputs = m.make_inputs(length=256, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    y0 = _run(m, inputs, ())
    _run(m, inputs, ("expand",))
    y1 = _run(m, inputs, ())
    assert torch.equal(y0, y1)


def test_expand_up_row_gate() -> None:
    """The up-gate (Boltz-2) expand has its own row gate (``MIN_ROWS_UP``); the default block keeps ``MIN_ROWS_PER['expand']``."""
    F = _F()
    old, old_min = F.PLACES, F.MIN_ROWS
    F.PLACES, F.MIN_ROWS = frozenset(("expand",)), None
    try:
        assert F.active(1920, 768, False, True, True) == frozenset()
        assert F.active(F.MIN_ROWS_UP, 768, False, True, True) == frozenset(("expand",))
        assert F.active(1920, 768, False, True, False) == frozenset(("expand",))
        assert F.active(2560, 1024, False, True, True) == frozenset()  # width 768 only
        assert F.active(2560, 768, False, False, True) == frozenset()  # a block outside the route
    finally:
        F.PLACES, F.MIN_ROWS = old, old_min
