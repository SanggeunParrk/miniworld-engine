"""EXPERIMENTAL FP8 placements of the H100 bf16 token DiT TRAINING block (``conditioned_transition.cuda.token_dit_fp8_train``,
docs/fp8_training.md "Token DiT"): the row quantizers against torch, each placement's gradients against the bf16 route (bounded, finite),
bitwise repeatability of the FP8 route, and the default route untouched (switch off = the same bytes as never importing the switch)."""

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
    from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8_train

    return token_dit_fp8_train


def _master(seed: int = 0) -> typing.Any:
    module = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(seed))
    out: typing.Any = C.cast(module, "master")
    return out.train().set_backend("cuda")


def _grads(m, A: int, L: int, places: str, e5: bool = False) -> list[torch.Tensor]:
    F = _F()
    old = F.PLACES, F.GRAD_E5M2
    F.PLACES, F.GRAD_E5M2 = F.parse(places), e5
    try:
        g = torch.Generator(device="cuda").manual_seed(1)
        inp = m.make_inputs(
            length=L, samples=A, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single_shared_cond"
        )
        s = inp["single"].detach().clone().requires_grad_(True)
        c = inp["cond"].detach()[:1].clone().requires_grad_(True)
        p = inp["pair"].detach().clone().requires_grad_(True)
        dout = torch.randn(A, 1, L, 768, device="cuda", generator=g).bfloat16()
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(s, c.expand(A, -1, -1, -1), p, inp["mask"])
        y.backward(dout)
        return [y.detach(), typing.cast(torch.Tensor, s.grad), typing.cast(torch.Tensor, c.grad), typing.cast(torch.Tensor, p.grad)] + [
            typing.cast(torch.Tensor, q.grad) for q in m.parameters() if q.requires_grad
        ]
    finally:
        F.PLACES, F.GRAD_E5M2 = old


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("C_", [768, 1536, 3072])
@pytest.mark.parametrize("e5", [False, True])
def test_q8_rows_matches_torch(C_: int, e5: bool) -> None:
    F = _F()
    g = torch.Generator(device="cuda").manual_seed(C_)
    x = (torch.randn(512, C_, device="cuda", generator=g) * torch.logspace(-4, 3, 512, device="cuda")[:, None]).bfloat16()
    x[7] = 0
    x8, s = F.q8(x, e5)
    fmax = 57344.0 if e5 else 448.0
    am = x.float().abs().amax(1, keepdim=True)
    ref_s = torch.where(am > 0, am / fmax, torch.ones_like(am))
    torch.testing.assert_close(s, ref_s, rtol=1e-6, atol=0)
    # one rounding to the 8-bit grid: half a unit in the last place (e4m3 2^-4, e5m2 2^-3 relative; the subnormal step absolute)
    v = x.float() / s
    half, sub = (2.0**-3, 2.0**-17) if e5 else (2.0**-4, 2.0**-10)
    assert bool(((x8.float() - v).abs() <= torch.maximum(v.abs() * half, torch.full_like(v, sub)) * 1.001).all())
    assert float(x8.float().abs().max()) <= fmax


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_swiglu_rows_match_torch() -> None:
    F = _F()
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

    g = torch.Generator(device="cuda").manual_seed(3)
    M, N = 384, 1536
    ab = torch.randn(M, 2 * N, device="cuda", generator=g).bfloat16()
    dh = torch.randn(M, N, device="cuda", generator=g).bfloat16()
    a, b = ab.float()[:, :N], ab.float()[:, N:]
    h8, sh = F.swiglu_q8(ab)
    h = torch.nn.functional.silu(a) * b
    assert ((h8.float() * sh - h).norm() / h.norm()) < 0.04
    T = ext(768, 48, rpb=2, dc=384, n=2, dp=128)
    dab0, h0 = torch.empty_like(ab), torch.empty_like(dh)
    T.swiglu_bwd(dh, ab, dab0, h0, False)
    dab, h1 = torch.empty_like(ab), torch.empty_like(dh)
    for e5 in (False, True):
        dab8, s = F.swiglu_bwd_q8(dh, ab, dab, h1, e5)
        # bf16 outputs as the default kernel's within one bf16 rounding (expf / fma order), the 8-bit copy one 8-bit rounding of dab
        assert ((dab.float() - dab0.float()).abs() <= 1e-2 * dab0.float().abs() + 1e-6).all()
        assert ((h1.float() - h0.float()).abs() <= 1e-2 * h0.float().abs() + 1e-6).all()
        assert ((dab8.float() * s - dab.float()).norm() / dab.float().norm()) < (0.08 if e5 else 0.04)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("places", ["transition", "attention", "all"])
@pytest.mark.parametrize("e5", [False, True])
def test_placement_gradients_near_bf16(places: str, e5: bool) -> None:
    m = _master()
    ref = _grads(m, 6, 256, "")
    got = _grads(m, 6, 256, places, e5)
    for i, (r, x) in enumerate(zip(ref, got, strict=True)):
        assert bool(torch.isfinite(x).all()), i
        rel = float((x.float() - r.float()).norm() / r.float().norm().clamp_min(1e-30))
        # measured (docs/fp8_training.md): FP8 vs the bf16 route 1-6 % on the large gradients; bound with margin
        assert rel < 0.15, (i, rel)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_fp8_route_is_bitwise_repeatable() -> None:
    m = _master()
    a = _grads(m, 6, 256, "all")
    b = _grads(m, 6, 256, "all")
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_switch_off_is_the_default_route() -> None:
    m = _master()
    ref = _grads(m, 6, 256, "")
    _grads(m, 6, 256, "all")  # an FP8 call in between (packs, caches) leaves the default bytes alone
    again = _grads(m, 6, 256, "")
    for x, y in zip(ref, again, strict=True):
        assert torch.equal(x, y)
