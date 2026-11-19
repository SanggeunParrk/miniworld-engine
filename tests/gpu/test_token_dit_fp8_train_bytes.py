"""Round-2 byte reductions of the H100 bf16 token DiT TRAINING block (``conditioned_transition.cuda.token_dit_fp8_train`` places
``nox``, docs/fp8_training.md "Token DiT round 2"): the output and every gradient are bit-identical to the default route, the
fp32 residual copy is not kept, and the default route (switch off) is untouched by the new template instantiations."""

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


def _run(m, A: int, L: int, places: str, default_bytes: bool = False) -> list[torch.Tensor]:
    F = _F()
    old = F.PLACES, F.MIN_ROWS, F.DEFAULT_BYTES
    F.PLACES, F.MIN_ROWS = F.parse(places), 0
    F.DEFAULT_BYTES = old[2] if default_bytes else frozenset()  # default_bytes: the route as shipped (nox + hsave on by default)
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
        F.PLACES, F.MIN_ROWS, F.DEFAULT_BYTES = old


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("places", ["nox", "hsave", "rx1", "nox,hsave,rx1", "bytes"])
@pytest.mark.parametrize("A,L", [(8, 128), (6, 200), (48, 128)])
def test_bytes_are_bit_identical(A: int, L: int, places: str) -> None:
    m = _master()
    ref = _run(m, A, L, "")
    got = _run(m, A, L, places)
    assert len(ref) == len(got) == 4 + sum(q.requires_grad for q in m.parameters())
    for i, (a, b) in enumerate(zip(ref, got, strict=True)):
        assert torch.equal(a, b), f"tensor {i}: nox differs from the default route ({places})"


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_nox_keeps_no_fp32_residual() -> None:
    from miniworld_engine.kernels.integrations import token_dit_train as TT

    F = _F()
    m = _master()
    n_f32 = {}
    for places in ("", "nox"):
        old = F.PLACES, F.MIN_ROWS, F.DEFAULT_BYTES
        F.PLACES, F.MIN_ROWS, F.DEFAULT_BYTES = F.parse(places), 0, frozenset()
        try:
            inp = m.make_inputs(length=128, samples=8, depth=None, dtype=torch.bfloat16, device="cuda", stream="token_single_shared_cond", generator=torch.Generator(device="cuda").manual_seed(1))
            s = inp["single"].detach().clone().requires_grad_(True)
            c = inp["cond"].detach()[:1].clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(s, c.expand(8, -1, -1, -1), inp["pair"], inp["mask"])
            fn = y.grad_fn
            saved = [t for t in getattr(fn, "saved_tensors", ())] if fn is not None else []
            n_f32[places] = sum(t.numel() * t.element_size() for t in saved if t is not None and t.dtype == torch.float32 and t.shape[-1:] == (768,))
        finally:
            F.PLACES, F.MIN_ROWS, F.DEFAULT_BYTES = old
    assert TT is not None
    # the saved fp32 [M, 768] residuals: x1 only with nox, x and x1 without
    assert n_f32["nox"] < n_f32[""]


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_bytes_preset_and_gate() -> None:
    F = _F()
    assert F.parse("bytes") == frozenset(("nox", "hsave", "rx1"))
    old = F.PLACES, F.MIN_ROWS
    F.PLACES, F.MIN_ROWS = F.parse("fast,nox"), 16384
    try:
        assert F.active(1024, 768, False, True, True) == frozenset(("nox", "hsave"))  # below the row gate: only the byte reductions (nox asked, hsave by default)
        assert F.active(32768, 768, False, True, True) == F.PLACES | F.DEFAULT_BYTES
        assert F.active(32768, 768, True, True, True) == frozenset()  # fp32: none
    finally:
        F.PLACES, F.MIN_ROWS = old


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("A,L", [(8, 128), (6, 200)])
def test_fused_q8_and_rx1_with_expand_are_bit_identical(A: int, L: int) -> None:
    """Round 4: res_adaln_b writes the e4m3 xt itself (FUSE_Q8) and rx1 recomputes x1: the expand route's output and gradients are
    bit-identical to the unfused / stored versions."""
    F = _F()
    m = _master()
    old_fuse = F.FUSE_Q8
    try:
        F.FUSE_Q8 = False
        ref = _run(m, A, L, "expand,nox,hsave")
        F.FUSE_Q8 = True
        for places in ("expand,nox,hsave", "expand,nox,hsave,rx1"):
            got = _run(m, A, L, places)
            for i, (a, b) in enumerate(zip(ref, got, strict=True)):
                assert torch.equal(a, b), f"tensor {i}: {places} differs from the unfused route"
    finally:
        F.FUSE_Q8 = old_fuse


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_rx1_keeps_no_fp32_x1() -> None:
    F = _F()
    m = _master()
    n_f32 = {}
    for places in ("nox", "nox,rx1"):
        old = F.PLACES, F.MIN_ROWS
        F.PLACES, F.MIN_ROWS = F.parse(places), 0
        try:
            inp = m.make_inputs(length=128, samples=8, depth=None, dtype=torch.bfloat16, device="cuda", stream="token_single_shared_cond", generator=torch.Generator(device="cuda").manual_seed(1))
            s = inp["single"].detach().clone().requires_grad_(True)
            c = inp["cond"].detach()[:1].clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(s, c.expand(8, -1, -1, -1), inp["pair"], inp["mask"])
            fn = y.grad_fn
            saved = [t for t in getattr(fn, "saved_tensors", ())] if fn is not None else []
            n_f32[places] = sum(t.numel() * t.element_size() for t in saved if t is not None and t.dtype == torch.float32 and t.shape[-1:] == (768,))
        finally:
            F.PLACES, F.MIN_ROWS = old
    assert n_f32["nox,rx1"] < n_f32["nox"]


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("A,L", [(8, 128), (6, 200), (48, 128)])
def test_default_route_takes_the_byte_reductions_and_is_unchanged(A: int, L: int) -> None:
    """Release candidate: with no switch the H100 bf16 block runs nox + hsave and equals the route without them bit for bit."""
    F = _F()
    assert F.DEFAULT_BYTES == frozenset(("nox", "hsave"))
    old = F.PLACES
    F.PLACES = frozenset()
    try:
        assert F.active(32768, 768, False, True, True) == frozenset(("nox", "hsave"))
        assert F.active(32768, 768, True, True, True) == frozenset()  # fp32: none
        assert F.active(32768, 768, False, False, True) == frozenset()  # non-default options: none
        assert F.active(32768, 768, False, True, False) == frozenset()  # not H100: none
    finally:
        F.PLACES = old
    m = _master()
    ref = _run(m, A, L, "", default_bytes=False)
    got = _run(m, A, L, "", default_bytes=True)
    for i, (a, b) in enumerate(zip(ref, got, strict=True)):
        assert torch.equal(a, b), f"tensor {i}"
