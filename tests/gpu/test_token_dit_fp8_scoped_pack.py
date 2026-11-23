"""Token DiT e4m3 weight pack and the row gates of graphs captured without ``static_weights()`` (``conditioned_transition.cuda.token_dit_fp8``):
the pack is made by one ``quant_rows`` pass and only for the placements in force, and the expand takes a captured graph whose weights are
remade at every replay only from ``MIN_ROWS_SCOPED`` rows (the module A-B of docs/fp8.md "Token DiT FP8 in graphs captured without static weights")."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine import capture

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _F():
    from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8

    return token_dit_fp8


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("shape", [(4096, 768), (768, 768), (768, 1024), (3072, 2048)])
def test_quant_weight_matches_torch_reference(shape) -> None:
    F = _F()
    w = (torch.randn(shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)) * 0.05).bfloat16()
    a8, sa = F.quant_weight(w)
    r8, sr = F.quant_weight_ref(w)
    torch.testing.assert_close(sa, sr, rtol=1e-6, atol=0)
    # multiply by the reciprocal against a divide: a rounding tie of the e4m3 cast may fall the other way (a fraction of a percent of the values)
    assert (a8.float() != r8.float()).float().mean() < 1e-2
    deq = a8.float() * sa[:, None]
    assert ((deq - w.float()).abs().max(1).values / w.float().abs().max(1).values).max() < 0.07  # 3 mantissa bits: half a step = 2^-4 relative


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_weight_pack_builds_only_the_placements_in_force() -> None:
    F = _F()
    g = torch.Generator(device="cuda").manual_seed(2)
    mk = lambda n, k: (torch.randn(n, k, device="cuda", generator=g) * 0.05).bfloat16()  # noqa: E731
    args = (mk(3072, 768), mk(768, 768), mk(768, 1536), (mk(3072, 768), 128), mk(3072, 1).reshape(-1))
    old = F.PLACES
    try:
        F.PLACES = frozenset(("expand",))
        pack = F.weight_pack(("t-expand",), *args)
        assert set(pack) == {"expand"}
        F.PLACES = frozenset(("qkvg", "expand", "out", "squeeze"))
        pack = F.weight_pack(("t-all",), *args)
        assert set(pack) == {"qkvg", "bqkvg", "expand", "out", "squeeze"}
    finally:
        F.PLACES = old


def _active_in_capture(F, *a, **kw):
    out = []
    g = torch.cuda.CUDAGraph()
    x = torch.zeros(1, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x += 1
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        x += 1
        out.append(F.active(*a, **kw))
    return out[0]


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_scoped_capture_row_gates() -> None:
    F = _F()
    old, old_min = F.PLACES, F.MIN_ROWS
    F.PLACES, F.MIN_ROWS = frozenset(("expand",)), None
    try:
        e = frozenset(("expand",))
        assert F.active(1920, 768, False, True) == e  # eager: the gate of every call (640)
        assert _active_in_capture(F, 1920, 768, False, True) == frozenset()  # weights remade at every replay
        assert _active_in_capture(F, F.MIN_ROWS_SCOPED - 128 + 0, 768, False, True) == frozenset()
        assert _active_in_capture(F, 3200, 768, False, True) == e
        assert _active_in_capture(F, 2560, 768, False, True, up=True) == e  # Boltz-2's up-gate keeps its own gate
        assert _active_in_capture(F, 5120, 768, False, False, False, True) == frozenset()  # RoseTTAFold3 parallel block
        assert _active_in_capture(F, 10240, 768, False, False, False, True) == e
        with capture.static_weights():
            assert _active_in_capture(F, 1920, 768, False, True) == e  # packs made once: the eager gates
            assert _active_in_capture(F, 5120, 768, False, False, False, True) == e
        F.MIN_ROWS = 0
        assert _active_in_capture(F, 128, 768, False, True) == e  # the explicit override wins
    finally:
        F.PLACES, F.MIN_ROWS = old, old_min
