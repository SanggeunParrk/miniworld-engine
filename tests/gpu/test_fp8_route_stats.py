"""H100: ``miniworld_engine.fp8.route_stats()`` -- the per-route call counters (observability only). Every route has a counter dict, a reset
zeroes them, an ``engine-fp8`` Pairformer block moves the TriMul / Transition / TriAttn counters, and the default (cuda) block moves none and
returns the same bytes before and after."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]

_ROUTES = {"triattn", "trimul", "opm", "pwa", "transition", "token_dit", "atom_dit"}


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _total(stats: dict) -> int:
    return sum(sum(d.values()) for d in stats.values())


def test_route_stats_shape_and_reset() -> None:
    _needs_h100()
    from miniworld_engine import fp8

    s = fp8.route_stats()
    assert set(s) == _ROUTES
    assert {"calls", "fallbacks", "capture_fallbacks"} <= set(s["trimul"])
    assert {"pv_fused", "pv_front", "pv_pass", "qk"} <= set(s["triattn"])
    assert {"calls", "wide_calls", "declined"} <= set(s["pwa"])
    assert {"calls", "expand", "declined"} <= set(s["token_dit"])
    assert {"calls", "pre", "post", "fallbacks"} <= set(s["atom_dit"])
    s["trimul"]["calls"] += 10**6  # a copy: editing it does not touch the counters
    assert fp8.route_stats()["trimul"]["calls"] != s["trimul"]["calls"]
    fp8.reset_route_stats()
    assert _total(fp8.route_stats()) == 0


def test_route_stats_follow_engine_fp8_calls() -> None:
    _needs_h100()
    from miniworld_engine import fp8
    from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig

    cfg = PairformerConfig(d_pair=128, model="af3")
    a = PairformerBlock(cfg, master_weights=True, backend="cuda")
    g = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():
        for name, p in a.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5 if p.dim() > 1 else (0.1 if name.endswith("bias") else 1.0) * torch.ones(p.shape))
    a = a.cuda().eval()
    b = PairformerBlock(cfg, master_weights=True, backend="engine-fp8").cuda().eval()
    b.load_state_dict(a.state_dict())
    n = 256
    gen = torch.Generator(device="cuda").manual_seed(2)
    pair = torch.randn(1, n, n, 128, device="cuda", generator=gen).bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    fp8.set_model("af3")
    try:
        with torch.no_grad():
            y0 = a(pair, mask)
            fp8.reset_route_stats()
            y1 = a(pair, mask)
            assert _total(fp8.route_stats()) == 0  # the default route moves no FP8 counter
            b(pair, mask)
            s = fp8.route_stats()
            assert s["trimul"]["calls"] > 0 and s["transition"]["calls"] > 0
            assert s["triattn"]["pv_fused"] > 0 and s["triattn"]["d128_calls"] >= s["triattn"]["pv_fused"]
            y2 = a(pair, mask)
    finally:
        fp8.set_model(None)
        fp8.reset_route_stats()
    assert torch.equal(y0, y1) and torch.equal(y0, y2)
