"""A config whose launch proves it cannot win is abandoned, not repeated.

`do_bench` picks its repeat count to fill a time budget, so a config whose single launch takes 27
seconds is charged for a warmup plus however many repeats the estimate asks for -- and nothing
bounded that. Measured on the A6000 rebuild, 1,163 units:

    median unit          0.012 s per launch
    99th percentile      0.286
    slowest healthy      0.786
    then                 1.8, 2.0, 7.6, 26.8, 28.0, 53.5

Those six units burned 42.7 of the build's 128.4 GPU-hours. The worst of them searched 540 configs
in 11.2 hours and its WINNER ran 0.133 ms -- a factor of 200,000 between the config that won and
the launches being timed alongside it.

`builder.py` still carries a `#:` comment describing this exact guard, with the measurement that
motivated it (one config at 468 seconds, 85% of its unit's benchmarking) and no constant beneath
it. The comment outlived its code; this is the code.

Two properties the tests below pin, because getting either wrong is worse than having no guard:

  * the FIRST launch is exempt and untimed -- it pays handle initialisation and a cold cache
    (23 s a call in that rebuild), so a config whose steady state is fast must not be judged on it;
  * the threshold is RELATIVE to the round's fastest config, and resets per round. What counts as
    slow is a property of the kernel AND the shape: the same config is 0.9 ms at L=128 and 27 s at
    L=384.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from miniworld_engine.autotune import capture as X


@pytest.fixture(autouse=True)
def _clean():
    X._ROUND_FASTEST.clear()
    X._OVER_BUDGET.clear()
    yield
    X._ROUND_FASTEST.clear()
    X._OVER_BUDGET.clear()


class _Clock:
    """A fake launch: `calls` records every invocation, `secs` is what each one costs."""

    def __init__(self, secs, now):
        self.secs, self.now, self.calls = secs, now, 0

    def __call__(self):
        self.calls += 1
        self.now[0] += self.secs


def _autotuner(inner):
    return SimpleNamespace(_do_bench=inner, configs=[], __dict__={"_do_bench": inner})


def _install(monkeypatch, ms, secs_per_launch, now):
    """Wrap an autotuner whose real `do_bench` reports `ms` and whose launches cost `secs`."""
    inner_calls = []

    def inner(call, quantiles=None):
        inner_calls.append(1)
        return [ms, ms, ms] if quantiles else ms

    at = SimpleNamespace(_do_bench=inner, configs=[])
    # The real clock, patched through pytest so it is restored. `_install_launch_budget` does
    # `import time` inside the function, so this is the object it will read.
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(X, "_op_name", lambda a: "op_probe")
    X._install_launch_budget(at)
    return at, inner_calls


@pytest.fixture
def now():
    return [0.0]


@pytest.fixture(autouse=True)
def _fake_cuda(monkeypatch):
    import sys
    fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_the_first_config_of_a_round_is_never_judged(monkeypatch, now):
    """There is nothing to compare it against, and a round whose first config is slow must still
    produce an anchor rather than abandoning everything."""
    at, inner = _install(monkeypatch, ms=5000.0, secs_per_launch=5.0, now=now)
    call = _Clock(5.0, now)
    got = at._do_bench(call, quantiles=(0.5, 0.2, 0.8))
    assert got[0] == 5000.0, "the first config was abandoned with no anchor to judge it by"
    assert inner, "the real do_bench was skipped for the first config"
    assert not X.over_budget()


def test_a_config_far_slower_than_the_round_best_is_abandoned(monkeypatch, now):
    at, inner = _install(monkeypatch, ms=0.1, secs_per_launch=0.0001, now=now)
    at._do_bench(_Clock(0.0001, now), quantiles=(0.5, 0.2, 0.8))   # anchor: 0.1 ms
    before = len(inner)
    slow = _Clock(30.0, now)                                        # 300,000x the anchor
    got = at._do_bench(slow, quantiles=(0.5, 0.2, 0.8))
    assert got == [float("inf")] * 3
    assert len(inner) == before, "the slow config was still handed to do_bench to repeat"
    assert slow.calls == 2, "a judged config costs one warmup and one timed launch, no more"
    assert X.over_budget() == {"op_probe": 1}


def test_a_config_within_the_ratio_is_measured_normally(monkeypatch, now):
    at, inner = _install(monkeypatch, ms=0.1, secs_per_launch=0.0001, now=now)
    at._do_bench(_Clock(0.0001, now), quantiles=(0.5, 0.2, 0.8))
    before = len(inner)
    got = at._do_bench(_Clock(0.001, now), quantiles=(0.5, 0.2, 0.8))   # 10x, under the ratio
    assert got[0] == 0.1
    assert len(inner) == before + 1, "a config well inside the budget was abandoned"
    assert not X.over_budget()


def test_a_microsecond_kernel_is_not_judged_on_noise(monkeypatch, now):
    """100x of 8 us is 0.8 ms, which a context switch can produce. The floor is what stops the
    ratio turning scheduling noise into a verdict."""
    at, inner = _install(monkeypatch, ms=0.008, secs_per_launch=0.000008, now=now)
    at._do_bench(_Clock(0.000008, now), quantiles=(0.5, 0.2, 0.8))
    before = len(inner)
    got = at._do_bench(_Clock(0.01, now), quantiles=(0.5, 0.2, 0.8))  # 1250x, but under the floor
    assert got[0] == 0.008, "a fast kernel was judged below the floor"
    assert len(inner) == before + 1
    assert X._LAUNCH_BUDGET_FLOOR_S >= 0.01


def test_the_anchor_does_not_cross_rounds() -> None:
    """One shape's anchor must not judge another's: the same config is 0.9 ms at L=128 and 27 s at
    L=384, and a stale anchor would abandon the whole of the larger shape."""
    import inspect
    src = inspect.getsource(X)
    assert "_ROUND_FASTEST.pop(id(self), None)" in src, (
        "prune_configs no longer clears the launch-budget anchor, so a round inherits the previous "
        "shape's idea of fast")


def test_the_guard_is_relative_not_a_fixed_number_of_seconds() -> None:
    """An absolute budget cannot describe both a layernorm at 12 us and a b2b GEMM at 200 us."""
    assert X._LAUNCH_BUDGET_X >= 10, "a ratio this tight will abandon configs that could win"
    assert 0 < X._LAUNCH_BUDGET_FLOOR_S <= 0.5


# --------------------------------------------------------------------------- #
# the other half: a tile no winner has ever been is not generated
# --------------------------------------------------------------------------- #
def test_no_three_dimensional_winner_is_below_the_tile_floor() -> None:
    """The rule's evidence, re-derived from the shipped cache rather than trusted.

    An axis floor cannot express this: `BLOCK_K=16` with `BLOCK_M1=64, BLOCK_N=64` wins here, and a
    per-axis minimum of 16 would delete the winner for 26% of shapes. The product can.
    """
    import json
    from pathlib import Path

    from miniworld_engine.autotune.configs import _MIN_TILE_BASE, _MIN_TILE_DIMS

    data = Path(__file__).resolve().parents[2] / "src/miniworld_engine/autotune/data"
    bad, seen = [], 0
    for f in data.glob("*/*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for ranked in (d.get("entries") or {}).values():
            if not ranked:
                continue
            blocks = [int(v) for k, v in (ranked[0].get("kwargs") or {}).items()
                      if k.startswith("BLOCK")]
            if len(blocks) < _MIN_TILE_DIMS:
                continue
            seen += 1
            product = 1
            for b in blocks:
                product *= b
            if product <= _MIN_TILE_BASE ** len(blocks):
                bad.append(f"{f.parent.name}: {blocks}")
    if not seen:
        import pytest
        pytest.skip("no multi-axis tuned entries committed yet")
    assert not bad, (
        f"the tile floor would delete a config that actually won on {len(bad)} shape(s): "
        f"{bad[:5]}. Either the floor is too high or the evidence for it has changed.")


def test_the_floor_only_applies_to_tiles_with_three_or_more_axes() -> None:
    """1-D and 2-D tiles are the reductions, where a 4-row tile legitimately wins."""
    from miniworld_engine.autotune.configs import _MIN_TILE_DIMS, _tile_too_small

    assert _MIN_TILE_DIMS >= 3
    assert not _tile_too_small({"BLOCK_M1": 4}, ["BLOCK_M1"])
    assert not _tile_too_small({"BLOCK_M1": 2, "BLOCK_K": 64}, ["BLOCK_M1", "BLOCK_K"])
    assert _tile_too_small({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
                           ["BLOCK_M", "BLOCK_N", "BLOCK_K"])
    assert not _tile_too_small({"BLOCK_M1": 64, "BLOCK_N": 64, "BLOCK_K": 16},
                               ["BLOCK_M1", "BLOCK_N", "BLOCK_K"])
