"""Most of a bench iteration is zeroing a buffer forty times bigger than the card's L2.

Measured on an idle A6000, one triton kernel:

    the card's L2                            6 MB
    the buffer `do_bench` zeroes           256 MB
    zeroing it                             390 us
    the kernel being timed                  10 us

So 97% of a timed iteration is the eviction, not the kernel. 256 MB is what the largest L2 in
triton's fleet needs; anything at or above twice the card's own L2 evicts just as completely.

The two knobs that follow are ONE decision, and these tests exist mostly to keep them that way.
`do_bench` picks its repeat count to fill a time budget, so a cheaper iteration ALONE buys more
iterations rather than less time:

    clear    warmup/rep    launches    wall     reported
    256 MB      25/100          348   104 ms      5.1 us
     16 MB      25/100        4,243   145 ms      8.2 us      <- cheaper iteration, slower bench
     16 MB       3/10           452    15 ms      8.2 us      <- 7x cheaper, 30% MORE samples
     16 MB        2/5           251     8 ms      8.2 us

And the last column is why this is off by default: the reported time changes, 5.1 us against
8.2 us for the same kernel, so what a build CHOOSES could change with it.
"""
from __future__ import annotations

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import capture


class _Autotuner:
    def __init__(self):
        self._do_bench = "triton's own"


class _Driver:
    """Stands in for the backend driver, whose buffer is the thing being resized."""

    def __init__(self):
        self.get_empty_cache_for_benchmark = None


_DRIVER = _Driver()


@pytest.fixture(autouse=True)
def _default(monkeypatch):
    _DRIVER.get_empty_cache_for_benchmark = None
    monkeypatch.setattr(capture, "_bench_driver", lambda: _DRIVER)
    settings.configure(bench_clear_mb=0, bench_rep_ms=0)
    capture._BENCH_T["budget"] = ""
    yield
    settings.configure(bench_clear_mb=0, bench_rep_ms=0)
    capture._BENCH_T["budget"] = ""


def test_the_default_leaves_triton_alone():
    a = _Autotuner()
    capture._use_a_smaller_bench_budget(a)
    assert a._do_bench == "triton's own"
    assert not capture._BENCH_T["budget"]


def test_half_the_decision_is_refused(capsys):
    """A smaller clear at triton's budget ran 4,243 launches against 348 -- slower, not faster.
    Taking it as an instruction would be doing the wrong thing carefully."""
    for clear, rep in ((16, 0), (0, 10)):
        settings.configure(bench_clear_mb=clear, bench_rep_ms=rep)
        a = _Autotuner()
        capture._use_a_smaller_bench_budget(a)
        assert a._do_bench == "triton's own"
        assert "one decision" in capsys.readouterr().out


def test_both_together_replace_the_bench():
    settings.configure(bench_clear_mb=16, bench_rep_ms=10)
    a = _Autotuner()
    capture._use_a_smaller_bench_budget(a)
    assert a._do_bench != "triton's own"
    assert callable(a._do_bench)
    assert _DRIVER.get_empty_cache_for_benchmark is capture._bench_clear_buffer, (
        "the budget was replaced and the buffer was not -- that is the row that ran SLOWER")


def test_the_warmup_keeps_tritons_own_ratio():
    """25:100. Not a number worth inventing a new one for."""
    settings.configure(bench_clear_mb=16, bench_rep_ms=40)
    capture._use_a_smaller_bench_budget(_Autotuner())
    assert "10/40 ms" in capture._BENCH_T["budget"]


def test_a_tiny_budget_still_warms_up_at_least_once():
    settings.configure(bench_clear_mb=16, bench_rep_ms=2)
    capture._use_a_smaller_bench_budget(_Autotuner())
    assert "1/2 ms" in capture._BENCH_T["budget"]


def test_the_log_says_which_budget_a_unit_used():
    """The reported times move with it, so a shard measured under one budget and a shard measured
    under another are not the same experiment. The log has to say which."""
    settings.configure(bench_clear_mb=16, bench_rep_ms=10)
    capture._use_a_smaller_bench_budget(_Autotuner())
    capture._BENCH_T["calls"] = 1
    try:
        assert "16 MB clear, 2/10 ms" in capture.precompile_summary()
    finally:
        capture._BENCH_T["calls"] = 0


def test_it_reaches_the_unit_on_its_command_line(tmp_path, monkeypatch):
    seen: list[str] = []

    class _Proc:
        returncode = 0

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder.subprocess, "run",
                        lambda cmd, **kw: (seen.clear(), seen.extend(cmd), _Proc())[-1])
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    d = tmp_path / "a"
    d.mkdir()
    builder._run_unit_subprocess(unit, 0, d, tmp_path, 4, bench_clear_mb=16, bench_rep_ms=10)
    assert "--bench-clear-mb" in seen
    assert seen[seen.index("--bench-clear-mb") + 1] == "16"
    assert seen[seen.index("--bench-rep-ms") + 1] == "10"


def test_half_the_decision_never_reaches_the_unit(tmp_path, monkeypatch):
    seen: list[str] = []

    class _Proc:
        returncode = 0

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder.subprocess, "run",
                        lambda cmd, **kw: (seen.clear(), seen.extend(cmd), _Proc())[-1])
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    d = tmp_path / "b"
    d.mkdir()
    builder._run_unit_subprocess(unit, 0, d, tmp_path, 4, bench_clear_mb=16)
    assert "--bench-clear-mb" not in seen
