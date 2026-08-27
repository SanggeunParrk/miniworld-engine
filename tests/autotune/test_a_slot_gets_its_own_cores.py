"""A unit that is MEASURING must not compete with other units' compile workers.

The measurement is what a build produces. With the node's cores pooled, a slot that is benching
runs one Python launch loop against every other slot's compile pool -- and each of those workers
forks a child per chunk, so four slots asking for 32 workers put about 256 runnable processes on
128 cores. Measured, one launch cost 21 ms inside a loaded build against 329 us on an idle card,
and the event timer's own step is 1.024 us: at 21 ms the configs are no longer distinguishable.

The trade is real in both directions and is why this is a flag. While a slot measures, the cores
it owns are idle and no other slot may borrow them.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune import builder


@pytest.fixture
def eight(monkeypatch):
    monkeypatch.setattr(builder.os, "sched_getaffinity", lambda _pid: set(range(8)))


def test_the_slices_are_disjoint_and_cover_the_allocation(eight):
    got = builder._core_slices(4)
    assert got == ["0,1", "2,3", "4,5", "6,7"]
    assert len({c for s in got for c in s.split(",")}) == 8


def test_it_slices_the_ALLOCATION_not_the_machine(monkeypatch):
    """Under Slurm the job owns a subset. Slicing the machine would hand a slot cores it may not
    run on, and `taskset` would fail on every unit."""
    monkeypatch.setattr(builder.os, "sched_getaffinity", lambda _pid: {64, 65, 66, 67})
    assert builder._core_slices(2) == ["64,65", "66,67"]


def test_one_slot_is_never_pinned(eight):
    """There is nothing for it to contend with."""
    assert builder._core_slices(1) == [""]


def test_too_few_cores_leaves_everything_unpinned(monkeypatch):
    """A core per slot is worse than the contention: a unit's compile pool would be one worker."""
    monkeypatch.setattr(builder.os, "sched_getaffinity", lambda _pid: set(range(6)))
    assert builder._core_slices(4) == [""] * 4


def test_a_machine_that_cannot_say_leaves_everything_unpinned(monkeypatch):
    def _boom(_pid):
        raise OSError("no affinity here")

    monkeypatch.setattr(builder.os, "sched_getaffinity", _boom)
    assert builder._core_slices(4) == [""] * 4


def test_the_slice_reaches_the_unit_as_taskset(tmp_path, monkeypatch):
    seen: list[str] = []

    class _Proc:
        returncode = 0

    monkeypatch.setattr(builder.subprocess, "run",
                        lambda cmd, **kw: (seen.clear(), seen.extend(cmd), _Proc())[-1])
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    d = tmp_path / "pinned"
    d.mkdir()
    builder._run_unit_subprocess(unit, 0, d, tmp_path, 4, cores="8,9,10,11")
    assert seen[:3] == ["taskset", "-c", "8,9,10,11"]


def test_no_slice_means_no_taskset(tmp_path, monkeypatch):
    seen: list[str] = []

    class _Proc:
        returncode = 0

    monkeypatch.setattr(builder.subprocess, "run",
                        lambda cmd, **kw: (seen.clear(), seen.extend(cmd), _Proc())[-1])
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    d = tmp_path / "free"
    d.mkdir()
    builder._run_unit_subprocess(unit, 0, d, tmp_path, 4)
    assert "taskset" not in seen
