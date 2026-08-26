"""Two units on one card may compile together; they may not measure together.

A unit alternates between compiling (a pool of processes, no GPU) and measuring (one GPU, one
core). On the A6000 rebuild those were 72% and 20% of unit wall, and neither overlapped the
other: for a fifth of every unit its fifteen compile workers sat idle, and for the rest the card
did nothing. Putting two units on a card fills both gaps.

What it must not do is let both measure at once. Two kernels sharing the SMs both read slower, by
an amount that drifts over a round, and a build whose readings drift picks a different config --
which is the entire output. So each card carries one lock, taken for a whole tuning round.
"""
from __future__ import annotations

import fcntl

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import capture


@pytest.fixture(autouse=True)
def _released():
    yield
    capture._bench_lock_release()
    settings.configure(bench_lock="")
    for k, v in list(capture._BENCH_LOCK.items()):
        if k != "fh":
            capture._BENCH_LOCK[k] = type(v)()


def _is_locked(path) -> bool:
    """True if some other file description holds this card's lock."""
    with open(path, "w") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        return False


def test_a_held_lock_shuts_the_other_unit_out(tmp_path):
    lock = tmp_path / "gpu0.benchlock"
    settings.configure(bench_lock=str(lock))
    capture._bench_lock_acquire()
    assert _is_locked(lock), "the other unit on this card could have started measuring"


def test_the_lock_is_released_when_the_round_ends(tmp_path):
    lock = tmp_path / "gpu0.benchlock"
    settings.configure(bench_lock=str(lock))
    capture._bench_lock_acquire()
    capture._bench_lock_release()
    assert not _is_locked(lock), "a card left locked stalls every later unit on it"


def test_a_round_that_dies_mid_sweep_does_not_strand_the_card(tmp_path):
    """`shutdown_precompile` runs on the way out of every unit, however the unit ended."""
    lock = tmp_path / "gpu0.benchlock"
    settings.configure(bench_lock=str(lock))
    capture._bench_lock_acquire()
    capture.shutdown_precompile()
    assert not _is_locked(lock)


def test_acquiring_twice_is_one_lock(tmp_path):
    """Rounds are not supposed to nest, but a double acquire must not need a double release."""
    lock = tmp_path / "gpu0.benchlock"
    settings.configure(bench_lock=str(lock))
    capture._bench_lock_acquire()
    capture._bench_lock_acquire()
    assert capture._BENCH_LOCK["rounds"] == 1
    capture._bench_lock_release()
    assert not _is_locked(lock)


def test_a_build_that_gave_no_lock_is_not_slowed_by_one():
    """One unit per card is the default; it must not pay for machinery it does not need."""
    settings.configure(bench_lock="")
    capture._bench_lock_acquire()
    assert capture._BENCH_LOCK["fh"] is None
    assert capture._BENCH_LOCK["rounds"] == 0


def test_the_wait_is_reported(tmp_path):
    """If sharing a card turns out to cost more than it saves, the log has to say so rather than
    the build just being slow."""
    lock = tmp_path / "gpu0.benchlock"
    settings.configure(bench_lock=str(lock))
    capture._bench_lock_acquire()
    capture._bench_lock_release()
    assert "bench-lock" in capture.precompile_summary()


def test_the_lock_reaches_the_unit_on_its_command_line(tmp_path, monkeypatch):
    """Not through the environment. Every other knob a unit takes is an argument, for the reason
    `_run_unit_subprocess` gives: what a unit did should be readable off the command line that ran
    it, not out of whatever shell started the build."""
    seen: list[str] = []

    class _Proc:
        returncode = 0

    def _run(cmd, **kw):
        seen.clear()
        seen.extend(cmd)
        assert "MINIWORLD_BENCH_LOCK" not in (kw.get("env") or {}), (
            "the lock must not travel in the environment: os.environ is the PARENT's, so a lock "
            "set there would follow every later unit onto whatever card it landed on")
        return _Proc()

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder.subprocess, "run", _run)
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    for share, want in ((False, False), (True, True)):
        shard_dir = tmp_path / f"share{share}"
        shard_dir.mkdir()
        builder._run_unit_subprocess(unit, 3, shard_dir, tmp_path, 4, share_card=share)
        assert ("--bench-lock" in seen) is want
        if want:
            assert seen[seen.index("--bench-lock") + 1].endswith("gpu3.benchlock"), (
                "the lock is per CARD; a per-unit lock would lock nothing")


def test_the_probe_pass_reaches_the_unit_the_same_way(tmp_path, monkeypatch):
    seen: list[str] = []

    class _Proc:
        returncode = 0

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder.subprocess, "run",
                        lambda cmd, **kw: (seen.clear(), seen.extend(cmd), _Proc())[-1])
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    for predict, want in ((False, False), (True, True)):
        shard_dir = tmp_path / f"pred{predict}"
        shard_dir.mkdir()
        builder._run_unit_subprocess(unit, 0, shard_dir, tmp_path, 4, predict=predict)
        assert ("--predict-unusable" in seen) is want
