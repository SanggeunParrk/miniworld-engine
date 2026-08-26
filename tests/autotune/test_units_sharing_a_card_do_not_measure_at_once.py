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
import os

import pytest

from miniworld_engine.autotune import capture


@pytest.fixture(autouse=True)
def _released():
    yield
    capture._bench_lock_release()
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


def test_a_held_lock_shuts_the_other_unit_out(tmp_path, monkeypatch):
    lock = tmp_path / "gpu0.benchlock"
    monkeypatch.setenv("MINIWORLD_BENCH_LOCK", str(lock))
    capture._bench_lock_acquire()
    assert _is_locked(lock), "the other unit on this card could have started measuring"


def test_the_lock_is_released_when_the_round_ends(tmp_path, monkeypatch):
    lock = tmp_path / "gpu0.benchlock"
    monkeypatch.setenv("MINIWORLD_BENCH_LOCK", str(lock))
    capture._bench_lock_acquire()
    capture._bench_lock_release()
    assert not _is_locked(lock), "a card left locked stalls every later unit on it"


def test_a_round_that_dies_mid_sweep_does_not_strand_the_card(tmp_path, monkeypatch):
    """`shutdown_precompile` runs on the way out of every unit, however the unit ended."""
    lock = tmp_path / "gpu0.benchlock"
    monkeypatch.setenv("MINIWORLD_BENCH_LOCK", str(lock))
    capture._bench_lock_acquire()
    capture.shutdown_precompile()
    assert not _is_locked(lock)


def test_acquiring_twice_is_one_lock(tmp_path, monkeypatch):
    """Rounds are not supposed to nest, but a double acquire must not need a double release."""
    lock = tmp_path / "gpu0.benchlock"
    monkeypatch.setenv("MINIWORLD_BENCH_LOCK", str(lock))
    capture._bench_lock_acquire()
    capture._bench_lock_acquire()
    assert capture._BENCH_LOCK["rounds"] == 1
    capture._bench_lock_release()
    assert not _is_locked(lock)


def test_a_build_that_gave_no_lock_is_not_slowed_by_one(monkeypatch):
    """One unit per card is the default; it must not pay for machinery it does not need."""
    monkeypatch.delenv("MINIWORLD_BENCH_LOCK", raising=False)
    capture._bench_lock_acquire()
    assert capture._BENCH_LOCK["fh"] is None
    assert capture._BENCH_LOCK["rounds"] == 0


def test_the_wait_is_reported(tmp_path, monkeypatch):
    """If sharing a card turns out to cost more than it saves, the log has to say so rather than
    the build just being slow."""
    lock = tmp_path / "gpu0.benchlock"
    monkeypatch.setenv("MINIWORLD_BENCH_LOCK", str(lock))
    capture._bench_lock_acquire()
    capture._bench_lock_release()
    assert "bench-lock" in capture.precompile_summary()


def test_only_a_shared_card_hands_out_a_lock(tmp_path, monkeypatch):
    """The env var is the whole contract between the runner and the unit."""
    seen = {}

    class _Proc:
        returncode = 0

    def _run(cmd, **kw):
        seen.update(kw.get("env") or {})
        return _Proc()

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder.subprocess, "run", _run)
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    for share, want in ((False, False), (True, True)):
        seen.clear()
        shard_dir = tmp_path / f"share{share}"
        shard_dir.mkdir()
        builder._run_unit_subprocess(unit, 3, shard_dir, tmp_path, 4, share_card=share)
        assert ("MINIWORLD_BENCH_LOCK" in seen) is want
        if want:
            assert seen["MINIWORLD_BENCH_LOCK"].endswith("gpu3.benchlock"), (
                "the lock is per CARD; a per-unit lock would lock nothing")
        assert seen["CUDA_VISIBLE_DEVICES"] == "3"


def test_the_environment_is_not_leaked_between_units(tmp_path, monkeypatch):
    """`os.environ` is the parent's; setting the lock on it would leave every later unit sharing
    one card's lock."""
    monkeypatch.delenv("MINIWORLD_BENCH_LOCK", raising=False)

    class _Proc:
        returncode = 0

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder.subprocess, "run", lambda cmd, **kw: _Proc())
    unit = builder.OpUnit(op="layernorm_stats_triton", dtype="bfloat16", length=256)
    builder._run_unit_subprocess(unit, 1, tmp_path, tmp_path, 4, share_card=True)
    assert "MINIWORLD_BENCH_LOCK" not in os.environ
