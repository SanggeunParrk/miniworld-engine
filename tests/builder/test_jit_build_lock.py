"""A leftover build lock must produce a message, not a hang.

`torch.utils.cpp_extension.load` serialises concurrent builds with a `FileBaton` on
`<build dir>/lock`, and `FileBaton.wait()` polls for that file to disappear **forever** -- no
timeout, no output. A build killed partway (a cancelled Slurm job, a Ctrl-C) leaves the file, and
every later run on that machine blocks on it.

That is not hypothetical. A run left a 13-hour-old lock in `layer_norm_cuda/`, and three GPU jobs
sat on it: one killed at 45 min, one that hit a 4-hour time limit, one stalled at test 42 of 100.
All three were indistinguishable from "slow" -- the suite prints nothing while waiting -- and it
took three investigations to find. Deleting the file moved the stalled job forward in 20 seconds.

`kernels/_nvcc.load_extension` makes both halves impossible: a lock too old to belong to a live
build is reclaimed, and a wait on a fresh one is bounded and raises naming the file. Correctness
still comes from torch's own baton; this only decides how long to tolerate it. The remedy the
message gives is one `rm`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from miniworld_engine.kernels._nvcc import (
    LOCK_WAIT_SECONDS,
    STALE_LOCK_SECONDS,
    _build_lock,
    clear_stale_lock,
    wait_for_lock,
)


def _lock(tmp_path: Path, age_seconds: float, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A lock created now, seen `age_seconds` later.

    The clock moves rather than the file's timestamps: `os.utime` rewrites mtime but also bumps
    ctime to now, and the guard reads max(mtime, ctime) -- a leftover lock is old on BOTH.
    """
    lock = tmp_path / "lock"
    lock.touch()
    later = time.time() + age_seconds
    monkeypatch.setattr(time, "time", lambda: later)
    return lock


def test_a_stale_lock_is_reclaimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact shape of the incident: 13 hours old, no process behind it."""
    lock = _lock(tmp_path, 13 * 3600, monkeypatch)
    assert clear_stale_lock(lock) is True
    assert not lock.exists()


def test_a_fresh_lock_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A live build owns its lock. Reclaiming it would corrupt a concurrent compile, which is a
    worse failure than the one being fixed."""
    lock = _lock(tmp_path, 5.0, monkeypatch)
    assert clear_stale_lock(lock) is False
    assert lock.exists()


def test_the_boundary_is_the_declared_threshold(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    assert clear_stale_lock(_lock(tmp_path, STALE_LOCK_SECONDS + 60, monkeypatch)) is True
    assert clear_stale_lock(_lock(tmp_path, STALE_LOCK_SECONDS - 60, monkeypatch)) is False


def test_an_nfs_exclusive_create_mtime_does_not_make_a_live_lock_stale(tmp_path: Path) -> None:
    """NFSv3 O_EXCL create stores its verifier in mtime: a lock made a second ago reads as 1981.
    Trusting mtime deleted a live rank's lock mid-build and hung the job (2026-09-27)."""
    lock = tmp_path / "lock"
    lock.touch()
    os.utime(lock, (375402281, 375402281))   # the mtime the compute node reported
    assert clear_stale_lock(lock) is False
    assert lock.exists()


def test_the_lock_is_looked_for_in_the_build_directory_it_uses(tmp_path: Path) -> None:
    """`load(build_directory=...)` batons there; guarding TORCH_EXTENSIONS_DIR instead left a
    days-old leftover unguarded and torch waited on it forever (2026-09-28)."""
    assert _build_lock("any_ext", str(tmp_path)) == tmp_path / "lock"


def test_a_missing_lock_is_not_an_error(tmp_path: Path) -> None:
    assert clear_stale_lock(tmp_path / "nope") is False


def test_no_lock_means_no_wait(tmp_path: Path) -> None:
    started = time.monotonic()
    wait_for_lock(tmp_path / "nope", limit=30.0)
    assert time.monotonic() - started < 1.0


def test_a_fresh_lock_that_never_clears_raises_and_names_the_file(tmp_path: Path) -> None:
    """The whole point: the previous behaviour here was to poll until the job's time limit."""
    lock = tmp_path / "lock"
    lock.touch()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match=r"rm .*lock"):
        wait_for_lock(lock, limit=3.0)
    waited = time.monotonic() - started
    assert 2.0 < waited < 15.0, f"waited {waited:.1f}s for a 3s limit"


def test_the_message_says_what_to_do(tmp_path: Path) -> None:
    """A message that names the problem but not the remedy would still cost an investigation."""
    lock = tmp_path / "lock"
    lock.touch()
    with pytest.raises(TimeoutError) as excinfo:
        wait_for_lock(lock, limit=1.0)
    text = str(excinfo.value)
    assert str(lock) in text
    assert "rm " in text
    assert "forever" in text or "leftover" in text


def test_the_defaults_are_ordered_sensibly() -> None:
    """A stale threshold above the wait limit would mean waiting the full limit on a lock that was
    already reclaimable."""
    assert STALE_LOCK_SECONDS >= LOCK_WAIT_SECONDS


def test_every_jit_load_goes_through_the_guard() -> None:
    """The guard is worth nothing on a call site that still uses `load` directly."""
    import ast

    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "miniworld_engine"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "_nvcc.py":
            continue                      # where load_extension itself calls load
        offenders.extend(
            f"{path.relative_to(root)}:{node.lineno}"
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module == "torch.utils.cpp_extension"
            and any(a.name == "load" for a in node.names))
    assert not offenders, (
        f"these import torch's `load` directly and so can still hang on a leftover lock: "
        f"{offenders}. Use kernels._nvcc.load_extension.")
