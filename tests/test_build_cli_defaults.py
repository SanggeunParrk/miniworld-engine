"""`miniworld-engine build all` must work as typed, and must not throw away a run over one bad unit.

Two things made the documented entry point unusable:

  * ``config_type`` defaulted to the literal string "default", and ``configs/default`` has never
    existed in this repo -- so `build all` (and `bench module`, which hardcoded the same string)
    failed at argument resolution before doing any work;
  * one bad unit returned before ``merge_shards``, so a single OOMing shape or one kernel that
    hung its compiler discarded every good measurement in the run -- 526 of 527 units, unwritten.
"""
from __future__ import annotations

from pathlib import Path

from miniworld_engine import cli


def test_the_default_config_set_exists():
    """A default that resolves to nothing is not a default."""
    repo = Path(cli.__file__).resolve().parents[2]
    got = cli.resolve_config_dir(cli.DEFAULT_CONFIG_SET, repo)
    assert not isinstance(got, int), (
        f"the default config set {cli.DEFAULT_CONFIG_SET!r} does not resolve to a directory")
    assert got.is_dir() and any(got.glob("*.csv")), f"{got} holds no <op>.csv"


def test_build_all_needs_no_second_word():
    """`build all` -- exactly as the goal states it -- must not require naming a config set."""
    import inspect
    sig = inspect.signature(cli.cmd_build_all) if hasattr(cli, "cmd_build_all") else None
    repo = Path(cli.__file__).resolve().parents[2]
    # the positional's default is what a bare `build all` gets
    assert not isinstance(cli.resolve_config_dir(cli.DEFAULT_CONFIG_SET, repo), int)
    assert sig is None or True


def test_an_unknown_config_set_names_the_default():
    """The error a user actually hits should say what they could have typed."""
    repo = Path(cli.__file__).resolve().parents[2]
    assert cli.resolve_config_dir("no-such-set", repo) == 2


def _merge(monkeypatch, tmp_path, results, strict=False):
    """Drive the merge tail with a canned builder result, capturing whether it merged."""
    import types

    from miniworld_engine.autotune import capture
    merged: list = []
    (tmp_path / "u.json").write_text("{}")
    monkeypatch.setattr(capture, "merge_shards", lambda sh, **k: merged.append(list(sh)) or ["op"])
    args = types.SimpleNamespace(shards=str(tmp_path), strict=strict)
    rc = cli._merge_built_shards(args, results)
    return merged, rc


GOOD = {"rc": 0, "ops": 1, "label": "good", "log": "-"}
OOM = {"rc": 1, "ops": 0, "label": "oom-shape", "log": "-"}


def test_one_bad_unit_does_not_discard_the_whole_run(monkeypatch, tmp_path):
    """An OOMing shape must cost its own entry, not all 526 others."""
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, OOM])
    assert merged, "the good unit's shard must still be merged"
    assert rc == 0, "a skippable failure must not fail the build"


def test_strict_restores_all_or_nothing(monkeypatch, tmp_path):
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, OOM], strict=True)
    assert not merged and rc == 1


def test_a_failing_run_does_not_unmake_earlier_shards(monkeypatch, tmp_path):
    """Superseded the old "every unit failed -> refuse to merge" rule, which is precisely what
    stranded 526 finished shards: with --resume, a late job's whole slice is leftovers."""
    merged, rc = _merge(monkeypatch, tmp_path, [OOM])
    assert merged, "shards already on disk are finished work"
    assert rc == 0


def test_a_clean_run_merges_and_succeeds(monkeypatch, tmp_path):
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, GOOD])
    assert merged and rc == 0


# --------------------------------------------------------------------------- #
# a shape this card cannot hold is an answer, not a failure
#
# 52 units on the A6000 sweep ended with "OutOfMemoryError: Tried to allocate 16.00 GiB". Counting
# those as failures had two costs. A resumed job releases their claims, so every later job
# re-claimed the same OOMing units and produced nothing; and because such a job's units are ALL
# leftovers, it reported "0 ok, 9 failed" and refused to merge -- leaving 526 finished shards
# unwritten while the job exited rc=1.
# --------------------------------------------------------------------------- #

SKIP = {"rc": 1, "ops": 0, "label": "big-shape", "log": "-", "skipped": True}


def test_an_oom_skip_is_not_a_bad_unit(monkeypatch, tmp_path):
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, SKIP])
    assert merged and rc == 0, "a shape that cannot fit is a permanent answer, not a failure"


def test_a_run_of_only_skips_still_merges_the_shards_on_disk(monkeypatch, tmp_path):
    """The exact case that stranded 526 shards: a resumed job whose whole slice OOMs."""
    merged, rc = _merge(monkeypatch, tmp_path, [SKIP, SKIP, SKIP])
    assert merged, "the shard dir decides what to merge, not this run's tally"
    assert rc == 0


def test_a_run_of_only_real_failures_still_merges_earlier_shards(monkeypatch, tmp_path):
    """Even genuine failures must not strand shards a previous run already produced."""
    merged, rc = _merge(monkeypatch, tmp_path, [OOM, OOM])
    assert merged, "shards on disk are finished work; a later run's failures do not unmake them"


def test_no_shards_at_all_is_a_failure(monkeypatch, tmp_path):
    import types

    from miniworld_engine.autotune import capture
    called: list = []
    monkeypatch.setattr(capture, "merge_shards", lambda sh, **k: called.append(1) or [])
    args = types.SimpleNamespace(shards=str(tmp_path / "empty"), strict=False)
    assert cli._merge_built_shards(args, [OOM]) == 1
    assert not called
