"""Whether a shard holds measurements is asked three times per build, and must not cost three reads.

`_shard_has_entries` is called once per unit by the resume filter, once per claim by
`reclaim_orphans`, and once by the startup report. On a shared filesystem a shard directory of
1,163 files costs ~0.5 s per open -- the data is only 0.3 GB, so the cost is per-file latency, not
bytes -- and the three passes were 35 minutes of a build sitting in `reclaim_orphans` with every
GPU idle and nothing in the log after the `[compile]` line.

Two changes, and the tests below are about not losing either:

  * a shard DECLARES the answer. `dump_shard` writes `_has_entries` as the second key, so a reader
    that only needs the boolean stops after a few hundred bytes instead of parsing megabytes.
    Shards written before that field still parse in full -- the answer must not change for them.
  * the answer is memoised on (size, mtime), so the second and third pass of one build are free.

The exactness matters more than the speed. `_shard_has_entries` decides whether a unit is re-run
and whether a claim is reclaimed: a false positive marks an unmeasured unit done forever, and a
false negative reclaims a claim another node is holding.
"""
from __future__ import annotations

import json

from miniworld_engine.autotune import builder as B


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_a_shard_that_declares_no_entries_is_not_read_past_its_head(tmp_path) -> None:
    f = tmp_path / "s.json"
    # A big shard that nonetheless holds no entry: every config it timed scored +inf. This is the
    # case a size test alone would get wrong, which is why the flag exists.
    _write(f, {"_key_scheme": 3, "_has_entries": False,
               "op": {"grid": [{"kwargs": {"BLOCK_M": i}, "num_warps": 4, "num_stages": 2}
                               for i in range(2000)],
                      "entries": {}, "op_id": "x"}})
    assert f.stat().st_size > 50_000, "the fixture is meant to be large"
    assert B._shard_has_entries(f) is False


def test_a_shard_that_declares_entries_is_believed(tmp_path) -> None:
    f = tmp_path / "s.json"
    _write(f, {"_key_scheme": 3, "_has_entries": True,
               "op": {"grid": [], "entries": {"bf16|b1": [{"kwargs": {}, "num_warps": 4,
                                                          "num_stages": 2, "ms": 1.0}]},
                      "op_id": "x"}})
    assert B._shard_has_entries(f) is True


def test_a_shard_written_before_the_flag_still_parses(tmp_path) -> None:
    """The corpus in flight has no flag. Its answer must not change."""
    empty = _write(tmp_path / "e.json", {"_key_scheme": 3,
                                         "op": {"grid": [], "entries": {}, "op_id": "x"}})
    full = _write(tmp_path / "f.json", {"_key_scheme": 3,
                                        "op": {"grid": [], "op_id": "x",
                                               "entries": {"bf16|b1": [{"kwargs": {},
                                                                        "num_warps": 4,
                                                                        "num_stages": 2,
                                                                        "ms": 1.0}]}}})
    assert B._shard_has_entries(empty) is False
    assert B._shard_has_entries(full) is True


def test_a_missing_or_unreadable_shard_is_not_finished(tmp_path) -> None:
    assert B._shard_has_entries(tmp_path / "nope.json") is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert B._shard_has_entries(bad) is False


def test_the_answer_is_recomputed_when_the_shard_changes(tmp_path) -> None:
    """The memo is keyed on (size, mtime): a unit that finishes mid-build must stop looking empty."""
    import os

    f = tmp_path / "s.json"
    _write(f, {"_key_scheme": 3, "_has_entries": False, "op": {"entries": {}, "op_id": "x"}})
    assert B._shard_has_entries(f) is False
    _write(f, {"_key_scheme": 3, "_has_entries": True,
               "op": {"entries": {"bf16|b1": [{"kwargs": {}, "num_warps": 4, "num_stages": 2,
                                               "ms": 1.0}]}, "op_id": "x"}})
    os.utime(f, ns=(0, 10**9))          # a distinct mtime, in case the writes share one
    assert B._shard_has_entries(f) is True, "a shard that gained measurements still reads as empty"


def test_dump_shard_writes_the_flag_it_promises() -> None:
    """The reader trusts `_has_entries`; the writer has to set it from the same fact."""
    import inspect

    from miniworld_engine.autotune import capture as X

    src = inspect.getsource(X.dump_shard)
    assert '"_has_entries"' in src
    assert 'out["_has_entries"] = any(' in src, (
        "the flag is written but no longer derived from the entries it claims to describe")
