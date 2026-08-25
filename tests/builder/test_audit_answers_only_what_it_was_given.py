"""A check that cannot pass is a check nobody runs.

`dev audit` reported 139 FAIL and exited 1 on every default invocation. None of them were defects:
88 were `check_reachability` saying "registered but NO build ever captured it" when it had been
given no `--shards` to look in, and 51 were `check_cache_coverage` comparing the shipped cache
against the key `cpu` because the login node has no CUDA device. Both are missing INPUTS reported
as broken ARTIFACTS, and the same file already had the right shape for it -- `check_key_spread`
emits one WARN naming the flag and returns.

The product standard behind this: an audit whose default output is a wall of false failures is
indistinguishable from one that found something, so it stops being read.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune import configs
from miniworld_engine.build import audit


def _levels(rep, check):
    return [f.level for f in rep.of(check)]


@pytest.fixture
def registered(monkeypatch):
    """`check_reachability` reads the ops THIS process registered, and a bare test process has
    registered none -- so without this the checks below pass over an empty list and prove nothing.
    `main` gets them from `import_all_kernels()`, which costs seconds; two names are enough."""
    monkeypatch.setattr(configs, "registered_ops",
                        lambda: frozenset({"some_op", "another_op"}))
    return {"some_op", "another_op"}


def test_reachability_without_shards_asks_for_shards_instead_of_failing(registered) -> None:
    rep = audit.Report()
    audit.check_reachability(rep, [])
    assert audit.FAIL not in _levels(rep, "reach"), (
        "no --shards is an unanswered question, not a defect in every registered op")
    rows = rep.of("reach")
    assert len(rows) == 1, rows
    assert "--shards" in rows[0].detail, rows[0]
    assert rep.stats["registered"] == len(registered)


def test_coverage_without_a_card_asks_for_a_card_instead_of_failing() -> None:
    rep = audit.Report()
    audit.check_cache_coverage(rep, "cpu")
    assert audit.FAIL not in _levels(rep, "coverage"), (
        "the shipped cache is keyed by card; 'cpu' is the absence of one, not a hole")
    rows = rep.of("coverage")
    assert len(rows) == 1, rows
    assert "--gpu" in rows[0].detail, rows[0]


def test_it_still_fails_when_the_evidence_says_so(tmp_path, registered) -> None:
    """The fix must not have turned the check off. Given a shard dir that is real but records
    nothing, reachability is answered -- and the answer is a failure."""
    (tmp_path / "empty.json").write_text('{"some_op": {"entries": {}}}')
    rep = audit.Report()
    audit.check_reachability(rep, [tmp_path])
    assert audit.FAIL in _levels(rep, "reach"), (
        "a build that captured nothing must still fail reachability")
