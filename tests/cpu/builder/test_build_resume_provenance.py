"""Resume never substitutes measurements from another GPU or build generation."""
from __future__ import annotations

import json

import pytest

from miniworld_engine.autotune import builder, plan, shard


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(shard, "provenance", lambda gpu=None: {
        "gpu": gpu or "current-gpu", "env_identity": "current-compiler"})


@pytest.mark.parametrize("stamp", [None,
    {"gpu": "other-gpu", "env_identity": "current-compiler"},
    {"gpu": "current-gpu", "env_identity": "old-compiler"},
    {"gpu": "current-gpu", "env_identity": "current-compiler"},
])
def test_only_matching_shard_can_skip_work(tmp_path, environment, stamp):
    data = {"_unit_complete": True, "_has_entries": True, "op": {"entries": {"bf16|128": [
        {"kwargs": {"BLOCK": 128}, "num_warps": 4, "ms": 1.0}]}}}
    if stamp is not None:
        data["_provenance"] = stamp
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(data))
    assert builder._shard_reusable(path) is (stamp == shard.provenance())


def test_generation_changes_with_source_grid_and_gpu(tmp_path, environment, monkeypatch):
    monkeypatch.setattr(plan, "source_identity", lambda: "source1")
    grid = tmp_path / "op.csv"
    grid.write_text("block\n32\n")
    first = builder._generation_for_work(tmp_path)
    assert builder._generation_for_work(tmp_path) == first
    grid.write_text("block\n64\n")
    assert builder._generation_for_work(tmp_path) != first
    grid.write_text("block\n32\n")
    monkeypatch.setattr(plan, "source_identity", lambda: "source2")
    assert builder._generation_for_work(tmp_path) != first
    monkeypatch.setattr(plan, "source_identity", lambda: "source1")
    monkeypatch.setattr(shard, "provenance", lambda: {
        "gpu": "another-gpu", "env_identity": "current-compiler"})
    assert builder._generation_for_work(tmp_path) != first


def test_driver_claims_are_separated_by_generation():
    first = builder.OpUnit("example_triton", 128, generation="gpu1")
    second = builder.OpUnit("example_triton", 128, generation="gpu2")
    assert first.stem != second.stem


@pytest.mark.parametrize("complete", [None, False, True])
def test_partial_timings_do_not_mark_a_module_complete(tmp_path, environment, complete):
    data = {"_has_entries": True, "_provenance": shard.provenance(),
            "op": {"entries": {"bf16|128": [{"ms": 1.0}]}}}
    if complete is not None:
        data["_unit_complete"] = complete
    path = tmp_path / "unit.json"
    path.write_text(json.dumps(data))
    claim = path.with_suffix(".claim")
    claim.touch()
    assert builder._shard_has_entries(path)
    assert builder._shard_reusable(path) is (complete is True)
    assert builder.reclaim_orphans(tmp_path) == ([] if complete else ["unit"])
    assert claim.exists() is (complete is True)
    assert json.loads(path.read_text()) == data


@pytest.mark.parametrize("complete", [False, True])
def test_failed_rebuild_does_not_reuse_an_old_complete_shard(tmp_path, environment, monkeypatch,
                                                          complete):
    from types import SimpleNamespace

    unit = builder.OpUnit("example_triton", 128)
    path = tmp_path / f"{unit.stem}.json"
    data = {"_has_entries": True, "_provenance": shard.provenance(),
            "_unit_complete": complete, "op": {"entries": {"bf16|128": [{"ms": 1.0}]}}}
    path.write_text(json.dumps(data))
    monkeypatch.setattr(builder, "_run_unit_process", lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr(builder, "visible_device", lambda device: "0")
    result = builder._run_unit_subprocess(unit, 0, tmp_path, tmp_path, 1)
    assert result["rc"] == 1
    assert result["ops"] == 1
    assert not path.with_suffix(".claim").exists()
    assert path.with_suffix(".failed").exists()
    assert not builder._shard_reusable(path)
    assert json.loads(path.read_text()) == data
