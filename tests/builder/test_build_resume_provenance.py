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
    data = {"_has_entries": True, "op": {"entries": {"bf16|128": [
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
