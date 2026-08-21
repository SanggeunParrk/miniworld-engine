"""The build's preflight must stay cheap when the config set is a real search grid.

`check()` constructs and runs every case once "at its smallest shape, which takes seconds" -- true
while every config set held one config per op. It inherits the SAME set as the build, so pointed
at a search grid the forward triggers a full autotune sweep: in the PARENT process, on one card,
with no compile fan-out, before a single unit is dispatched. Measured against configs/grid
(205,266 configs): 15 minutes in, zero units claimed, seven of eight GPUs still at 4 MiB and the
eighth at 0% utilisation.
"""
from __future__ import annotations

from miniworld_engine.autotune.builder import _one_config_per_op
from miniworld_engine.autotune.configs import _LISTS


def _fake_grid(monkeypatch, sizes):
    lists = {op: [f"cfg{i}" for i in range(n)] for op, n in sizes.items()}
    monkeypatch.setattr("miniworld_engine.autotune.configs._LISTS", lists)
    return lists


def test_the_preflight_sees_one_config_per_op(monkeypatch):
    lists = _fake_grid(monkeypatch, {"big": 15552, "small": 3, "one": 1})
    with _one_config_per_op():
        assert {op: len(v) for op, v in lists.items()} == {"big": 1, "small": 1, "one": 1}


def test_the_config_space_is_restored_afterwards(monkeypatch):
    lists = _fake_grid(monkeypatch, {"big": 15552, "small": 3})
    before = {op: list(v) for op, v in lists.items()}
    with _one_config_per_op():
        pass
    assert {op: list(v) for op, v in lists.items()} == before


def test_restoring_keeps_the_same_list_objects(monkeypatch):
    """Autotuners hold the list `configs_for` handed them BY IDENTITY -- that identity is also how
    capture recovers which op an autotuner belongs to. Rebinding instead of mutating in place
    would leave every existing autotuner pointing at the truncated list forever."""
    lists = _fake_grid(monkeypatch, {"big": 15552})
    live = lists["big"]
    with _one_config_per_op():
        assert lists["big"] is live
    assert lists["big"] is live
    assert len(live) == 15552


def test_the_space_is_restored_even_if_the_preflight_raises(monkeypatch):
    """A case that fails its smoke test must not leave every op pinned to one config -- the build
    would then run to completion and 'tune' a single candidate."""
    lists = _fake_grid(monkeypatch, {"big": 15552})
    try:
        with _one_config_per_op():
            raise RuntimeError("a case failed to build")
    except RuntimeError:
        pass
    assert len(lists["big"]) == 15552
