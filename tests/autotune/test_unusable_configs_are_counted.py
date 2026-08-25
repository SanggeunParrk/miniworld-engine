"""A config this card cannot run is correctly not stored -- and was correctly not mentioned.

One real unit: `grid=864`, and the line under it said `527 configs, best 0.0338ms`. The other 337
appeared nowhere. They are not a defect -- triton returns `[inf, inf, inf]` for a config it cannot
launch (`OutOfResources: out of resource: shared memory, Required: 514048, Hardware limit: 101376`
on an A6000) and `_record_one` must not store a config that does not run. What was a defect is
that 39% of the searched space vanished with no line saying so, in a build whose whole purpose is
to search that space.

It is not reconstructible from the shard either, and I got it wrong trying: `prune_configs`
returns the full list, so nothing is pruned, and triton swallows the OutOfResources itself, so no
exception reaches the capture layer. The count has to be kept where the drop happens.
"""
from __future__ import annotations

from typing import ClassVar

import pytest

from miniworld_engine.autotune import capture


class _Cfg:
    def __init__(self, bm):
        self.kwargs = {"BLOCK_M1": bm}
        self.num_warps = 4
        self.num_stages = 2


@pytest.fixture(autouse=True)
def _clean():
    capture.reset()
    yield
    capture.reset()


def _slot(op, grid, entries):
    capture._CAPTURE[op] = {"grid": [_Cfg(i) for i in range(grid)], "op_id": "",
                            "entries": entries}


def test_the_summary_names_what_could_not_run() -> None:
    _slot("k", 864, {("bfloat16", "shape_key=256"): {"s": (_Cfg(32), 0.0338)}})
    capture._UNUSABLE["k"] = 337
    out = capture.summary()
    assert "grid=864" in out, out
    assert "unusable=337" in out, "the 337 that could not run are still invisible"
    assert "%" in out, "a bare count does not say how much of the space it is"


def test_it_says_zero_rather_than_saying_nothing() -> None:
    """'nothing was dropped' and 'dropping is not reported' looked identical."""
    _slot("k", 12, {("bfloat16", "shape_key=256"): {"s": (_Cfg(32), 0.01)}})
    assert "unusable=0" in capture.summary()


def test_the_counter_is_fed_by_the_drop_itself(monkeypatch) -> None:
    """Not by a second pass over the data -- the reason is only known at the drop point."""
    monkeypatch.setattr(capture, "_op_name", lambda at: "kern")
    class _AT:
        configs: ClassVar = [_Cfg(32)]
    capture._record_one(_AT(), _Cfg(32), {}, float("inf"))
    assert capture._UNUSABLE == {"kern": 1}, capture._UNUSABLE
    capture._record_one(_AT(), _Cfg(64), {}, float("inf"))
    assert capture._UNUSABLE == {"kern": 2}, capture._UNUSABLE


def test_reset_clears_it_with_the_capture() -> None:
    capture._UNUSABLE["k"] = 5
    capture.reset()
    assert not capture._UNUSABLE
