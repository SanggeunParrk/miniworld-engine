"""A shape the build did not predict should be able to BUILD, not guess and forget.

`_miss` hands back a bounded `autotune_miss_cap` subset -- 24 of however many -- because the cache
read and the build share one call site and the full grid there is 205,266 configs, which is right
for a build and ruinous for a forward. But that bounded search is not written anywhere: the next
process meets the same shape, pays for the same 24 configs, and the cache never learns a shape
`build all` failed to predict.

`autotune_on_miss_shards` is the other choice, off by default: name a directory and a miss searches
the full grid and this process's measurements land in its own shard there, for `dev merge` to fold
in. Per process, never the in-repo tree -- the rule `dump_shard` already states.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import cache, capture


class _Cfg:
    def __init__(self, **kw):
        self.kwargs = kw
        self.num_warps, self.num_stages, self.maxnreg = 4, 2, None


@pytest.fixture(autouse=True)
def _clean():
    capture_was_installed = capture._orig_bench is not None
    cache._WARNED.clear() if hasattr(cache, "_WARNED") else None
    prev = settings.configure()
    cache._ON_MISS_ARMED = False
    yield
    settings.configure(**{f: getattr(prev, f) for f in prev.__dataclass_fields__}) \
        if hasattr(prev, "__dataclass_fields__") else settings.configure()
    if not capture_was_installed:
        capture.uninstall()
    cache._ON_MISS_ARMED = False


def _miss(configs):
    return cache._miss("op_probe", "cpu", "bfloat16", "no tuned autotune cache", configs)


def test_by_default_a_miss_is_bounded_and_forgotten() -> None:
    """The shipped behaviour: a small search, and nothing written."""
    got = _miss([_Cfg(BLOCK_M1=b) for b in range(64)])
    assert got is not None
    assert len(got) == settings.current().autotune_miss_cap
    assert not cache._ON_MISS_ARMED, "nothing should have been armed"


def test_the_directory_turns_a_miss_into_a_full_search(tmp_path: Path) -> None:
    """`None` is what tells triton to keep the whole grid -- the same answer a build gets."""
    settings.configure(autotune_on_miss_shards=str(tmp_path))
    assert _miss([_Cfg(BLOCK_M1=b) for b in range(64)]) is None
    assert cache._ON_MISS_ARMED, "the capture that makes the search worth keeping was not armed"


def test_the_shard_lands_in_that_directory_and_nowhere_else(tmp_path: Path) -> None:
    """A shard per PROCESS, in the named directory. Never the committed cache: `dump_shard`'s own
    contract is that a single `merge_shards` writer is the only thing that touches the in-repo
    tree, and a runtime process is not that writer."""
    from miniworld_engine.autotune import capture

    settings.configure(autotune_on_miss_shards=str(tmp_path))
    _miss([_Cfg(BLOCK_M1=b) for b in range(64)])
    capture._CAPTURE.clear()
    cfg = _Cfg(BLOCK_M1=64)
    capture._CAPTURE["op_probe"] = {
        "grid": [cfg], "op_id": "", "searched": {},
        "entries": {("bfloat16", "shape_key=256"): {"sig": (cfg, 1.0)}},
    }
    # the atexit hook, called directly -- pytest's process is not going to exit here
    out = tmp_path / "probe.json"
    capture.dump_shard(str(out))
    assert out.exists()
    assert json.loads(out.read_text()), "the shard is empty"
    assert not list(Path(cache.__file__).parent.glob("data/**/on-miss-*.json"))


def test_the_default_stays_off() -> None:
    """It blocks the first launch at an unseen shape for the length of a build unit -- 15 s to 9
    minutes, measured in this repo's own build logs. That is an offline cost, and a default that
    hands it to an inference request would be worse than the guess it replaces."""
    assert settings.Settings().autotune_on_miss_shards == ""
