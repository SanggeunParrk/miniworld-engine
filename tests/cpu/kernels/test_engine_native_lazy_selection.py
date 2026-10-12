"""Wide native search spaces must not be rebuilt on every default-path launch."""

from collections.abc import Sequence
from typing import Any

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import cache, native

OP = "trimul_fwd_sm90_cuda"


class _Counting(Sequence):
    """A declared space that counts how many candidates the selector actually materialises."""

    def __init__(self, candidates, state):
        self._candidates = tuple(candidates)
        self._state = state

    def __len__(self):
        return len(self._candidates)

    def __getitem__(self, index):
        if isinstance(index, slice):
            picked = self._candidates[index]
            self._state["converted"] += len(picked)
            return [dict(c) for c in picked]
        picked = self._candidates[index]
        self._state["converted"] += 1
        return dict(picked)


@pytest.fixture
def resolver(monkeypatch):
    previous = settings.current()
    settings.configure(run_autotune=False)
    state: dict[str, Any] = {"data": None, "converted": 0}
    monkeypatch.setattr(cache, "gpu_key", lambda *_: "test_h100")
    monkeypatch.setattr(cache, "env_identity", lambda: "compiler")
    monkeypatch.setattr(cache, "build_rev", lambda *_: 1)
    monkeypatch.setattr(cache, "_load", lambda *_: state["data"])
    monkeypatch.setattr(cache, "_warn_once", lambda *args, **kwargs: None)
    monkeypatch.setattr(native, "source_identity", lambda: "implementation")
    space = [{"tile_m": m, "tile_n": n} for m in (64, 128, 192) for n in (64, 128, 256)]

    def publish(config, *, identity="implementation"):
        row = cache.as_cfg_dict({"kwargs": dict(config)})
        row["ms"] = 1.0
        state["data"] = {
            "build_rev": 1,
            "op_identity": identity,
            "env_identity": "compiler",
            "key_scheme": cache.KEY_SCHEME,
            "entries": {"bfloat16|shape": [row]},
        }

    def resolve(candidates=None):
        return native.choose_config(
            OP,
            _Counting(space if candidates is None else candidates, state),
            dtype="bfloat16",
            bucket="shape",
        )

    yield state, space, publish, resolve
    settings.configure(**previous.__dict__)


def test_missing_and_stale_cache_only_convert_default(resolver):
    state, space, publish, resolve = resolver
    assert resolve() == space[0]
    assert state["converted"] == 1
    publish(space[-1], identity="stale")
    state["converted"] = 0
    assert resolve() == space[0]
    assert state["converted"] == 1


def test_valid_cache_and_narrowed_space_still_validate_membership(resolver):
    state, space, publish, resolve = resolver
    publish(space[-1])
    assert resolve() == space[-1]
    assert state["converted"] == len(space)
    # An existing winner outside a narrowed launch space must not be returned.
    assert resolve(space[:2]) == space[0]
    # Selection is not memoized across cache publication or invalidation.
    publish(space[1])
    assert resolve() == space[1]
    state["data"] = None
    assert resolve() == space[0]


def test_lazy_space_is_repeatable_and_does_not_expose_mutable_configs():
    space = [{"tile_m": 64}, {"tile_m": 128}, {"tile_m": 192}]
    canonical = native._CacheConfigView(space)
    assert list(canonical) == list(canonical)
    assert canonical[:2] == list(canonical)[:2]
    value = canonical[0]
    value["kwargs"]["tile_m"] = -1
    assert canonical[0]["kwargs"]["tile_m"] == space[0]["tile_m"] == 64
