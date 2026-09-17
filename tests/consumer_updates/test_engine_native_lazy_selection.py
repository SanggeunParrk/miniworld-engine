"""Wide native search spaces must not be rebuilt on every default-path launch."""

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import cache, cute_config, native


@pytest.fixture
def resolver(monkeypatch):
    previous = settings.current()
    settings.configure(run_autotune=False)
    state = {"data": None, "converted": 0}
    monkeypatch.setattr(cache, "gpu_key", lambda *_: "test_h100")
    monkeypatch.setattr(cache, "env_identity", lambda: "compiler")
    monkeypatch.setattr(cache, "build_rev", lambda *_: 1)
    monkeypatch.setattr(cache, "_load", lambda *_: state["data"])
    monkeypatch.setattr(cache, "_warn_once", lambda *args, **kwargs: None)
    monkeypatch.setattr(native, "source_identity", lambda: "implementation")
    original = cute_config.config_to_kwargs
    space = cute_config.gated_sm90_candidates()
    cute_config._cached_candidate_signatures.cache_clear()

    def convert(config):
        state["converted"] += 1
        return original(config)

    monkeypatch.setattr(cute_config, "config_to_kwargs", convert)

    def publish(config, *, identity="implementation"):
        row = cache.as_cfg_dict({"kwargs": original(config)})
        row["ms"] = 1.0
        state["data"] = dict(
            build_rev=1,
            op_identity=identity,
            env_identity="compiler",
            key_scheme=cache.KEY_SCHEME,
            entries={"bfloat16|shape": [row]},
        )

    def resolve(candidates=None):
        return cute_config.resolve_config(
            "trimul_inproj_masked_sm90_cute",
            space if candidates is None else candidates,
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
    state["converted"] = 0
    assert resolve() == space[-1]
    assert state["converted"] == 0
    # An existing winner outside a narrowed launch space must not be returned.
    assert resolve(space[:2]) == space[0]
    # Selection is not memoized across cache publication or invalidation.
    publish(space[1])
    assert resolve() == space[1]
    state["data"] = None
    assert resolve() == space[0]


def test_lazy_space_is_repeatable_and_does_not_expose_mutable_configs():
    space = cute_config.gated_sm90_candidates()[:3]
    view = cute_config._CandidateKwargs(space)
    canonical = native._CacheConfigView(view)
    assert list(canonical) == list(canonical)
    assert canonical[:2] == list(canonical)[:2]
    value = view[0]
    value["tile_m"] = -1
    assert view[0]["tile_m"] == space[0].tile_m
