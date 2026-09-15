"""Corrupt/partial caches and failed native rounds must fail safely on CPU."""
import json
from dataclasses import replace

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import cache, capture, native

OP = "layernorm_linear_fwd_foldstats_sm90_cute"
GRID = [{"tile_m": 64}, {"tile_m": 128}]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_ACTIVE", replace(settings.current(), run_autotune=False))
    monkeypatch.setattr(native, "source_identity", lambda: "source")
    monkeypatch.setattr(cache, "gpu_key", lambda *_: "test-sm90")
    monkeypatch.setattr(cache, "env_identity", lambda: "env")
    monkeypatch.setattr(cache, "build_rev", lambda _: 1)
    monkeypatch.setattr(cache, "_scheme_stale", lambda *_: False)
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cache, "_load_cache", {})
    path = tmp_path / OP / "test-sm90.json"
    path.parent.mkdir()
    return path


def choose():
    return native.choose_config(OP, GRID, dtype="bfloat16", bucket="shape")


def data(entries):
    return {"op_identity": "source", "env_identity": "env", "build_rev": 1,
            "entries": {"bfloat16|shape": entries}}


@pytest.mark.parametrize("body", ['{"entries":', "null", "[]", '"cache"', "7", "true"])
def test_corrupt_root_uses_default(runtime, body):
    runtime.write_text(body)
    assert choose() == GRID[0]


@pytest.mark.parametrize("bad", [None, {}, {"kwargs": None}, {"kwargs": {"tile_m": []}},
                                 {"kwargs": GRID[0], "num_warps": "bad", "num_stages": 0}])
def test_bad_record_does_not_hide_valid_runner_up(runtime, bad):
    runtime.write_text(json.dumps(data([bad, cache.config_to_dict(GRID[1], 1.0)])))
    assert choose() == GRID[1]


@pytest.mark.parametrize("ms", [float("nan"), float("inf"), -float("inf"), 0, -1, None, True, "fast"])
def test_unusable_timing_cannot_win_from_disk(runtime, ms):
    bad = cache.config_to_dict(GRID[0])
    bad["ms"] = ms
    runtime.write_text(json.dumps(data([bad, cache.config_to_dict(GRID[1], 1.0)])))
    assert choose() == GRID[1]


def test_unmeasured_candidate_is_not_a_native_winner(runtime):
    runtime.write_text(json.dumps(data([cache.config_to_dict(GRID[0]),
                                        cache.config_to_dict(GRID[1], 1.0)])))
    assert choose() == GRID[1]


@pytest.mark.parametrize("entries", [None, [], "incomplete", 7])
def test_partial_entry_map_uses_default(runtime, entries):
    payload = data([])
    payload["entries"] = entries
    runtime.write_text(json.dumps(payload))
    assert choose() == GRID[0]


@pytest.mark.parametrize("measurement", [float("nan"), float("inf"), 0, -1])
def test_failed_measurement_is_searched_but_never_published(monkeypatch, tmp_path, measurement):
    import torch
    import triton.testing

    from miniworld_engine.autotune import native_compile
    monkeypatch.setattr(settings, "_ACTIVE", replace(settings.current(), run_autotune=True,
                                                     bench_clear_mb=0, bench_rep_ms=0))
    monkeypatch.setattr(native, "source_identity", lambda: "source")
    monkeypatch.setattr(native, "gpu_key", lambda *_: "test-sm90")
    monkeypatch.setattr(capture, "gpu_key", lambda *_: "test-sm90")
    monkeypatch.setattr(native_compile, "precompile", lambda *_: {})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(capture, "_bench_lock_acquire", lambda: None)
    monkeypatch.setattr(capture, "_bench_lock_release", lambda: None)
    monkeypatch.setattr(triton.testing, "do_bench", lambda fn, **kw: measurement)
    capture.reset()
    try:
        with pytest.raises(RuntimeError, match="every native configuration failed"):
            native.choose_config(OP, GRID, dtype="bfloat16", bucket="shape", run=lambda _: None)
        assert not native._WINNERS
        assert not capture._NATIVE_LOCK_HELD
        shard = tmp_path / "failed.json"
        capture.dump_shard(str(shard))
        payload = json.loads(shard.read_text())
        assert payload["_has_entries"] is False
        assert len(payload[OP]["searched"]["bfloat16|shape"]) == len(GRID)
        assert not payload[OP]["entries"]
    finally:
        capture.reset()
