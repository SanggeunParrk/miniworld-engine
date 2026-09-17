"""Native tuning must retain measured evidence across processes and grid edits."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import (
    cache,
    capture,
    cute_config,
    native,
    native_compile,
    native_history,
)

OP = "trimul_inproj_masked_sm90_cute"


@pytest.fixture
def tuner(tmp_path, monkeypatch):
    import torch
    import triton.testing

    previous = settings.current()
    settings.configure(run_autotune=True, bench_clear_mb=1, bench_rep_ms=1)
    capture.reset()
    capture.set_incremental(True)
    capture.set_round_cache(str(tmp_path / "rounds"))
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path / "data")
    cache._load_cache.clear()
    monkeypatch.setattr(native, "gpu_key", lambda _: "test_h100")
    monkeypatch.setattr(cache, "gpu_key", lambda *_: "test_h100")
    monkeypatch.setattr(capture, "gpu_key", lambda *_: "test_h100")
    monkeypatch.setattr(cache, "driver_identity", lambda *_: "")
    monkeypatch.setattr(capture, "_bench_lock_acquire", lambda: None)
    monkeypatch.setattr(capture, "_bench_lock_release", lambda: None)
    monkeypatch.setattr(capture, "_use_a_smaller_bench_budget", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    state: dict[str, Any] = {"compiled": [], "ran": [], "last": None, "results": {}}

    def precompile(op, configs, bucket):
        state["compiled"].extend(c["tile"] for c in configs)
        return {
            i: state["results"][c["tile"]]
            for i, c in enumerate(configs)
            if c["tile"] in state["results"]
        }

    monkeypatch.setattr(native_compile, "precompile", precompile)

    def run(c):
        state["ran"].append(c["tile"])
        state["last"] = c["tile"]

    monkeypatch.setattr(
        triton.testing, "do_bench", lambda *a, **kw: float(state["last"])
    )

    def choose(tiles, bucket="physical shape A"):
        native.reset()
        return native.choose_config(
            OP, [{"tile": n} for n in tiles], dtype="bfloat16", bucket=bucket, run=run
        )

    state.update(choose=choose, run=run)
    yield state
    capture.reset()
    capture.set_round_cache("")
    cache._load_cache.clear()
    settings.configure(**vars(previous))


def test_resume_grid_add_remove_and_workload_isolation(tuner):
    t = tuner
    assert t["choose"]([2, 3]) == {"tile": 2}
    assert t["choose"]([1, 2, 3]) == {"tile": 1}
    assert t["compiled"] == [2, 3, 1]
    assert t["choose"]([3]) == {"tile": 3}
    assert t["choose"]([1, 2, 3]) == {"tile": 1}
    assert t["compiled"] == [2, 3, 1]
    t["choose"]([1, 2, 3], bucket="physical shape B")
    assert t["compiled"] == [2, 3, 1, 1, 2, 3]


def test_all_timings_survive_merge_and_topk_narrowing(tuner, tmp_path):
    t = tuner
    t["choose"](list(range(1, 9)))
    shard = tmp_path / "shard.json"
    capture.dump_shard(str(shard), unit_complete=True)
    capture.merge_shards([str(shard)], gpu="test_h100", top_k=2)
    data = cache._load(OP, "test_h100")
    assert data is not None
    record = next(iter(next(iter(data["measurements"].values())).values()))
    assert len(record["entries"]) == 2
    assert len(record["timings"]) == 8
    # A new build directory has only the committed cache, not the old journal.
    capture.reset()
    capture.set_round_cache(str(tmp_path / "new-rounds"))
    assert t["choose"]([7, 8, 9]) == {"tile": 7}
    assert t["compiled"] == list(range(1, 10))
    capture.flush(gpu="test_h100", top_k=2)
    capture.reset()
    capture.set_round_cache(str(tmp_path / "third-rounds"))
    assert t["choose"]([1, 7, 8, 9]) == {"tile": 1}
    assert t["compiled"] == list(range(1, 10))


def test_rebuild_and_benchmark_profile_force_measurement(tuner):
    t = tuner
    t["choose"]([1, 2])
    capture.set_incremental(False)
    t["choose"]([1, 2])
    capture.set_incremental(True)
    settings.configure(bench_rep_ms=2)
    t["choose"]([1, 2])
    assert t["compiled"] == [1, 2, 1, 2, 1, 2]


def test_timeout_retried_without_false_coverage(tuner):
    t = tuner
    t["results"][2] = {"status": "timeout", "log": "timeout.log"}
    with pytest.raises(RuntimeError, match="incomplete native tuning"):
        t["choose"]([1, 2])
    slot = capture._CAPTURE[OP]
    _, _, searched, _ = next(iter(capture._captured_entries(slot)))
    assert len(searched) == 1
    del t["results"][2]
    t["choose"]([1, 2])
    assert t["compiled"] == [1, 2, 2]


def test_checkpoint_survives_interruption(tmp_path):
    def interrupted():
        with native_history.session(str(tmp_path), ("identity",)) as history:
            history.record("one", {"status": "ok", "ms": 0.1})
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    with native_history.session(str(tmp_path), ("identity",)) as history:
        assert history.records["one"]["ms"] == 0.1
    with native_history.session(str(tmp_path), ("other source",)) as history:
        assert not history.records


def test_corrupt_journal_rebuilds(tmp_path):
    with native_history.session(str(tmp_path), ("identity",)) as history:
        path = history.path
    path.write_text("truncated {")
    with native_history.session(str(tmp_path), ("identity",)) as history:
        assert not history.records


@pytest.mark.parametrize(
    ("factory", "count"),
    [
        ("gated_sm90_candidates", 448),
        ("plain_sm90_candidates", 512),
        ("fused_lnl_candidates", 240),
        ("lnbwd_pp_candidates", 48),
        ("tm2_candidates", 4),
    ],
)
def test_grid_is_unique_and_roundtrips(factory, count):
    configs = getattr(cute_config, factory)()
    assert len(configs) == count
    rows = [cute_config.config_to_kwargs(c) for c in configs]
    assert len({json.dumps(c, sort_keys=True) for c in rows}) == count
    assert [cute_config.kwargs_to_config(c) for c in rows] == configs
    for c in configs:
        cute_config.validate_hopper_config(c)
        assert c.cluster_m * c.cluster_n <= 4


def test_fixed_epilogue_constraints():
    assert all(
        not c.is_dynamic_persistent and c.tile_m != 192
        for c in cute_config.fused_lnl_candidates()
    )
    assert len(cute_config.lnbwd_candidates(128)) == 48
    assert len(cute_config.lnbwd_candidates(192)) == 32
    assert len(cute_config.lnbwd_candidates(256)) == 16
    assert not cute_config.lnbwd_candidates(272)
    assert all(
        c.cluster_n == 1 and c.pingpong for c in cute_config.lnbwd_pp_candidates()
    )
    assert all(c.tile_n % 32 == 0 for c in cute_config.gated_sm90_candidates())


def test_swizzle_compile_dedup_keeps_scheduler_distinct():
    c = cute_config.gated_sm90_candidates()[0]
    bucket = repr(
        (
            (
                ((264, 128), (128, 1), "torch.bfloat16"),
                ((128, 512), (512, 1), "torch.bfloat16"),
                ((1, 264), (264, 1), "torch.float32"),
            ),
            (True,),
        )
    )
    task = lambda cfg: native_compile.task_for(
        OP, cute_config.config_to_kwargs(cfg), bucket
    )
    assert native_compile.task_id(task(c)) == native_compile.task_id(
        task(replace(c, max_swizzle_size=1))
    )
    assert native_compile.task_id(task(c)) != native_compile.task_id(
        task(replace(c, is_dynamic_persistent=True))
    )


def test_policy_edit_does_not_invalidate_source_identity(monkeypatch):
    original = Path.read_text
    native.source_identity.cache_clear()
    before = native.source_identity()

    def changed(path, *a, **kw):
        text = original(path, *a, **kw)
        return (
            text.replace("SWIZZLES = (1, 2, 4, 8)", "SWIZZLES = (1, 8)")
            if path.name == "cute_config.py"
            else text
        )

    monkeypatch.setattr(Path, "read_text", changed)
    native.source_identity.cache_clear()
    assert native.source_identity() == before
    monkeypatch.undo()
    native.source_identity.cache_clear()


def test_failure_provenance_and_rebuild_invalidates_old_winner(tuner, tmp_path):
    t = tuner
    t["choose"]([1, 2])
    capture.flush(gpu="test_h100")
    log = tmp_path / "compiler.log"
    log.write_text("ValueError: unsupported tile layout")
    t["results"][1] = {"status": "failed", "returncode": 1, "log": str(log)}
    capture.reset()
    capture.set_incremental(False)
    assert t["choose"]([1, 2]) == {"tile": 2}
    capture.flush(gpu="test_h100")
    data = cache._load(OP, "test_h100")
    assert data is not None
    record = next(iter(next(iter(data["measurements"].values())).values()))
    assert [c["kwargs"]["tile"] for c in record["timings"]] == [2]
    assert (
        native_history.compile_failure_status({"status": "failed", "returncode": -9})
        == "retryable_failure"
    )


def test_policy_change_reopens_units_without_dropping_measurements(monkeypatch):
    from miniworld_engine.autotune import builder, plan, shard

    original = Path.read_bytes
    monkeypatch.setattr(plan, "source_identity", lambda: "dispatch")
    monkeypatch.setattr(shard, "provenance", lambda: {"gpu": "test"})
    native.source_identity.cache_clear()
    implementation = native.source_identity()
    generation = builder._generation_for_work(None)

    def changed(path, *a, **kw):
        data = original(path, *a, **kw)
        return (
            data + b"\n# larger candidate policy\n"
            if path.name == "cute_config.py"
            else data
        )

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert builder._generation_for_work(None) != generation
    native.source_identity.cache_clear()
    assert native.source_identity() == implementation
    monkeypatch.undo()
    native.source_identity.cache_clear()


def test_coverage_uses_exact_entry_space():
    c = cute_config.config_to_kwargs(cute_config.gated_sm90_candidates()[0])
    cfg = cache.as_cfg_dict({"kwargs": c})
    grid = [cfg]
    h = cache.config_space_hash(grid)
    bucket = repr(((((264, 128), (128, 1), "torch.bfloat16"),), ()))
    key = "bfloat16|" + bucket
    data = {"entries": {key: [cfg]}, "config_space": [repr(cache._sig_from_dict(cfg))]}
    assert native.pending_candidates(OP, data) == 448
    data.update(grids={h: data["config_space"]}, entry_grids={key: [h]})
    assert native.pending_candidates(OP, data) == 447


def test_runtime_uses_explicit_last_native_profile(tuner, monkeypatch):
    import triton.testing

    t = tuner
    t["choose"]([1, 2])
    capture.flush(gpu="test_h100")
    capture.reset()
    settings.configure(bench_rep_ms=2)
    monkeypatch.setattr(triton.testing, "do_bench", lambda *a, **kw: 3.0 - t["last"])
    assert t["choose"]([1, 2]) == {"tile": 2}
    capture.flush(gpu="test_h100")
    selected = cache.select_config(
        OP,
        dtype="bfloat16",
        bucket="physical shape A",
        candidates=[cache.as_cfg_dict({"kwargs": {"tile": i}}) for i in (1, 2)],
        op_id=native.source_identity(),
    )
    assert selected is not None
    assert selected["kwargs"] == {"tile": 2}


def test_cached_grid_cannot_be_mutated_by_callers():
    first = cute_config.gated_sm90_candidates()
    first.clear()
    second = cute_config.gated_sm90_candidates()
    assert len(second) == 448
    first = cute_config.lnbwd_candidates(128)
    first.pop()
    assert len(cute_config.lnbwd_candidates(128)) == 48
