"""Native tuning must retain measured evidence across processes and grid edits."""

import json
from typing import Any

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import (
    cache,
    capture,
    hopper_cuda_config,
    native,
    native_compile,
    native_history,
)

OP = "transition_bwd_gate_sm90_cuda"


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
    ("kind", "width"),
    [("b2b", 128), ("b2b", 256), ("expand_gate", 128), ("expand_gate", 512),
     ("gatebwd", 128), ("gatebwd", 256), ("gatebwd", 512)],
)
def test_grid_is_unique_and_every_config_compiles_distinctly(kind, width):
    configs = hopper_cuda_config.candidates(kind, width)
    assert configs
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == len(configs)
    flags = {tuple(hopper_cuda_config.defines(kind, width, c)) for c in configs}
    assert len(flags) == len(configs)


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


def test_coverage_uses_exact_entry_space():
    bucket = repr(((((264, 128), (128, 1), "torch.bfloat16"),), ()))
    declared = native.candidates_for(OP, bucket)
    cfg = cache.as_cfg_dict({"kwargs": declared[0]})
    grid = [cfg]
    h = cache.config_space_hash(grid)
    key = "bfloat16|" + bucket
    data = {"entries": {key: [cfg]}, "config_space": [repr(cache._sig_from_dict(cfg))]}
    assert native.pending_candidates(OP, data) == len(declared)
    data.update(grids={h: data["config_space"]}, entry_grids={key: [h]})
    assert native.pending_candidates(OP, data) == len(declared) - 1


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
    count = len(hopper_cuda_config.candidates("gatebwd", 128))
    first = hopper_cuda_config.candidates("gatebwd", 128)
    first[0]["bn"] = -1
    first.clear()
    second = hopper_cuda_config.candidates("gatebwd", 128)
    assert len(second) == count
    assert second[0]["bn"] != -1
