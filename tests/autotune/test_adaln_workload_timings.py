"""Different physical AdaLN workloads must not compete using raw milliseconds."""
import json
from types import SimpleNamespace

import pytest
import torch
import triton

from miniworld_engine.autotune import cache, capture

OP = "adaln_fwd_gate_triton"
KEY = "bfloat16|shape_key=384"

@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cache, "env_identity", lambda: "test-env")
    monkeypatch.setattr(capture, "gpu_key", lambda: "GPU")
    monkeypatch.setattr(cache, "driver_identity", lambda op: "test-driver")
    cache._load_cache.clear()
    monkeypatch.setattr(capture, "_CAPTURE", {})
    monkeypatch.setattr(capture, "_op_name", lambda at: OP)
    monkeypatch.setattr(capture, "op_identity", lambda at: "source")
    configs = [triton.Config({"BLOCK_M": 64}, num_warps=w) for w in (1, 2, 4)]
    def tuner(rows=512, strided=False):
        x = torch.empty(rows, 8, dtype=torch.bfloat16)
        if strided: x = x.t().contiguous().t()
        return SimpleNamespace(arg_names=["X", "M", "shape_key"], nargs={"X": x, "M": rows},
                               configs=configs, keys=["shape_key"])
    return configs, tuner

def publish(configs, rows, times, measurement):
    return cache.store_ranked_configs(OP, "GPU", "bfloat16", "shape_key=384",
        [(configs[i], ms) for i, ms in times], cache.config_space_hash(configs),
        configs=configs, entry_configs=[configs[i] for i, _ in times],
        op_id="source", top_k=1, measurement=measurement)

def test_workload_tracks_rows_and_strides_but_not_storage_address(env):
    _, tuner = env
    def work(t): return cache.measurement_workload(OP, t, meta={"shape_key":384})
    assert work(tuner()) == work(tuner())
    assert work(tuner()) != work(tuner(18432))
    assert work(tuner()) != work(tuner(strided=True))

def test_small_times_cannot_displace_large_workload_winner(env):
    configs, tuner = env
    small=cache.measurement_workload(OP,tuner())
    large=cache.measurement_workload(OP,tuner(18432))
    p=publish(configs,512,[(0,.01),(2,.02)],small)
    p=publish(configs,18432,[(0,.70),(2,.22)],large)
    data=json.loads(p.read_text())
    assert {c["num_warps"] for c in data["entries"][KEY]} == {1,4}
    assert cache.workload_record(data,KEY,large)["entries"][0]["num_warps"]==4
    assert cache.configs_to_bench(OP,"GPU",configs,entry_key=KEY,op_id="source",measurement=large)==[configs[1]]
    unknown=cache.measurement_workload(OP,tuner(8192))
    assert cache.configs_to_bench(OP,"GPU",configs,entry_key=KEY,op_id="source",measurement=unknown)==configs

def test_legacy_times_are_not_ranked_against_repaired_measurements(env):
    configs,tuner=env
    p=publish(configs,512,[(0,.01)],None)
    large=cache.measurement_workload(OP,tuner(18432))
    p=publish(configs,18432,[(2,.22)],large)
    assert [c["num_warps"] for c in json.loads(p.read_text())["entries"][KEY]]==[4]
    before=p.read_bytes()
    publish(configs,512,[(0,.001)],None)
    assert p.read_bytes()==before

def test_capture_and_shard_merge_preserve_each_workloads_winner(env,tmp_path):
    configs,tuner=env
    for rows,times in [(512,[(0,.01),(2,.02)]),(18432,[(0,.70),(2,.22)])]:
        at=tuner(rows)
        for idx,ms in times:capture._record_one(at,configs[idx],{"shape_key":384},ms)
    shard=tmp_path/"shard.json"
    capture.dump_shard(str(shard))
    raw=json.loads(shard.read_text())
    assert raw["_has_entries"]
    assert len(raw[OP]["measurements"][KEY])==2
    capture.merge_shards([str(shard)],top_k=1,gpu="GPU")
    data=cache._load(OP,"GPU")
    assert data is not None
    assert {c["num_warps"] for c in data["entries"][KEY]}=={1,4}
    assert len(data["measurements"][KEY])==2

def test_known_timings_require_same_workload(env,monkeypatch):
    configs,tuner=env
    at=tuner(18432)
    large=cache.measurement_workload(OP,at,meta={"shape_key":384})
    publish(configs,18432,[(2,.22)],large)
    monkeypatch.setattr(capture,"gpu_key",lambda:"GPU")
    assert list(capture._known_timings(at,{"shape_key":384}).values())==[.22]
    assert capture._known_timings(tuner(512),{"shape_key":384})=={}


def test_flush_preserves_workload_metadata(env):
    configs,tuner=env
    at=tuner(18432)
    capture._record_one(at,configs[2],{"shape_key":384},.22)
    capture.flush(top_k=1,gpu="GPU")
    data=cache._load(OP,"GPU")
    assert data is not None
    measurement=cache.measurement_workload(OP,at,meta={"shape_key":384})
    assert cache.workload_record(data,KEY,measurement)["entries"][0]["num_warps"]==4
    assert "buckets=1" in capture.summary()


def test_new_workload_cannot_hit_tritons_old_in_process_winner(env):
    _,tuner=env
    at=tuner(512);at.cache={"logical-key": "legacy-winner"}
    small=cache.measurement_workload(OP,at)
    capture._prepare_workload(at,small)
    assert not at.cache
    at.cache["logical-key"]="small-winner"
    capture._prepare_workload(at,small)
    assert at.cache["logical-key"]=="small-winner"
    capture._prepare_workload(at,cache.measurement_workload(OP,tuner(18432)))
    assert not at.cache


def test_workload_contract_covers_every_registered_triton_family(env):
    import csv
    from pathlib import Path
    _,tuner=env
    registry=Path(cache.__file__).parents[1]/"kernels/registry.csv"
    rows=list(csv.DictReader(registry.open()))
    for row in rows:
        if row["backend"]=="triton":
            assert cache.measurement_workload(row["kernel"],tuner()) is not None,row["kernel"]

def test_old_writer_cannot_poison_profiled_runtime_candidates(env):
    configs,tuner=env
    large=cache.measurement_workload(OP,tuner(18432))
    p=publish(configs,18432,[(2,.22)],large)
    data=json.loads(p.read_text())
    data["entries"][KEY]=[cache.config_to_dict(configs[0],.001)]
    assert [c["num_warps"] for c in cache.runtime_candidates(data,KEY)]==[4]


def test_helper_code_changes_invalidate_measured_candidates(env):
    configs,tuner=env
    work=cache.measurement_workload(OP,tuner(18432))
    work["implementation"]="helper-v1"
    p=publish(configs,18432,[(0,.01)],work)
    data=json.loads(p.read_text())
    assert cache.runtime_candidates(data,KEY,"helper-v2")==[]
    work["implementation"]="helper-v2"
    p=publish(configs,18432,[(2,.22)],work)
    data=json.loads(p.read_text())
    assert len(data["measurements"][KEY])==1
    assert [c["num_warps"] for c in cache.runtime_candidates(data,KEY,"helper-v2")]==[4]


@pytest.fixture(autouse=True)
def current_merge_source(monkeypatch):
    from miniworld_engine.autotune import cache_status
    monkeypatch.setattr(cache_status, "_current_op_identity", lambda op: "source")
