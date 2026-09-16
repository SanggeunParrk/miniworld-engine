"""Regression cases from the September build audit, independent of GPU timing."""
from __future__ import annotations

import argparse
import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest
import triton

from miniworld_engine.autotune import cache, capture, derive, round_cache


def test_stream_is_part_of_unit_identity():
    a = derive.DeriveUnit("transition", "token_pair", 128, (("d_hidden", 384),),
                          "miniworld", "bfloat16", "", "eval", None)
    b = dataclasses.replace(a, stream="token_single")
    assert a.label != b.label
    work = derive.units(derive.module_rows())
    assert len({u.label for u in work}) == len(work)


def test_single_input_cases_preserve_the_batch_axis(monkeypatch):
    import torch

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder, "_pair", lambda b, l, d, dt: torch.empty(b, l, l, d))
    wanted = {"transition"}
    found = set()
    for case in builder.cases():
        if case.name in wanted:
            args = case.inputs(2, 3, case.dims[0], torch.float32, case.stream_for(0))
            assert isinstance(args, tuple)
            assert len(args) == 1
            assert args[0].shape[:3] == (2, 3, 3)
            found.add(case.name)
    assert found == wanted


def test_registry_write_failure_preserves_previous_csv(tmp_path, monkeypatch):
    import csv
    out = tmp_path / "registry.csv"
    out.write_text("previous complete registry")

    def fail(*args):
        raise OSError("disk write failed")

    monkeypatch.setattr(csv.DictWriter, "writerows", fail)
    with pytest.raises(OSError, match="disk write"):
        derive.write_kernel_registry({}, "sm86", out)
    assert out.read_text() == "previous complete registry"
    assert not list(tmp_path.glob("*.tmp"))


def test_build_and_derivation_enumerate_identical_gpu_units(monkeypatch):
    from miniworld_engine.autotune import builder, plan
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    cases = builder.cases()
    by_name = {c.name: c for c in cases}
    actual = {plan.label(u, by_name[u.case]) for u in builder.units(cases)}
    expected = {u.label for u in derive.units(derive.module_rows(), arch="sm86")}
    assert actual == expected
    assert not any(u.impl == "cute" or u.option == ("trimul_impl", "cute")
                   for u in derive.units(derive.module_rows(), arch="sm86"))


def test_failed_derivation_does_not_publish_partial_registry(tmp_path, monkeypatch):
    from miniworld_engine import cli
    unit = derive.units(derive.module_rows())[0]
    monkeypatch.setattr(derive, "derive_all", lambda *a, **k: ({}, [(unit, "TypeError: bug")]))
    out = tmp_path / "registry.csv"
    out.write_text("previous complete result")
    args = argparse.Namespace(arch="sm86", out=str(out), per_unit="")
    assert cli.cmd_derive(args) == 1
    assert out.read_text() == "previous complete result"
    assert json.loads(out.with_suffix(".errors.json").read_text())[0]["reason"] == "TypeError: bug"


def test_coverage_rejects_empty_and_stale_entries(tmp_path, monkeypatch):
    from miniworld_engine.autotune import cache_status, configs
    cfg = triton.Config({"BLOCK": 16}, num_warps=4)
    monkeypatch.setattr(derive, "kernel_rows", lambda arch: [
        {"kernel": "op", "dtype": "float32", "bucket": "shape_key=128"}])
    monkeypatch.setattr(configs, "configs_for", lambda op: [cfg])
    monkeypatch.setattr(cache_status, "_current_op_identity", lambda op: "source")
    monkeypatch.setattr(cache, "env_identity", lambda: "env")
    monkeypatch.setattr(cache, "build_rev", lambda op: 1)
    (tmp_path / "op").mkdir()
    path = tmp_path / "op" / "GPU.json"
    data: dict[str, Any] = {"key_scheme": cache.KEY_SCHEME, "build_rev": 1, "env_identity": "env",
            "op_identity": "source", "config_space_hash": cache.config_space_hash([cfg]),
            "entries": {"float32|shape_key=128": []}}
    path.write_text(json.dumps(data))
    assert len(derive.coverage("sm86", "GPU", tmp_path)["missing"]) == 1
    data["entries"]["float32|shape_key=128"] = [cache.config_to_dict(cfg, 1.0)]
    path.write_text(json.dumps(data))
    assert not derive.coverage("sm86", "GPU", tmp_path)["missing"]
    monkeypatch.setattr(cache_status, "_current_implementation_identity", lambda op: "helper-v2")
    key = "float32|shape_key=128"
    data["measurements"] = {key: {"profile": {
        "workload": {"implementation": "helper-v1"}, "entries": data["entries"][key]}}}
    path.write_text(json.dumps(data))
    assert len(derive.coverage("sm86", "GPU", tmp_path)["missing"]) == 1
    data["measurements"] = {key: {"profile": {
        "workload": {"implementation": "helper-v2"}, "entries": data["entries"][key]}}}
    path.write_text(json.dumps(data))
    assert not derive.coverage("sm86", "GPU", tmp_path)["missing"]
    data["op_identity"] = "old source"
    path.write_text(json.dumps(data))
    assert len(derive.coverage("sm86", "GPU", tmp_path)["missing"]) == 1


def test_incremental_keeps_old_winner_beside_only_new_configs(monkeypatch):
    a, b, c = [triton.Config({"BLOCK": n}) for n in (16, 32, 64)]
    tuner = SimpleNamespace()
    monkeypatch.setattr(capture, "_op_name", lambda t: "op")
    monkeypatch.setattr(capture, "gpu_key", lambda: "GPU")
    monkeypatch.setattr(capture, "_entry_key", lambda *a: "float32|128")
    monkeypatch.setattr(capture, "op_identity", lambda t: "id")
    monkeypatch.setattr(capture, "configs_to_bench", lambda *a, **k: [c])
    monkeypatch.setattr(capture, "_known_timings", lambda *a: {repr(cache._sig(b)): 1.0})
    monkeypatch.setattr(capture, "_REUSED_TIMINGS", {})
    monkeypatch.setattr(capture, "_INCREMENTAL", True)
    assert capture._skip_measured(tuner, {}, [a, b, c]) == [c, b]


def test_round_cache_reuses_only_same_identity(tmp_path):
    identity = ("GPU", "op", "source", "env", 3, 1, "float32|128")
    with round_cache.transaction(str(tmp_path), identity) as data:
        data["config"] = 1.2
    with round_cache.transaction(str(tmp_path), identity) as data:
        assert data == {"config": 1.2}
    with round_cache.transaction(str(tmp_path), (*identity[:-1], "float32|256")) as data:
        assert data == {}


def test_cuda_pin_cannot_bypass_dtype_compatibility(monkeypatch):
    import torch

    from miniworld_engine.kernels.layernorm import compile_native as ln
    monkeypatch.setattr(ln, "_ln_bwd_override", lambda: "cuda")
    x = torch.empty(2, 128, dtype=torch.bfloat16)
    w = torch.empty(128, dtype=torch.float32)
    assert ln._resolve_bwd_path(2, 128, x, x, w, x, x) != "cuda"


def test_plan_rejects_changed_sources_or_registry(tmp_path, monkeypatch):
    from miniworld_engine.autotune import plan
    registry = tmp_path / "registry.csv"
    registry.write_text("complete registry")
    evidence = tmp_path / "units.json"
    evidence.write_text(json.dumps({"complete": True, "arch": "sm86", "units": {}, "errors": []}))
    monkeypatch.setattr(plan, "source_identity", lambda: "source1")
    plan.stamp(evidence, registry, "source1")
    assert plan.load("sm_86", registry, evidence)["complete"]
    monkeypatch.setattr(plan, "source_identity", lambda: "source2")
    with pytest.raises(ValueError, match="stale"):
        plan.load("sm86", registry, evidence)
    monkeypatch.setattr(plan, "source_identity", lambda: "source1")
    registry.write_text("partial replacement")
    with pytest.raises(ValueError, match="stale"):
        plan.load("sm86", registry, evidence)


def test_plan_keeps_every_missing_key_and_rejects_unreachable(monkeypatch):
    from miniworld_engine.autotune import plan
    cases = [SimpleNamespace(name="module")]
    work = [SimpleNamespace(case="module", length=n, label=f"module[{n}]") for n in (128, 256, 512)]
    monkeypatch.setattr(plan, "label", lambda u, c: u.label)
    evidence = {"module[128]": ["a", "b"], "module[256]": ["b", "c"], "module[512]": ["a"]}
    selected = plan.select(work, cases, evidence, {"a", "b", "c"})
    assert len(selected) == 2
    assert set().union(*(set(evidence[u.label]) for u in selected)) == {"a", "b", "c"}
    assert plan.select(work, cases, evidence, set()) == []
    with pytest.raises(ValueError, match="no runnable"):
        plan.select(work, cases, evidence, {"d"})


def test_launch_budget_does_not_retry_a_poisoned_context(monkeypatch):
    import torch

    fatal = RuntimeError("CUDA error: an illegal memory access was encountered")
    calls = []

    def launch():
        calls.append("launch")
        raise fatal

    def inner(*args, **kwargs):
        calls.append("benchmark")
        raise AssertionError("must not benchmark a poisoned context")

    tuner = SimpleNamespace(_do_bench=inner)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    capture._install_launch_budget(tuner)
    with pytest.raises(RuntimeError) as caught:
        tuner._do_bench(launch, quantiles=[0.5, 0.2, 0.8])
    assert caught.value is fatal
    assert calls == ["launch"]
