"""Build only the live B8/L8192 native BF16 MPNN training workloads.

Capture, compile budgets, incremental round storage, and merge use repository APIs.
This runner must be executed on an allocated GPU node.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from benchmarks.runners.bench import BenchConfig, measured_result
from benchmarks.runners.measurement import (
    benchmark_source_hash,
    compile_for_benchmark,
    observe_execution,
)
from benchmarks.runners.mpnn_attribution import POLICIES, model_for
from benchmarks.runners.mpnn_blocks import make_block, tensors
from benchmarks.runners.mpnn_training import logits_for, make_inputs, param_audit
from triton.runtime.autotuner import Autotuner

from miniworld_engine import settings
from miniworld_engine._atomic import write_json
from miniworld_engine.autotune import cache, capture
from miniworld_engine.autotune.configs import op_of

CONTEXTS = [f"full:{p}" for p in POLICIES if p != "pytorch"] + [
    "block:features:compute",
    "block:features:memory",
    "block:encoder:compute",
    "block:encoder:memory",
    "block:decoder:compute",
]


def context_case(name):
    if name.startswith("full:"):
        model, _ = model_for(name.split(":")[1], 0.25)
        param_audit(model)
        inputs = make_inputs(8, 8192)
        params = list(model.parameters())

        def forward():
            logits = logits_for(model, inputs)
            assert logits.dtype == torch.bfloat16
            return F.cross_entropy(logits.float(), inputs[1])

        return forward, params, params
    _, block, backend = name.split(":")
    return make_block(block, backend, 8, 8192, 0.25)


def invocation(tuner, args, kwargs):
    op = op_of(tuner.configs)
    if not op or not op.startswith("mpnn_"):
        return None
    nargs = dict(zip(tuner.arg_names, args, strict=False))
    measurement = cache.measurement_workload(op, tuner, nargs, kwargs)
    key = f"{cache.dtype_of_args(nargs)}|{cache.bucket_of_autotuner(tuner, nargs, kwargs)}"
    return {
        "op": op,
        "key": key,
        "workload_id": cache.workload_id(measurement),
        "measurement": measurement,
        "grid_count": len(tuner.configs),
    }


def run_context(name, golden=None, compare=None):
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    fn, leaves, params = context_case(name)
    compiled = compile_for_benchmark(fn, fullgraph=True)

    def step():
        for p in leaves:
            p.grad = None
        output = tensors(compiled())
        if name.startswith("full:"):
            output[0].backward()
        else:
            torch.manual_seed(42)
            upstream = tuple(torch.randn_like(x) * 0.01 for x in output)
            torch.autograd.backward(output, upstream)
        return output

    with observe_execution() as evidence:
        output = step()
        torch.cuda.synchronize()
    assert evidence.compiled and all(torch.isfinite(x).all() for x in output)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.dtype == p.dtype
        for p in leaves
    )
    result = {"context": name, "compiled": evidence.compiled, "finite_gradients": True}
    if golden is not None or compare is not None:
        # Compile/tune warmup above must not consume the seed used for this comparison.
        torch.manual_seed(20260916)
        output = step()
        torch.cuda.synchronize()
        actual = {
            "output": tuple(x.detach().cpu() for x in output),
            "gradients": tuple(p.grad.detach().cpu() for p in params),
        }
        if golden is not None:
            torch.save(actual, golden)
        if compare is not None:
            expected = torch.load(compare, map_location="cpu", weights_only=True)
            errors = {}
            for kind in ["output", "gradients"]:
                a = torch.cat([x.flatten().double() for x in actual[kind]])
                b = torch.cat([x.flatten().double() for x in expected[kind]])
                errors[kind] = float((a - b).norm() / b.norm().clamp_min(1e-30))
                assert errors[kind] < 0.02, errors
            result["fallback_vs_tuned_relative_l2"] = errors
    gc.collect()
    torch.cuda.empty_cache()
    return result


def inventory(root):
    plan: dict[str, Any] = {
        "source_hash": benchmark_source_hash(),
        "contexts": {},
        "golden": [],
    }
    original = Autotuner.run
    current = [""]

    def record(tuner, *args, **kwargs):
        row = invocation(tuner, args, kwargs)
        if row:
            plan["contexts"][current[0]][
                row["op"] + "|" + row["key"] + "|" + row["workload_id"]
            ] = row
        return original(tuner, *args, **kwargs)

    Autotuner.run = record
    for name in CONTEXTS:
        current[0] = name
        plan["contexts"][name] = {}
        print("INVENTORY", name, flush=True)
        golden = (
            root / (name.replace(":", "-") + ".golden.pt")
            if name in ["full:all_compute", "full:current"]
            else None
        )
        run_context(name, golden=golden)
        if golden:
            plan["golden"].append({"context": name, "path": str(golden)})
        write_json(root / "inventory.json", plan)
    assert benchmark_source_hash() == plan["source_hash"]
    # A minimal context cover per operation, preserving every observed physical workload.
    operations = sorted(
        {v["op"] for rows in plan["contexts"].values() for v in rows.values()}
    )
    units = []
    for op in operations:
        by_context = {
            name: {k for k, v in rows.items() if v["op"] == op}
            for name, rows in plan["contexts"].items()
        }
        remaining = set().union(*by_context.values())
        selected = []
        while remaining:
            best = max(by_context, key=lambda name: len(by_context[name] & remaining))
            selected.append(best)
            remaining -= by_context[best]
        rows = {
            k: v
            for name in selected
            for k, v in plan["contexts"][name].items()
            if v["op"] == op
        }
        units.append(
            {
                "op": op,
                "contexts": selected,
                "workloads": rows,
                "candidate_workloads": sum(v["grid_count"] for v in rows.values()),
            }
        )
    plan["units"] = sorted(units, key=lambda u: u["candidate_workloads"], reverse=True)
    plan["status"] = "complete"
    write_json(root / "inventory.json", plan)
    print(
        "PLAN",
        [
            (u["op"], len(u["workloads"]), u["candidate_workloads"])
            for u in plan["units"]
        ],
        flush=True,
    )


def build(root, op, jobs, workload_id=None):
    plan = json.loads((root / "inventory.json").read_text())
    assert benchmark_source_hash() == plan["source_hash"]
    unit = next(x for x in plan["units"] if x["op"] == op)
    if workload_id is not None:
        unit = {
            **unit,
            "workloads": {
                k: v
                for k, v in unit["workloads"].items()
                if v["workload_id"] == workload_id
            },
        }
        assert unit["workloads"], ("unknown workload", op, workload_id)
    suffix = f".{workload_id}" if workload_id is not None else ""
    shard = root / "shards" / f"{op}{suffix}.json"
    shard.parent.mkdir(exist_ok=True)
    os.environ["MINIWORLD_SMEM_LOG"] = str(shard.with_suffix(".smem"))
    settings.configure(
        run_autotune=False, capture=True, compile_jobs=jobs, predict_unusable=False
    )
    capture.set_incremental(True)
    capture.set_round_cache(str(root / ".round-cache"))
    # Other operations use their cached winner or ordinary bounded fallback. Only
    # this operation bypasses the runtime subset and searches its complete grid.
    ordinary_run = Autotuner.run
    # A bucket can contain distinct JIT specializations (e.g. zero vs nonzero
    # chunk offsets). Reusing bucket-only compile settlements would skip the
    # parallel/budgeted compiler, sending the new specialization through the
    # parent's unbounded disk-cache-hit path instead.
    ordinary_round_id = capture._round_id

    def physical_round_id(autotuner, kwargs) -> str:
        identity = ordinary_round_id(autotuner, kwargs)
        if op_of(autotuner.configs) == op:
            measurement = cache.measurement_workload(op, autotuner, meta=kwargs)
            identity += "|physical=" + cache.workload_id(measurement)
        return identity

    # Scoped runtime instrumentation, matching the capture module's own hooks.
    capture._round_id = physical_round_id  # ty: ignore[invalid-assignment]
    capture.install()
    captured_run = Autotuner.run
    capture.load_compile_state(str(shard))
    seen = set()
    status = {
        "op": op,
        "status": "running",
        "source_hash": plan["source_hash"],
        "done_workloads": [],
    }

    def save_shard():
        capture.dump_shard(str(shard))
        data = json.loads(shard.read_text())
        data = {k: v for k, v in data.items() if k.startswith("_") or k == op}
        data["_has_entries"] = bool(data.get(op, {}).get("entries"))
        write_json(shard, data)

    def selected(tuner, *args, **kwargs):
        if op_of(tuner.configs) != op:
            return ordinary_run(tuner, *args, **kwargs)
        row = invocation(tuner, args, kwargs)
        mark = row["op"] + "|" + row["key"] + "|" + row["workload_id"]
        if workload_id is not None and mark not in unit["workloads"]:
            return ordinary_run(tuner, *args, **kwargs)
        assert mark in unit["workloads"], ("unplanned workload", mark)
        if mark not in seen:
            print(
                "BUILD_ROUND",
                row["key"],
                row["workload_id"],
                "configs",
                len(tuner.configs),
                flush=True,
            )
        with cache.without_cached_subset():
            result = captured_run(tuner, *args, **kwargs)
        if mark not in seen:
            seen.add(mark)
            save_shard()
            status["done_workloads"] = sorted(seen)
            write_json(shard.with_suffix(".status.json"), status)
            print("ROUND_DONE", len(seen), "/", len(unit["workloads"]), flush=True)
        return result

    Autotuner.run = selected
    try:
        for name in unit["contexts"]:
            print("BUILD_CONTEXT", name, flush=True)
            run_context(name)
        assert seen == set(unit["workloads"]), (len(seen), len(unit["workloads"]))
        assert not capture.record_errors(), capture.record_errors()
        assert benchmark_source_hash() == plan["source_hash"]
        status["status"] = "complete"
    except Exception as exc:
        status.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_shard()
        write_json(shard.with_suffix(".status.json"), status)
        print(capture.precompile_summary(), flush=True)
        print(capture.summary(), flush=True)
        capture.shutdown_precompile()


def verify(root):
    plan = json.loads((root / "inventory.json").read_text())
    assert benchmark_source_hash() == plan["source_hash"]
    cache.clear_cache_misses()
    observed = {}
    original = Autotuner.run

    def record(tuner, *args, **kwargs):
        row = invocation(tuner, args, kwargs)
        if row:
            data = cache._load(row["op"], cache.gpu_key())
            assert data is not None
            assert not cache.measurement_mismatch(
                row["op"], data, cache.op_identity(tuner)
            )
            assert cache.runtime_candidates(
                data, row["key"], cache.implementation_identity(tuner)
            )
            # Exact measurement evidence, not merely another shape in the bucket.
            profile = cache.workload_record(data, row["key"], row["measurement"])
            assert profile and profile["entries"], ("missing physical workload", row)
            mark = row["op"] + "|" + row["key"] + "|" + row["workload_id"]
            if mark not in observed:
                remaining = cache.configs_to_bench(
                    row["op"],
                    cache.gpu_key(),
                    tuner.configs,
                    entry_key=row["key"],
                    op_id=cache.op_identity(tuner),
                    measurement=row["measurement"],
                )
                assert not remaining, ("incomplete grid", mark, len(remaining))
                row["searched_count"] = len(profile.get("searched", []))
                row["winner_count"] = len(profile["entries"])
            observed.setdefault(mark, row)
        return original(tuner, *args, **kwargs)

    Autotuner.run = record
    rows = []
    golden = {x["context"]: x["path"] for x in plan["golden"]}
    for name in CONTEXTS:
        print("VERIFY", name, flush=True)
        rows.append(run_context(name, compare=golden.get(name)))
        assert not cache.cache_misses(), cache.cache_misses()
    expected = {k for rows in plan["contexts"].values() for k in rows}
    assert set(observed) == expected
    write_json(
        root / "verification.json",
        {
            "status": "ok",
            "cache_misses": [],
            "workloads": observed,
            "rows": rows,
            "source_hash": plan["source_hash"],
        },
    )


def compare_block(root, name):
    source = benchmark_source_hash()
    cache.clear_cache_misses()
    fn, leaves, params = context_case(name)
    compiled = compile_for_benchmark(fn, fullgraph=True)
    output = tensors(compiled())
    torch.manual_seed(42)
    upstream = tuple(torch.randn_like(x) * 0.01 for x in output)
    del output

    def step():
        output = tensors(compiled())
        torch.autograd.backward(output, upstream)
        return output

    conf = BenchConfig(
        target="mpnn_actual_module",
        level="module",
        mode="training",
        metric="time",
        compile=True,
        precision="bf16",
        cudagraph="disabled",
    )
    samples = []
    for _ in range(7):
        r = measured_result(
            conf=conf,
            func=step,
            grad_to_none=leaves,
            params=params,
            is_train=True,
            input_dtype="float32" if ":features:" in name else "bfloat16",
            parameter_dtype="bfloat16+float32",
            execution_path=name,
            reference="pytorch",
        )
        assert r.compiled and r.cudagraph == "disabled"
        samples.append(r._asdict())
    assert not cache.cache_misses(), cache.cache_misses()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.dtype == p.dtype
        for p in leaves
    )
    assert benchmark_source_hash() == source
    write_json(
        root / (name.replace(":", "-") + ".json"),
        {
            "context": name,
            "status": "ok",
            "samples": samples,
            "median_ms": statistics.median(x["value"] for x in samples),
            "cache_misses": [],
            "source_hash": source,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "device": torch.cuda.get_device_name(),
            "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "batch": 8,
            "length": 8192,
            "neighbors": 48,
            "hidden": 128,
            "precision": "native_bf16_fp32_norm",
            "autocast": False,
            "dropout": 0 if ":features:" in name else 0.25,
            "compile": True,
            "cudagraph": "disabled",
            "scope": "module forward+all input/parameter backward, no optimizer",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["inventory", "build", "verify", "compare-block"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--op")
    p.add_argument("--context")
    p.add_argument("--compile-jobs", type=int, default=12)
    p.add_argument(
        "--workload-id", help="Build one inventoried profile into a separate shard"
    )
    a = p.parse_args()
    if a.mode == "build" and not a.op:
        p.error("build requires --op")
    if a.mode == "compare-block" and not a.context:
        p.error("compare-block requires --context")
    a.root.mkdir(parents=True, exist_ok=True)
    assert "A6000" in torch.cuda.get_device_name() and not torch.is_autocast_enabled(
        "cuda"
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.mode == "inventory":
        inventory(a.root)
    elif a.mode == "build":
        build(a.root, a.op, a.compile_jobs, a.workload_id)
    elif a.mode == "verify":
        verify(a.root)
    else:
        compare_block(a.root, a.context)


if __name__ == "__main__":
    main()
