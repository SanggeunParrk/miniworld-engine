"""Native CPU workers must match the runtime compiler ABI and isolate failures."""
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from miniworld_engine.autotune import native_compile as nc
from miniworld_engine.autotune.cute_config import (
    config_to_kwargs,
    fused_lnl_candidates,
    gated_sm90_candidates,
    lnbwd_candidates,
    plain_sm90_candidates,
)
from miniworld_engine.autotune.native import tensor_key


@pytest.mark.parametrize("case", ["m1", "m2", "swiglu", "gate", "dgrad", "dab"])
@pytest.mark.parametrize("major", ["k", "m"])
def test_worker_compiles_exactly_the_runtime_signature(monkeypatch, case, major):
    import quack.cache

    from miniworld_engine import settings
    from miniworld_engine.kernels import _quack_compat
    monkeypatch.setattr(_quack_compat, "is_compile_only", lambda: True)
    monkeypatch.setattr(quack.cache, "compile_only_mode", nullcontext)
    x = torch.empty(7, 128, dtype=torch.bfloat16)
    if major == "m":
        x = x.t().contiguous().t()
    w, y = torch.empty(128, 128, dtype=x.dtype), torch.empty(7, 128, dtype=x.dtype)
    row, col = torch.empty(1, 7), torch.empty(1, 128)
    specs = {
        "m1": ("layernorm_linear.cute.gemm_layernorm_linear", "_compile_gemm_lnl", "gemm_layernorm_linear",
               "layernorm_linear_fwd_foldstats_sm90_cute", plain_sm90_candidates()[0], [x, w, y, row, row, col, col], ()),
        "m2": ("layernorm_linear.cute.gemm_layernorm_linear_fused", "_compile_fused", "gemm_lnl_fused",
               "layernorm_linear_fwd_sm90_cute", fused_lnl_candidates()[0], [x, w, y, col, col, None, None],
               (bool(settings.current().lnl_ws), 1e-5)),
        "swiglu": ("transition.cute.gemm_transition_swiglu", "_compile_gemm_ln_swiglu", "gemm_ln_swiglu",
                   "transition_swiglu_fwd_sm90_cute", gated_sm90_candidates()[0], [x, w, y, row, row, col, col], ("None",)),
        "gate": ("transition.cute.backward_gatebwd", "_compile_gemm_dln_gatebwd", "gemm_dln_gatebwd",
                 "transition_gate_bwd_sm90_cute", gated_sm90_candidates()[0], [x, w, y, y, y, row, row, col, col], ()),
        "dgrad": ("layernorm_linear.cute.dgrad_lnbwd", "_compile", "dgrad_lnbwd_cute",
                  "layernorm_linear_bwd_dx_sm90_cute", lnbwd_candidates(128)[0], [x, w, y, col.flatten(), row.flatten()], ()),
        "dab": ("transition.cute.dab_lnbwd", "_compile", "transition_dab_lnbwd_cute",
                "transition_bwd_dx_sm90_cute", lnbwd_candidates(128)[0],
                [x, w, y, col.flatten(), row.flatten(), row.flatten()], ()),
    }
    path, compiler, launch, op, config, tensors, extra = specs[case]
    module = importlib.import_module("miniworld_engine.kernels." + path)
    monkeypatch.setattr(module, "get_device_capacity", lambda _: (9, 0))
    if hasattr(module, "is_compile_only"):
        monkeypatch.setattr(module, "is_compile_only", lambda: True)
    calls = []
    monkeypatch.setattr(module, compiler, lambda *a, **kw: calls.append((a, kw)))
    args = tensors[:5] if case == "m2" else tensors
    getattr(module, launch)(*args, config=config)
    runtime = calls.pop()
    task = nc.task_for(op, config_to_kwargs(config), tensor_key(*tensors, extra=extra))
    nc.compile_task(json.loads(json.dumps(task)))
    assert calls == [runtime]


def test_workers_are_parallel_and_failure_does_not_kill_other_candidates(tmp_path, monkeypatch):
    # A tiny executable substitutes for a compiler, exercising actual exec, signal
    # and timeout handling without importing CUDA or running an assembler.
    compiler = tmp_path / "compiler"
    compiler.write_text(f"#!{sys.executable}\n" + """
import json, os, resource, signal, sys, time
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
task = json.load(open(sys.argv[-1]))
print(json.dumps({"start": time.monotonic(), "cuda": os.environ["CUDA_VISIBLE_DEVICES"],
                  "jobs": os.environ["MAX_JOBS"]}), flush=True)
if task["op"] == "crash": os.kill(os.getpid(), signal.SIGABRT)
time.sleep(5 if task["op"] == "timeout" else .2)
print(json.dumps({"end": time.monotonic()}), flush=True)
""")
    compiler.chmod(0o755)
    monkeypatch.setattr(nc.sys, "executable", str(compiler))
    tasks = [{"op": op, "config": {}, "tensors": [], "extra": []}
             for op in ("ok1", "ok2", "crash", "timeout")]
    results = nc.run_tasks([*tasks, tasks[0]], jobs=2, timeout=1, directory=tmp_path / "out")
    assert len(results) == 4
    assert [results[nc.task_id(t)]["status"] for t in tasks] == ["ok", "ok", "failed", "timeout"]
    first, second = [Path(results[nc.task_id(t)]["log"]).read_text().splitlines() for t in tasks[:2]]
    start1, end1 = json.loads(first[0]), json.loads(first[1])
    start2 = json.loads(second[0])
    assert start1["cuda"] == start2["cuda"] == ""
    assert start1["jobs"] == "1"
    assert start2["start"] < end1["end"]


def test_nvcc_base_arch_listing_keeps_hopper_specific_target(monkeypatch):
    from miniworld_engine.kernels import _nvcc
    monkeypatch.setattr(_nvcc, "ensure_cuda_home", lambda: None)
    monkeypatch.setattr(_nvcc, "supported_arches", lambda: frozenset({"compute_80", "compute_90"}))
    assert _nvcc.gencodes("90a") == ["-gencode=arch=compute_90a,code=sm_90a"]
    assert _nvcc.gencodes("100a") == []


def test_cuda_candidates_fit_the_shared_output_shuffle():
    from miniworld_engine.autotune.hopper_cuda_config import candidates
    for kind in ("b2b", "expand_gate", "gatebwd"):
        for width in ((128, 256) if kind == "b2b" else (128, 256, 512)):
            grid = candidates(kind, width)
            assert grid
            for config in grid:
                rows = 64 * config["warpgroups"]
                assert (rows * 128 <= width * config["bn"] if kind == "b2b"
                        else rows <= config["stages"] * config["kt"])
