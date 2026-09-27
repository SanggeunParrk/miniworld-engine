"""Exercise the real common timer/CSV bodies with CPU tensors and timer boundaries."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SPEC = importlib.util.spec_from_file_location("measurement_contract_runtime", ROOT / "benchmarks/runners/measurement.py")
assert SPEC is not None
assert SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


@pytest.fixture
def runner(monkeypatch):
    from torch._dynamo.backends import registry
    original_lookup = registry.lookup_backend
    monkeypatch.setattr(registry, "lookup_backend", lambda name: (
        (lambda gm, args, **kwargs: gm.forward) if name == "inductor" else original_lookup(name)))
    tree = ast.parse((ROOT / "benchmarks/runners/bench.py").read_text())
    wanted = {"BenchResult", "measured_result", "actual_compiled_flag", "csv_row", "mode_label",
              "is_inference_mode", "result_unit"}
    body: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    ns = dict(contextlib=contextlib, torch=torch, NamedTuple=NamedTuple, Callable=Callable, BenchConfig=SimpleNamespace,
              DTV1_IMPL="dtv1", MINIWORLD_IMPL="miniworld",
              _compile_wrap_now=lambda: "custom_op", os=SimpleNamespace(environ={}),
              **{name: getattr(runtime, name) for name in ("UnsupportedBenchmark", "compile_for_benchmark",
                  "observe_execution", "parameter_dtype_of", "require_compile_evidence", "input_shapes_of",
                  "snapshot_outputs", "check_execution_outputs", "check_finite_outputs", "check_stochastic_graph_replay")})
    def no_spec(value):
        raise ValueError(value)
    ns["parse_implementation_spec"] = no_spec
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    counts = {"time": 0, "memory": 0, "capture": 0}
    def timer(func, **kwargs):
        counts["time"] += 1
        func()
        return {"median_ms": 1.0}
    def memory(func):
        counts["memory"] += 1
        func()
        return {"median_mb": 2.0}
    def capture(func, params, is_train):
        counts["capture"] += 1
        func()
        return SimpleNamespace(replay=func)
    ns.update(bench_time=timer, bench_memory=memory, capture_cudagraph=capture)
    exec(compile(ast.Module(body=body, type_ignores=[]), "<real bench bodies>", "exec"), ns)
    torch.compiler.reset()
    yield ns, counts
    torch.compiler.reset()


def conf(**kwargs):
    defaults = {"level": "kernel", "target": "probe", "compile": True, "cudagraph": "disabled", "metric": "time",
                    "mode": "inference", "precision": 32, "allow_tf32": False, "sweep_axis": "seq_len", "n_layers": 1,
                    "n_augment": 2, "mask_prob": 0.0, "dropout": 0.0, "d_pair": 4, "d_single": 4, "d_single_token": 4,
                    "d_single_atom": 4, "d_pair_atom": 4}
    return SimpleNamespace(**(defaults | kwargs))


def measure(ns, config, fn, **kwargs):
    return ns["measured_result"](conf=config, func=fn, grad_to_none=[], params=[], is_train=False,
                                input_dtype="float32", parameter_dtype="", execution_path="probe",
                                reference="torch", **kwargs)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("graph", ["disabled", "manual", "graphed"])
def test_timer_records_what_executed(runner, compiled, graph):
    ns, counts = runner
    x = torch.arange(4, dtype=torch.float32)
    result = measure(ns, conf(compile=compiled, cudagraph=graph), lambda: x.sin() + 1)
    assert result.compiled is compiled
    assert result.cudagraph == ("disabled" if graph == "disabled" else "manual")
    assert result.compiled_graphs == int(compiled)
    assert counts == {"time": 1, "memory": 0, "capture": int(graph != "disabled")}
    assert result.measurement_scope == "forward"


def test_compile_noop_is_not_a_successful_compiled_row(runner):
    ns, counts = runner
    x = torch.ones(2)
    with pytest.raises(runtime.UnsupportedBenchmark, match="no compiled graph"):
        measure(ns, conf(), lambda: x)
    assert counts["time"] == 0


def test_memory_graph_request_is_rejected_before_measuring_eager(runner):
    ns, counts = runner
    with pytest.raises(runtime.UnsupportedBenchmark, match="graph memory"):
        measure(ns, conf(metric="memory", cudagraph="manual"), lambda: torch.ones(2))
    assert counts == {"time": 0, "memory": 0, "capture": 0}


def test_memory_without_graph_reports_disabled(runner):
    ns, counts = runner
    x = torch.ones(2)
    result = measure(ns, conf(metric="memory", compile=False), lambda: x.sin())
    assert result.value == 2.0
    assert result.cudagraph == "disabled"
    assert counts["memory"] == 1
    assert counts["capture"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_result_cannot_be_csv_success(runner, value):
    ns, _ = runner
    result = ns["BenchResult"](value=value, compiled=True, cudagraph="disabled")
    row = ns["csv_row"](conf=conf(), run_name="test", device_name="CPU", seq_len=4,
                        implementation="probe", result=result)
    assert row["status"] == "failed"
    assert row["value"] is None
    assert row["compiled"] is None
    assert row["compile_requested"] is True


def test_missing_execution_evidence_cannot_be_csv_success(runner):
    ns, _ = runner
    row = ns["csv_row"](conf=conf(), run_name="test", device_name="CPU", seq_len=4,
                        implementation="probe", result=ns["BenchResult"](value=1.0))
    assert row["status"] == "failed"
    assert row["compiled"] is None


def test_failed_run_does_not_claim_requested_graph_or_compile(runner):
    ns, _ = runner
    row = ns["csv_row"](conf=conf(cudagraph="manual"), run_name="test", device_name="CPU", seq_len=4,
                        implementation="probe", result=None, status="failed", error="compile failed")
    assert row["compiled"] is None
    assert row["cudagraph"] == ""
    assert row["compile_requested"] is True
    assert row["cudagraph_requested"] == "manual"
    assert row["measurement_schema"] == 2


def test_compiled_output_change_is_not_timed(runner):
    ns, counts = runner
    x = torch.ones(2)
    ns["compile_for_benchmark"] = lambda fn: lambda: fn() + 10
    with pytest.raises(RuntimeError, match="timed execution differs"):
        measure(ns, conf(), lambda: x + 1)
    assert counts["time"] == 0


def test_corrupt_graph_replay_is_not_timed(runner):
    ns, counts = runner
    x = torch.ones(2)
    def corrupt_capture(fn, params, is_train):
        output = fn()
        return SimpleNamespace(replay=lambda: output.fill_(float("nan")))
    ns["capture_cudagraph"] = corrupt_capture
    with pytest.raises(RuntimeError, match="non-finite"):
        measure(ns, conf(compile=False, cudagraph="manual"), lambda: x + 1)
    assert counts["time"] == 0


def test_training_step_starts_with_fresh_gradients(runner):
    ns, _ = runner
    x = torch.ones(3, requires_grad=True)
    weight = torch.ones(3, requires_grad=True)
    def training_step():
        (x * weight).sum().backward()
    ns["measured_result"](conf=conf(level="module", mode="training", compile=False),
                          func=training_step, grad_to_none=[x, weight], params=[weight], is_train=True,
                          input_dtype="float32", parameter_dtype="float32", execution_path="probe",
                          reference="torch")
    torch.testing.assert_close(weight.grad, torch.ones(3))
    torch.testing.assert_close(x.grad, torch.ones(3))


@pytest.mark.parametrize(("level", "compiled"), [("kernel", False), ("module", False), ("module", True)])
def test_nan_output_is_never_a_successful_measurement(runner, level, compiled):
    ns, counts = runner
    x = torch.full((2,), float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        measure(ns, conf(level=level, compile=compiled), lambda: x)
    assert counts["time"] == 0


def test_finite_gradients_do_not_hide_nan_training_output(runner):
    ns, counts = runner
    x = torch.ones(2, requires_grad=True)
    def training_step():
        y = x + float("nan")
        y.sum().backward()
        return y
    with pytest.raises(RuntimeError, match="non-finite"):
        ns["measured_result"](conf=conf(level="module", mode="training", compile=False),
                               func=training_step, grad_to_none=[x], params=[], is_train=True,
                               input_dtype="float32", parameter_dtype="", execution_path="probe", reference="torch")
    assert counts["time"] == 0


def test_run_identity_preserves_configuration_and_separates_repetitions():
    first = runtime.make_run_provenance({"d_pair": 128, "mask_prob": 0.0}, "source-a")
    repeat = runtime.make_run_provenance({"d_pair": 128, "mask_prob": 0.0}, "source-a")
    width = runtime.make_run_provenance({"d_pair": 384, "mask_prob": 0.0}, "source-a")
    mask = runtime.make_run_provenance({"d_pair": 128, "mask_prob": 0.5}, "source-a")
    assert first["config_hash"] == repeat["config_hash"]
    assert first["run_id"] != repeat["run_id"]
    assert len({first["config_hash"], width["config_hash"], mask["config_hash"]}) == 3
