"""Runtime regression: a request, tracing, and reference work are not measured compilation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SPEC = importlib.util.spec_from_file_location("benchmark_measurement", ROOT / "benchmarks/runners/measurement.py")
assert SPEC is not None
assert SPEC.loader is not None
measurement = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = measurement
SPEC.loader.exec_module(measurement)


@pytest.fixture(autouse=True)
def reset():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def test_trace_without_execution_does_not_establish_compilation():
    probe = measurement.CompileProbe("callable", fullgraph=True,
                                     backend=lambda graph, inputs: lambda x: x + 1)
    with measurement.observe_execution() as observed:
        executable = probe(None, [])
    assert probe.graphs_created == 1
    assert not observed.compiled
    with pytest.raises(measurement.UnsupportedBenchmark):
        measurement.require_compile_evidence(True, observed)
    with measurement.observe_execution() as executed:
        assert executable(3) == 4
    assert executed.compiled


def test_reference_only_execution_cannot_label_an_eager_measurement_compiled():
    probe = measurement.CompileProbe("reference", fullgraph=True,
                                     backend=lambda graph, inputs: graph.forward)
    compiled = torch.compile(lambda x: x.sin() + 1, backend=probe, fullgraph=True)
    x = torch.arange(8, dtype=torch.float32)
    compiled(x)  # tracing and executing the reference happens before measurement
    with measurement.observe_execution() as observed:
        x.cos()
    assert probe.graphs_created > 0
    assert not observed.compiled
    with pytest.raises(measurement.UnsupportedBenchmark):
        measurement.require_compile_evidence(True, observed)


def test_actual_compiled_callable_provides_execution_evidence():
    probe = measurement.CompileProbe("callable", fullgraph=True,
                                     backend=lambda graph, inputs: graph.forward)
    fn = torch.compile(lambda x: x.sin() + 1, backend=probe, fullgraph=True)
    x = torch.arange(8, dtype=torch.float32)
    with measurement.observe_execution() as observed:
        torch.testing.assert_close(fn(x), x.sin() + 1)
    measurement.require_compile_evidence(True, observed)
    assert observed.scopes == {"callable:fullgraph"}
    assert len(observed.graphs) == 1


def test_noop_compile_cannot_pass_as_a_compiled_measurement():
    probe = measurement.CompileProbe("callable", fullgraph=False,
                                     backend=lambda graph, inputs: graph.forward)
    fn = torch.compile(lambda x: x, backend=probe)
    with measurement.observe_execution() as observed:
        fn(torch.ones(2))
    assert not observed.compiled
    with pytest.raises(measurement.UnsupportedBenchmark):
        measurement.require_compile_evidence(True, observed)


def test_nested_observation_does_not_leak_reference_evidence():
    probe = measurement.CompileProbe("reference", fullgraph=True,
                                     backend=lambda graph, inputs: lambda: None)
    run = probe(None, [])
    with measurement.observe_execution() as outer, measurement.observe_execution() as inner:
        run()
    assert inner.compiled
    assert not outer.compiled


def test_eager_request_rejects_observed_compiled_execution():
    evidence = measurement.ExecutionEvidence(graphs={(1, 1)})
    with pytest.raises(RuntimeError, match="compile=False"):
        measurement.require_compile_evidence(False, evidence)


def test_parameter_dtype_reports_all_parameters():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16)))
    model.register_parameter("norm", torch.nn.Parameter(torch.ones(3, dtype=torch.float32)))
    assert measurement.parameter_dtype_of(model) == "bfloat16+float32"


def test_real_inductor_options_and_execution():
    x = torch.arange(8, dtype=torch.float32)
    compiled = measurement.compile_for_benchmark(lambda value: value.sin() + 1, fullgraph=True)
    with measurement.observe_execution() as observed:
        torch.testing.assert_close(compiled(x), x.sin() + 1)
    measurement.require_compile_evidence(True, observed)
    assert observed.scopes == {"callable:fullgraph"}


def test_changed_sources_cannot_publish_the_original_identity(monkeypatch):
    monkeypatch.setattr(measurement, "benchmark_source_hash", lambda: "current")
    measurement.require_source_identity("current")
    with pytest.raises(RuntimeError, match="sources changed"):
        measurement.require_source_identity("previous")
