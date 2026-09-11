"""Execute the official LayerNorm bench body on CPU with mocked launch/timing boundaries."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NamedTuple, TypedDict

import pytest
import torch
from omegaconf import DictConfig
from pydantic import BaseModel, ValidationError, model_validator

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "runners" / "bench.py"


@pytest.fixture
def runner(monkeypatch):
    names = {"BenchConfig", "AccuracyFields", "BenchResult", "tensor_metrics", "_acc_grad",
             "_flat", "as_bench_result", "is_inference_mode", "bench_kernel_layernorm_bwd"}
    tree = ast.parse(BENCH.read_text())
    body: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in names]
    namespace: dict[str, Any] = {"torch": torch, "Literal": Literal, "NamedTuple": NamedTuple, "TypedDict": TypedDict,
                 "BaseModel": BaseModel, "model_validator": model_validator, "Any": Any, "DictConfig": DictConfig, "FP32_PRECISION": 32,
                 "BF16": torch.bfloat16, "DEVICE": torch.device("cpu"), "_NO_GRAPH_TARGETS": set(),
                 "ImplementationType": type("Impl", (), {
                     "PYTORCH": type("Value", (), {"value": "pytorch"}),
                     "TRITON": type("Value", (), {"value": "triton"})})}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(BENCH), "exec"), namespace)
    namespace["BenchConfig"].model_rebuild(_types_namespace=namespace)
    observed: dict[str, Any] = {"compiled": [], "measured": [], "affine": [], "outputs": [], "dw_error": 0.0, "pins": []}

    def measured_result(**kwargs):
        observed["measured"].append(kwargs)
        observed["outputs"].append(kwargs["func"]())
        return namespace["BenchResult"](value=1.0, input_dtype=kwargs["input_dtype"],
                                         parameter_dtype=kwargs["parameter_dtype"])

    def compile_spy(fn, **kwargs):
        observed["compiled"].append(kwargs)

        def compiled():
            return fn()
        compiled.__dict__["is_compiled_probe"] = True
        return compiled

    def backward(dy, x, weight, mean, rstd):
        observed["affine"].append(weight.dtype)
        xf = x.reshape(-1, x.shape[-1]).float()
        dyf = dy.reshape_as(xf).float()
        xhat = (xf - mean[:, None]) * rstd[:, None]
        dxhat = dyf * weight.float()
        dx = rstd[:, None] * (dxhat - dxhat.mean(-1, keepdim=True)
                              - xhat * (dxhat * xhat).mean(-1, keepdim=True))
        dw = (dyf * xhat).sum(0) + observed["dw_error"]
        db = dyf.sum(0)
        return dx.to(x.dtype).view_as(x), dw.to(weight.dtype), db.to(weight.dtype)

    fake_module = ModuleType("miniworld_engine.kernels.layernorm.compile_native")
    def dispatch(dy, x, weight, mean, rstd):
        from miniworld_engine import settings
        observed["pins"].append(settings.current().layernorm_bwd_path)
        return backward(dy, x, weight, mean, rstd)

    fake_module.__dict__["_dispatch_bwd"] = dispatch
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(torch, "compile", compile_spy)
    namespace["UnsupportedBenchmark"] = NotImplementedError
    namespace["compile_for_benchmark"] = lambda fn, fullgraph=False: torch.compile(
        fn, backend="inductor", dynamic=False, fullgraph=fullgraph,
        options={"triton.cudagraphs": False})
    namespace["measured_result"] = measured_result
    return namespace, observed


def config(namespace, **kwargs):
    return namespace["BenchConfig"](target="layernorm_bwd", level="kernel", mode="training",
                                     metric="time", precision="bf16", d_pair=128, **kwargs)


@pytest.mark.parametrize("implementation", ["pytorch", "triton_atomic", "triton_persistent"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("graph", ["disabled", "manual"])
def test_requested_compile_and_graph_regime_reaches_measurement(runner, implementation, compiled, graph):
    namespace, seen = runner
    conf = config(namespace, compile=compiled, cudagraph=graph, layernorm_weight_precision=32)
    result = namespace["bench_kernel_layernorm_bwd"](conf, 2, implementation, None)
    assert result.input_dtype == "bfloat16"
    assert result.parameter_dtype == "float32"
    assert [x.dtype for x in seen["outputs"][0]] == [torch.bfloat16, torch.float32, torch.float32]
    assert len(seen["compiled"]) == int(compiled)
    if compiled:
        assert seen["compiled"][0] == {"backend": "inductor", "dynamic": False,
                                      "fullgraph": True, "options": {"triton.cudagraphs": False}}
        assert seen["measured"][0]["func"].is_compiled_probe
    assert seen["measured"][0]["conf"].cudagraph == graph
    assert result.grad_max_abs == 0
    expected_path = {"triton_atomic": "atomic", "triton_persistent": "persistent"}.get(implementation)
    assert seen["pins"] == ([] if expected_path is None else [expected_path, expected_path])


@pytest.mark.parametrize(("precision", "expected"), [(None, torch.bfloat16), ("bf16", torch.bfloat16),
                                                (32, torch.float32)])
def test_optional_affine_dtype_preserves_default_and_controls_reference(runner, precision, expected, capsys):
    namespace, seen = runner
    conf = config(namespace, cudagraph="disabled", layernorm_weight_precision=precision)
    namespace["bench_kernel_layernorm_bwd"](conf, 2, "triton_atomic", None)
    assert seen["affine"] == [expected, expected]
    report = json.loads(capsys.readouterr().out.split("LAYERNORM_BWD_GRADIENTS ")[1])
    assert set(report["gradients"]) == {"dx", "dw", "db"}
    assert report["gradients"]["dw"]["reference_dtype"] == str(expected).replace("torch.", "")


def test_dw_error_is_visible_in_csv_and_per_gradient_diagnostics(runner, capsys):
    namespace, seen = runner
    seen["dw_error"] = 1.0
    conf = config(namespace, cudagraph="disabled", layernorm_weight_precision=32)
    result = namespace["bench_kernel_layernorm_bwd"](conf, 2, "triton_atomic", None)
    report = json.loads(capsys.readouterr().out.split("LAYERNORM_BWD_GRADIENTS ")[1])
    assert result.grad_max_abs > 0.9
    assert report["gradients"]["dx"]["max_abs"] == 0
    assert report["gradients"]["dw"]["max_abs"] > 0.9
    assert report["gradients"]["db"]["max_abs"] == 0


def test_unsupported_affine_precision_is_rejected(runner):
    namespace, _ = runner
    with pytest.raises(ValidationError):
        config(namespace, layernorm_weight_precision=16)


def test_hand_cuda_is_not_called_with_fp32_affine(runner):
    namespace, seen = runner
    with pytest.raises(NotImplementedError):
        namespace["bench_kernel_layernorm_bwd"](
            config(namespace, layernorm_weight_precision=32), 2, "cuda", None)
    assert not seen["measured"]


@pytest.mark.parametrize("failure", ["compile", "measure", None])
def test_dispatch_pin_is_restored_on_success_and_failure(runner, monkeypatch, failure):
    from miniworld_engine import settings
    namespace, _ = runner
    previous = settings.configure(layernorm_bwd_path="cuda")

    def fail(*args, **kwargs):
        assert settings.current().layernorm_bwd_path == "persistent"
        raise RuntimeError("probe failure")

    try:
        if failure == "compile":
            monkeypatch.setattr(torch, "compile", fail)
        elif failure == "measure":
            namespace["measured_result"] = fail
        conf = config(namespace, compile=True, cudagraph="manual", layernorm_weight_precision=32)
        if failure:
            with pytest.raises(RuntimeError, match="probe failure"):
                namespace["bench_kernel_layernorm_bwd"](conf, 2, "triton_persistent", None)
        else:
            namespace["bench_kernel_layernorm_bwd"](conf, 2, "triton_persistent", None)
        assert settings.current().layernorm_bwd_path == "cuda"
    finally:
        settings.configure(layernorm_bwd_path=previous.layernorm_bwd_path)
