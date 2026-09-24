"""Run every official module bench's setup on CPU, intercepting GPU measurement only."""
from __future__ import annotations

import ast
import contextlib
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple, TypedDict, cast

import pytest
import torch
from benchmarks.runners.measurement import UnsupportedBenchmark, parameter_dtype_of
from omegaconf import DictConfig
from pydantic import BaseModel, model_validator

from miniworld_engine import modules
from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention
from miniworld_engine.modules.swa_atom_attention.module import build_attention_params
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

BENCH = Path(__file__).resolve().parents[2] / "benchmarks/runners/bench.py"
TARGETS = (
    "triangle_multiplication", "triangle_multiplication_bidirectional", "triangle_attention",
    "transition", "conditioned_transition", "adaptive_layernorm", "augmented_attention_token",
    "augmented_attention_atom", "swa_atom_attention", "dit", "dit_atom", "swa_dit",
)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(("mode", "expected"), [("inference", "manual"), ("training", "disabled")])
def test_auto_graph_regime_is_uniform_for_all_modules(runner, target, mode, expected):
    namespace, _, _ = runner
    conf = namespace["BenchConfig"](target=target, level="module", mode=mode, metric="time")
    assert conf.cudagraph == expected


@pytest.fixture
def runner(monkeypatch):
    names = {"BenchConfig", "AccuracyFields", "BenchResult", "ImplementationSpec",
             "is_inference_mode", "module_miniworld_spec", "triton_miniworld_spec",
             "parse_implementation_spec", "as_bench_result", "tensor_metrics", "_paired_trimul_dropout"}
    tree = ast.parse(BENCH.read_text())
    body: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and (node.name in names or node.name.startswith("bench_module_"))]
    namespace: dict[str, Any] = {"torch": torch, "nn": torch.nn, "Literal": Literal, "NamedTuple": NamedTuple,
                 "TypedDict": TypedDict, "BaseModel": BaseModel, "model_validator": model_validator, "Any": Any, "DictConfig": DictConfig,
                 "FP32_PRECISION": 32, "DEVICE": torch.device("cpu"), "_NO_GRAPH_TARGETS": set(),
                 "MINIWORLD_IMPL": "miniworld", "DTV1_IMPL": "dtv1",
                 "BIAS_ONLY_V_IMPL": "bias_only_v", "cast": cast, "FabricLike": Any,
                 "importlib": importlib, "contextlib": contextlib,
                 "SWA3DRoPEAttention": SWA3DRoPEAttention,
                 "build_attention_params": build_attention_params,
                 "BidirectionalTriangleMultiplication": BidirectionalTriangleMultiplication}
    namespace.update({name: getattr(modules, name) for name in (
        "AdaptiveLayerNorm", "AugmentedAttentionPairBias", "ConditionedTransition",
        "ImplementationType", "Transition", "TriangleAttention", "TriangleMultiplication")})
    namespace["parameter_dtype_of"] = parameter_dtype_of
    namespace["UnsupportedBenchmark"] = UnsupportedBenchmark
    exec(compile(ast.Module(body=body, type_ignores=[]), str(BENCH), "exec"), namespace)
    namespace["BenchConfig"].model_rebuild(_types_namespace=namespace)
    seen = SimpleNamespace(models=[], compiled=[], compile_options=[], measured=[])

    def compile_spy(model, *args, **kwargs):
        seen.compiled.append(model)
        seen.compile_options.append(kwargs)

    def setup_module(model):
        seen.models.append(model)
        return model

    def measured_result(**kwargs):
        seen.measured.append(kwargs)
        return namespace["BenchResult"](value=1.0, input_dtype=kwargs["input_dtype"],
                                         parameter_dtype=kwargs["parameter_dtype"],
                                         execution_path=kwargs["execution_path"])

    # This fixture runs constructors on CPU and does not execute a FlashAttention backend.
    from miniworld_engine.modules.swa_atom_attention import module as swa_module
    monkeypatch.setattr(swa_module, "_flash_backend", lambda device: "fa2")
    monkeypatch.setattr(torch.nn.Module, "compile", compile_spy)
    namespace["compile_module_for_benchmark"] = compile_spy
    namespace["measured_result"] = measured_result
    fabric = SimpleNamespace(setup_module=setup_module,
                             backward=lambda output, gradient: output.backward(gradient))
    return namespace, seen, fabric


def config(namespace, target, **kwargs):
    defaults = {"target": target, "level": "module", "mode": "inference", "metric": "time",
                "precision": 32, "compile": True, "cudagraph": "disabled", "n_layers": 2,
                "n_augment": 1, "d_pair": 16, "d_single": 16, "d_single_token": 32,
                "d_single_atom": 32, "d_pair_atom": 8}
    defaults.update(kwargs)
    return namespace["BenchConfig"](**defaults)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("compiled", [False, True])
def test_every_module_honors_mode_depth_compile_and_dtype(runner, target, mode, compiled):
    namespace, seen, fabric = runner
    conf = config(namespace, target, mode=mode, compile=compiled)
    impl = "miniworld" if target == "swa_atom_attention" else "pytorch"
    result = namespace[f"bench_module_{target}"](conf, 2, impl, fabric)
    model = seen.models[0]
    assert all(m.training == (mode == "training") for m in model.modules())
    assert len(model.layers) == 2
    assert (model in seen.compiled) == compiled
    if target == "swa_dit" and compiled:
        assert seen.compile_options[0]["fullgraph"] is True
    assert result.input_dtype == "float32"
    assert result.parameter_dtype == namespace["parameter_dtype_of"](model)
    assert len(seen.measured) == 1
    assert seen.measured[0]["is_train"] == (mode == "training")
    assert all(m.p_drop == conf.dropout for m in model.modules() if hasattr(m, "p_drop"))
    if mode == "training":
        # Execute the actual training closure while replacing only the GPU module forward.
        # The returned tensor is required for the common finite-output validation.
        model.forward = lambda *args: args[0] * 2
        assert isinstance(seen.measured[0]["func"](), torch.Tensor)


@pytest.mark.parametrize("target", ["triangle_multiplication", "triangle_multiplication_bidirectional"])
@pytest.mark.parametrize("implementation", ["pytorch", "triton", "miniworld", "cuequivariance"])
def test_trimul_variants_have_identical_dropout_and_requested_precision(runner, monkeypatch,
                                                                      target, implementation):
    namespace, seen, fabric = runner
    # The kernel itself is GPU-only. Keep real module construction, but make forward an identity.
    def forward(model, pair, mask):
        return pair * 1.0

    monkeypatch.setattr(modules.TriangleMultiplication, "forward", forward)
    monkeypatch.setattr(BidirectionalTriangleMultiplication, "forward", forward)
    namespace[f"bench_module_{target}"](
        config(namespace, target, mode="training", precision=32), 2, implementation, fabric)
    model, reference = seen.compiled
    assert all(m.p_drop == 0.25 for root in (model, reference)
               for m in root.modules() if hasattr(m, "p_drop"))
    assert {p.dtype for p in model.parameters()} == {torch.float32}


@pytest.mark.parametrize("target", ["transition", "dit", "augmented_attention_token"])
def test_bf16_metadata_reports_actual_mixed_parameters(runner, target):
    namespace, seen, fabric = runner
    result = namespace[f"bench_module_{target}"](
        config(namespace, target, precision="bf16-mixed"), 2, "pytorch", fabric)
    assert result.input_dtype == "bfloat16"
    assert result.parameter_dtype == "bfloat16+float32"
    assert {p.dtype for p in seen.models[0].parameters()} == {torch.bfloat16, torch.float32}


@pytest.mark.parametrize("target", ["adaptive_layernorm", "augmented_attention_atom"])
def test_measurement_oom_propagates_to_failed_row_handler(runner, target):
    namespace, _, fabric = runner

    def oom(**kwargs):
        raise torch.cuda.OutOfMemoryError("contract regression")

    namespace["measured_result"] = oom
    with pytest.raises(torch.cuda.OutOfMemoryError, match="contract regression"):
        namespace[f"bench_module_{target}"](config(namespace, target), 2, "pytorch", fabric)


@pytest.mark.parametrize("implementation", ["triton", "cute", "cuda", "cuequivariance"])
def test_swa_rejects_labels_with_no_separate_implementation(runner, implementation):
    namespace, seen, fabric = runner
    with pytest.raises(UnsupportedBenchmark, match="implements pytorch and miniworld"):
        namespace["bench_module_swa_atom_attention"](
            config(namespace, "swa_atom_attention"), 2, implementation, fabric)
    assert not seen.measured


def test_dit_training_includes_pair_gradient_and_requested_mask(runner):
    namespace, seen, fabric = runner
    namespace["bench_module_dit"](
        config(namespace, "dit", mode="training", mask_prob=1.0), 2, "pytorch", fabric)
    measured = seen.measured[0]
    pair = measured["grad_to_none"][2]
    assert pair.shape == (1, 2, 2, 16)
    assert pair.requires_grad
    masks = []
    handle = seen.models[0].register_forward_pre_hook(lambda model, args: masks.append(args[3]))
    try:
        measured["func"]()
    finally:
        handle.remove()
    assert pair.grad is not None
    assert len(masks) == 1
    assert not masks[0].any()


def test_token_conditioning_uses_declared_condition_width(runner):
    namespace, seen, fabric = runner
    namespace["bench_module_augmented_attention_token"](
        config(namespace, "augmented_attention_token"), 2, "pytorch", fabric)
    model = seen.models[0]
    args_seen = []
    handle = model.register_forward_pre_hook(lambda model, args: args_seen.append(args))
    try:
        with torch.no_grad():
            seen.measured[0]["func"]()
    finally:
        handle.remove()
    assert args_seen[0][0].shape[-1] == 32
    assert args_seen[0][1].shape[-1] == 16


@pytest.mark.parametrize(("target", "implementation"), [
    ("conditioned_transition", "cute"),
    ("adaptive_layernorm", "cuda"), ("dit", "cuequivariance"), ("swa_dit", "cute"),
])
def test_unsupported_modules_raise_explicit_status_instead_of_nan(runner, target, implementation):
    namespace, seen, fabric = runner
    with pytest.raises(UnsupportedBenchmark):
        namespace[f"bench_module_{target}"](config(namespace, target), 2, implementation, fabric)
    assert not seen.measured


@pytest.mark.parametrize("target", ["triangle_multiplication", "triangle_multiplication_bidirectional",
                                    "triangle_attention"])
def test_dropout_rng_runs_inside_every_timed_training_forward(runner, monkeypatch, target):
    namespace, seen, fabric = runner
    namespace[f"bench_module_{target}"](
        config(namespace, target, mode="training"), 8, "pytorch", fabric)
    model = seen.models[0]
    # The accuracy-only shared mask must be gone; the real module creates new masks.
    method = "_make_drop_scale" if target == "triangle_attention" else "_make_drop_row_scale"
    assert all(method not in layer.__dict__ for layer in model.layers)
    calls = []
    for layer in model.layers:
        original = getattr(layer, method)
        def record(pair, p, original=original):
            scale = original(pair, p)
            calls.append((p, scale.clone()))
            return scale
        monkeypatch.setattr(layer, method, record)
    outputs = []
    for _ in range(2):
        for tensor in seen.measured[0]["grad_to_none"]:
            tensor.grad = None
        outputs.append(seen.measured[0]["func"]().detach().clone())
    assert not torch.equal(*outputs)
    assert len(calls) == 4  # two layers, two timed steps
    assert all(p == .25 and (scale == 0).any() and (scale > 0).any() for p, scale in calls)
    assert not torch.equal(calls[0][1], calls[2][1])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("mode", ["inference", "training"])
def test_default_dropout_policy_records_effective_probability(runner, target, mode):
    ns, _, _ = runner
    expected = .25 if mode == "training" and target.startswith("triangle_") else 0.
    for extra in ({}, {"dropout": "auto"}):
        conf = ns["BenchConfig"](target=target, level="module", mode=mode, metric="time", **extra)
        assert conf.dropout == expected


@pytest.mark.parametrize("kwargs", [{"dropout": -.1}, {"dropout": 1},
    {"dropout": .25, "mode": "inference"}, {"dropout": .25, "target": "transition"}])
def test_invalid_dropout_regimes_are_not_silently_ignored(runner, kwargs):
    ns, _, _ = runner
    with pytest.raises(ValueError, match="dropout"):
        config(ns, **({"target": "triangle_multiplication", "mode": "training"} | kwargs))


@pytest.mark.parametrize("target", ["triangle_multiplication", "triangle_multiplication_bidirectional"])
def test_dtv1_registers_vendor_ops_before_the_first_direct_launch(runner, monkeypatch, target):
    namespace, _, fabric = runner
    order = []
    original_import = importlib.import_module

    def import_spy(name, *args, **kwargs):
        if name == "cuequivariance_ops_torch.fused_layer_norm_torch":
            order.append("registration")
            # Registration loads CUDA vendor libraries; exercise ordering on CPU
            # and leave actual operator loading to the isolated GPU regression.
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    def launch(pair, *args, **kwargs):
        assert order[0] == "registration"
        order.append("launch")
        return torch.zeros_like(pair)

    monkeypatch.setattr(importlib, "import_module", import_spy)
    namespace["fused_triangle_multiplicative_update_dtv1"] = launch
    namespace["fused_bidirectional_dtv1"] = launch
    namespace[f"bench_module_{target}"](config(namespace, target), 2, "dtv1", fabric)
    assert order.count("launch") == 2


@pytest.mark.parametrize(("direction", "expected"), [
    ("outgoing", [True, True]), ("incoming", [False, False]), ("alternating", [True, False])])
@pytest.mark.parametrize("implementation", ["pytorch", "miniworld", "cuequivariance"])
def test_trimul_direction_reaches_every_layer(runner, monkeypatch, direction, expected, implementation):
    namespace, seen, fabric = runner
    monkeypatch.setattr(modules.TriangleMultiplication, "forward", lambda self, pair, mask: pair * 1.0)
    namespace["bench_module_triangle_multiplication"](
        config(namespace, "triangle_multiplication", trimul_direction=direction), 2, implementation, fabric)
    for model in seen.compiled:
        assert [layer.outgoing for layer in model.layers] == expected


@pytest.mark.parametrize("target", ["swa_atom_attention", "swa_dit"])
@pytest.mark.parametrize("implementation", ["pytorch", "miniworld"])
def test_swa_propagates_reference_implementation(runner, target, implementation):
    namespace, seen, fabric = runner
    namespace[f"bench_module_{target}"](config(namespace, target), 2, implementation, fabric)
    cores = [m for m in seen.compiled[0].modules() if isinstance(m, SWA3DRoPEAttention)]
    assert len(cores) == 2
    assert all((m.implementation.value == "pytorch") == (implementation == "pytorch") for m in cores)


@pytest.mark.parametrize("target", ["triangle_multiplication", "triangle_multiplication_bidirectional",
                                    "triangle_attention"])
def test_target_yaml_exposes_auto_dropout_to_hydra(runner, target):
    from omegaconf import OmegaConf
    ns, _, _ = runner
    yaml = OmegaConf.load(BENCH.parents[1] / "modules" / target / "configs" / "bench.yaml")
    assert yaml.dropout == "auto"
    yaml.mode = "training"
    resolved = ns["BenchConfig"](**cast("dict[str, Any]", OmegaConf.to_container(yaml, resolve=True)))
    assert resolved.dropout == .25
    assert resolved.cudagraph == "disabled"
    yaml.mode = "inference"
    resolved = ns["BenchConfig"](**cast("dict[str, Any]", OmegaConf.to_container(yaml, resolve=True)))
    assert resolved.dropout == 0
    assert resolved.cudagraph == "manual"


def test_atom_dit_uses_full_pair_bias_block_and_atom_dimensions(runner):
    namespace, seen, fabric = runner
    namespace["bench_module_dit_atom"](
        config(namespace, "dit_atom", mode="training", mask_prob=1.0), 2, "pytorch", fabric)
    measured = seen.measured[0]
    single, cond, pair = measured["grad_to_none"][:3]
    assert single.shape == cond.shape == (1, 1, 16, 32)
    assert pair.shape == (1, 16, 16, 8)
    from miniworld_engine.modules.dit import DiTBlock
    assert all(isinstance(layer, DiTBlock) for layer in seen.models[0].layers)
    measured["func"]()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (single, cond, pair))


@pytest.mark.parametrize("target", ["dit", "dit_atom", "augmented_attention_token",
                                    "augmented_attention_atom"])
@pytest.mark.parametrize("implementation", ["pytorch", "triton", "miniworld"])
def test_pair_bias_attention_propagates_adaln_implementation(runner, target, implementation):
    namespace, seen, fabric = runner
    namespace[f"bench_module_{target}"](
        config(namespace, target), 2, implementation, fabric)
    attentions = [m for m in seen.models[0].modules()
                  if isinstance(m, modules.AugmentedAttentionPairBias)]
    assert attentions
    for attention in attentions:
        assert attention.ada_ln_in.implementation == attention.implementation
        assert (attention.ada_ln_in._backend.value == "pytorch") == (implementation == "pytorch")
