"""Exercise every kernel target's dtype contract using real CPU baseline tensors."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "runners" / "bench.py"
FIXED_BEFORE = {"dual_gemm_epilogue", "gemm_epilogue", "transition_b2b", "triangle_attention",
                "bias_only_attention", "augmented_attention", "fused_ln_mask", "gemm_gate",
                "gemm_gate_bwd", "dual_gemm_epilogue_bwd", "transition_b2b_bwd", "gemm_epilogue_bwd"}
AUTOGRAD = {"adaln_bwd", "transition_b2b_bwd", "gemm_epilogue_bwd"}


class UnsupportedBenchmark(NotImplementedError):
    """The standalone AST namespace shares the harness's explicit unsupported contract."""


class Result(NamedTuple):
    value: float = 0.0
    input_dtype: str = ""
    parameter_dtype: str = ""
    input_shapes: str = ""
    output_max_abs: float | None = None
    output_rel_frob: float | None = None
    output_cosine: float | None = None
    grad_max_abs: float | None = None
    grad_rel_frob: float | None = None
    grad_cosine: float | None = None


def sources():
    tree = ast.parse(BENCH.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


FUNCTIONS = sources()
TARGETS = sorted(name.removeprefix("bench_kernel_") for name in FUNCTIONS if name.startswith("bench_kernel_"))


def implementations(target):
    labels = set()
    for node in ast.walk(FUNCTIONS[f"bench_kernel_{target}"]):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "implementation":
            for comparator in node.comparators:
                values = comparator.elts if isinstance(comparator, (ast.Set, ast.List, ast.Tuple)) else [comparator]
                labels.update(value.value for value in values if isinstance(value, ast.Constant)
                              and isinstance(value.value, str))
    return labels


@pytest.fixture
def runner():
    helper_names = {"_fwd_result", "_bwd_autograd_result", "_acc_grad", "_acc_fwd", "_flat", "tensor_metrics"}
    body: list[ast.stmt] = [node for name, node in sources().items()
                           if name in helper_names or name.startswith("bench_kernel_")]
    namespace: dict[str, Any] = {"torch": torch, "BF16": torch.bfloat16, "FP32_PRECISION": 32,
                                 "DEVICE": torch.device("cpu"), "AccuracyFields": dict,
                                 "UnsupportedBenchmark": UnsupportedBenchmark}
    seen = []

    def measured_result(**kwargs):
        with torch.set_grad_enabled(kwargs["is_train"]):
            outputs = kwargs["func"]()
        seen.append((kwargs, outputs))
        return Result(input_dtype=kwargs["input_dtype"], parameter_dtype=kwargs["parameter_dtype"])

    namespace["measured_result"] = measured_result
    exec(compile(ast.Module(body=body, type_ignores=[]), str(BENCH), "exec"), namespace)
    return namespace, seen


def config(target, **kwargs):
    defaults: dict[str, Any] = {"target": target, "precision": 32, "compile": False,
                                "cudagraph": "disabled", "d_pair": 32, "d_single_atom": 32,
                                "n_augment": 2, "mask_prob": 0.2, "layernorm_weight_precision": None}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("target", TARGETS)
def test_every_pytorch_baseline_executes_requested_fp32(target, runner):
    namespace, seen = runner
    result = namespace[f"bench_kernel_{target}"](config(target), 2, "pytorch", None)
    assert result.input_dtype == "float32"
    assert result.parameter_dtype == ("" if target in {"triangle_attention", "bias_only_attention",
                                                       "augmented_attention"} else "float32")
    outputs = seen[-1][1]
    tensors = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
    assert tensors
    assert all(t.dtype == torch.float32 for t in tensors)


@pytest.mark.parametrize(("target", "implementation"),
                         [(target, impl) for target in sorted(FIXED_BEFORE)
                          for impl in sorted(implementations(target) - {"pytorch"})])
def test_unverified_backend_fp32_is_explicitly_rejected_before_allocation(target, implementation, runner, monkeypatch):
    namespace, seen = runner

    def no_allocation(*args, **kwargs):
        raise AssertionError("dtype rejection must precede allocation")

    monkeypatch.setattr(torch, "randn", no_allocation)
    with pytest.raises(UnsupportedBenchmark, match="BF16"):
        namespace[f"bench_kernel_{target}"](config(target), 2, implementation, None)
    assert not seen


@pytest.mark.parametrize("target", sorted(AUTOGRAD))
def test_prebuilt_autograd_compile_is_rejected_before_allocating(target, runner, monkeypatch):
    namespace, seen = runner

    def no_allocation(*args, **kwargs):
        raise AssertionError("compile rejection must precede allocation")

    monkeypatch.setattr(torch, "randn", no_allocation)
    with pytest.raises(UnsupportedBenchmark, match="compiled backward"):
        namespace[f"bench_kernel_{target}"](config(target, compile=True), 2, "pytorch", None)
    assert not seen


def test_augmented_attention_uses_requested_axes_and_reports_actual_shapes(runner):
    namespace, _ = runner
    result = namespace["bench_kernel_augmented_attention"](
        config("augmented_attention", n_augment=3, d_pair=64), 2, "pytorch", None)
    shapes = json.loads(result.input_shapes)
    assert shapes["arg0"] == [3, 1, 2, 2, 32]
    assert shapes["arg3"] == [1, 2, 2, 2]


@pytest.mark.parametrize("target", ["triangle_attention", "bias_only_attention", "augmented_attention"])
def test_attention_does_not_silently_round_requested_width(target, runner):
    namespace, _ = runner
    with pytest.raises(UnsupportedBenchmark, match="multiple"):
        namespace[f"bench_kernel_{target}"](config(target, d_pair=33), 2, "pytorch", None)


def test_mask_probability_controls_actual_mask(runner):
    namespace, seen = runner
    namespace["bench_kernel_fused_ln_mask"](config("fused_ln_mask", mask_prob=1.0), 2, "pytorch", None)
    assert torch.count_nonzero(seen[-1][1]).item() == 0


def test_tail_normalizes_during_setup_only(runner, monkeypatch):
    from miniworld_engine.modules.conditioned_transition.module import (
        ConditionedTransition,
    )
    namespace, _ = runner
    original = ConditionedTransition.__init__
    calls = []

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.ada_ln_in.register_forward_hook(lambda *args: calls.append("adaln"))

    monkeypatch.setattr(ConditionedTransition, "__init__", init)
    namespace["bench_kernel_conditioned_transition_tail"](
        config("conditioned_transition_tail"), 2, "pytorch", None)
    assert calls == ["adaln"]


def test_autograd_helper_detects_parameter_gradient_error_with_correct_dx(runner):
    namespace, _ = runner
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([3.0], requires_grad=True)
    out = x * w
    result = namespace["_bwd_autograd_result"](
        config("test"), out, [x, w], torch.ones_like(out),
        [torch.tensor([3.0]), torch.tensor([0.0])], path="probe", ref="probe", dtype="float32")
    assert result.grad_max_abs == 2.0


def test_audit_covers_every_registered_target_and_implementation():
    assert len(TARGETS) == 17
    assert sum(len(implementations(target)) for target in TARGETS) == 52


def test_gate_backward_passes_required_row_scale_and_sequence_length(runner, monkeypatch):
    import sys
    from types import ModuleType

    name = "miniworld_engine.kernels.trimul_inproj.triton.gate_elem"
    module = ModuleType(name)
    calls = []

    def gate_elem_bwd(dy, x_n, proj, gate, weight, dropscale, seq_len):
        assert seq_len == 2
        assert dropscale.shape == (2, 32)
        assert dropscale.dtype == dy.dtype
        assert torch.equal(dropscale, torch.ones_like(dropscale))
        calls.append(seq_len)
        d_proj = dy.float() * gate.float()
        d_glog = dy.float() * proj.float() * gate.float() * (1 - gate.float())
        return (d_proj.to(dy.dtype), (d_glog @ weight.float().t()).to(dy.dtype),
                (x_n.float().t() @ d_glog).to(dy.dtype))

    module.__dict__["gate_elem_bwd"] = gate_elem_bwd
    monkeypatch.setitem(sys.modules, name, module)
    namespace, seen = runner
    result = namespace["bench_kernel_gemm_gate_bwd"](
        config("gemm_gate_bwd", precision="bf16-mixed"), 2, "gate_elem_bwd", None)
    assert calls == [2, 2]
    assert result.grad_rel_frob == 0
    assert len(seen[-1][1]) == 3


def test_direct_ktiled_compile_is_rejected_before_allocation(runner, monkeypatch):
    namespace, seen = runner

    def no_allocation(*args, **kwargs):
        raise AssertionError("known unsupported compilation must reject before allocation")

    monkeypatch.setattr(torch, "randn", no_allocation)
    with pytest.raises(UnsupportedBenchmark, match="no opaque compile entry"):
        namespace["bench_kernel_transition_b2b"](
            config("transition_b2b", precision="bf16-mixed", compile=True),
            2, "transition_b2b_ktiled", None)
    assert not seen
