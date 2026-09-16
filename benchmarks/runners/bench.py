# vendored + trimmed from team-gm psk/benchmark : benchmarks/runners/bench.py
# Single bench entry for miniworld-engine. Drops the model-level
# Pairformer / DiffusionTransformer benches; keeps the kernel-wrapping layers.
import contextlib
import csv
import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, TypedDict, cast

# When run as `python benchmarks/runners/bench.py`, sys.path[0] is
# `.../benchmarks/runners`, which can
# shadow stdlib `profile`. Point it at the repo root and add `src/` so the
# `miniworld_engine` package imports without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path[0] = str(_REPO_ROOT)
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import hydra
import numpy as np
import torch
import torch.nn as nn
import triton
from benchmarks.runners.measurement import (
    UnsupportedBenchmark,
    benchmark_source_hash,
    check_execution_outputs,
    check_finite_outputs,
    compile_for_benchmark,
    compile_module_for_benchmark,
    input_shapes_of,
    make_run_provenance,
    observe_execution,
    parameter_dtype_of,
    require_compile_evidence,
    require_source_identity,
    snapshot_outputs,
)
from omegaconf import DictConfig
from pydantic import BaseModel, model_validator
from triton.runtime.autotuner import Autotuner

from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    AugmentedAttentionPairBias,
    ConditionedTransition,
    ImplementationType,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention
from miniworld_engine.modules.swa_atom_attention.module import build_attention_params
from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_engine.modules.triangle_multiplication.baseline_dtv1_bidir import (
    fused_bidirectional_dtv1,
)
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_engine.modules.triangle_multiplication.module import _load_cute_fns

DTV1_IMPL = "dtv1"
MINIWORLD_IMPL = "miniworld"
#: An ABLATION, not a backend: the attention logits ARE the pair bias, so there is no query, no
#: key, no qk^T and no qk RMSNorm -- `out = softmax(bias) @ v`. Everything around the core is the
#: module's own (AdaLN, the projections that remain, both sigmoid gates, the conditioning scale),
#: which is what makes the difference readable as the cost of the qk half.
#: The bias is (B, L, L, H) and carries no augmentation axis, so ONE softmax serves all A samples
#: where the full path computes A of them -- most of the training saving is there.
BIAS_ONLY_V_IMPL = "bias_only_v"
if not torch.cuda.is_available():
    msg = "CUDA is not available. Please run on a machine with a CUDA-capable GPU."
    raise RuntimeError(msg)
DEVICE = torch.device("cuda")
FP32_PRECISION = 32


#: Kept for callers inspecting benchmark support. FA2 no-grad packing now has static
#: shapes, so SWA no longer requires a target-specific graph exception.
_NO_GRAPH_TARGETS: frozenset[str] = frozenset()


class BenchConfig(BaseModel):
    d_single: int = 384
    d_pair: int = 128
    d_single_token: int = 768
    d_single_atom: int = 128
    d_pair_atom: int = 16

    n_layers: int = 1
    trimul_direction: Literal["outgoing", "incoming", "alternating"] = "outgoing"
    n_augment: int = 32
    mask_prob: float = 0.2
    min_seq_len: int = 64
    max_seq_len: int = 384
    seq_len_step: int = 64
    min_d_pair: int = 64
    max_d_pair: int = 256
    d_pair_step: int = 64
    d_pair_values: list[int] | None = None
    sweep_axis: Literal["seq_len", "d_pair"] = "seq_len"
    sweep_seq_len: int = 512

    # Dedicated parallel cache builder: when set, an autotune-capture run dumps its timings to
    # THIS shard file instead of the in-repo cache (autotune.builder merges shards).
    autotune_shard: str = ""
    #: directory of <op>.csv config files; every kernel's autotune grid comes from
    #: here. Applied BEFORE the kernels import -- triton keeps the config list it is
    #: handed only when non-empty, so a late selection has no effect.
    config_dir: str = ""
    # Pin a device-calibrated dispatch switch for the duration of a capture. The card picks
    # one side for the shapes swept here, so the other side's kernels never fire and never
    # get captured -- yet they still run in production at other shapes, with no cached
    # configs at all. A build sweeps each side explicitly. "" = let the engine decide.
    pin_gate_backend: str = ""
    # Resolved before validation: dropout-bearing training modules default to .25;
    # inference and modules without dropout use 0. Explicit 0 supports diagnostic runs.
    dropout: float = 0.0
    #: Worker processes for parallel pre-compilation of each autotune round (0 = auto).
    compile_jobs: int = 0
    #: Pin the inference LN+proj concat fusion; None = let the engine decide. Typed bool, not str:
    #: hydra parses `+pin_infer_concat=true` as a bool and a str field rejects it.
    pin_infer_concat: bool | None = None
    #: Pin transition's hand-CUDA fused b2b forward on/off; None = let the engine decide.
    pin_transition_cuda_b2b: bool | None = None

    #: What to bench, and in which of the two namespaces. `level` is not a label on `target`: the
    #: two levels are SEPARATE namespaces, and the same name legitimately exists in both -- a
    #: kernel and the module built out of it (`triangle_attention` is a kernel target AND a module
    #: target today; `transition`, `layernorm`, ... are one bench away from being both). A single
    #: flat namespace is what forced the kernel side to abbreviate its names, purely to dodge
    #: those collisions.
    #: `level` is exactly the directory the target lives under: "kernel" -> `benchmarks/kernels/`,
    #: "module" -> `benchmarks/modules/` (see `target_dir`), with no exceptions.
    target: str
    level: Literal["kernel", "module"]
    implementations: list[str] = [
        ImplementationType.PYTORCH.value,
        ImplementationType.TRITON.value,
    ]
    mode: Literal["inference", "training"]
    metric: Literal["time", "memory"]
    compile: bool = False
    # Manual graph capture wraps the measured callable after requested compilation.
    # Auto uses graphs for inference timing and no graph for training or memory.
    cudagraph: Literal["disabled", "manual", "graphed", "auto"] = "auto"
    allow_tf32: bool = True
    precision: Literal[32, "bf16", "bf16-mixed"] = 32
    #: LayerNorm backward affine dtype; None preserves the input dtype.
    layernorm_weight_precision: Literal[32, "bf16"] | None = None
    #: Opt-in escape hatch for the "ref1" eager floor: an uncompiled, un-graphed baseline is
    #: normally refused (a raw eager number must never be mistaken for the shipped kernel), but the
    #: two-reference methodology wants exactly that point (ref1 = compile off + no graph, ref2 =
    #: compile on + graph auto), so a run that sets this deliberately is allowed through.
    allow_eager: bool = False
    name_suffix: str = ""
    swa_component: Literal["block", "modulation", "rope", "swiglu", "residual", "sigmoid_gate"] = "block"
    swa_kernels: list[str] | None = None
    swa_active_gates: bool = False

    @model_validator(mode="before")
    @classmethod
    def _resolve_module_augmentation(cls, values: Any) -> Any:
        # Resolve before int validation so all consumers and CSVs see a concrete count.
        # Explicit counts remain usable for sweeps; module defaults follow the workload.
        if (isinstance(values, (dict, DictConfig)) and values.get("level") == "module"
                and values.get("n_augment", "auto") == "auto"):
            values = dict(values)
            values["n_augment"] = 5 if values.get("mode") == "inference" else 48
        if isinstance(values, (dict, DictConfig)):
            values = dict(values)
            if values.get("dropout", "auto") == "auto":
                active = (values.get("level") == "module"
                          and values.get("mode") != "inference"
                          and values.get("target") in {
                              "triangle_multiplication", "triangle_multiplication_bidirectional",
                              "triangle_attention"})
                values["dropout"] = 0.25 if active else 0.0
        return values

    @model_validator(mode="after")
    def _resolve_auto_cudagraph(self) -> "BenchConfig":
        if self.cudagraph == "auto":
            # All inference timing targets, including SWA, capture the actual measured call.
            # A capture failure is reported; it never silently changes this request to OFF.
            if self.metric == "memory":
                self.cudagraph = "disabled"
            else:
                self.cudagraph = "manual" if is_inference_mode(self.mode) else "disabled"
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.level == "module" and self.dropout:
            if self.target not in {"triangle_multiplication", "triangle_multiplication_bidirectional",
                                   "triangle_attention"}:
                raise ValueError(f"{self.target} has no module dropout")
            if is_inference_mode(self.mode):
                raise ValueError("inference requires dropout=0 (or auto)")
            if self.cudagraph != "disabled":
                raise ValueError("training with dropout requires cudagraph=disabled")
        if not self.implementations:
            raise ValueError("implementations must not be empty")
        if self.n_layers < 1 or self.n_augment < 1:
            raise ValueError("n_layers and n_augment must be positive")
        if self.sweep_axis == "seq_len":
            if self.min_seq_len < 1 or self.max_seq_len < self.min_seq_len or self.seq_len_step < 1:
                raise ValueError("seq_len sweep must contain at least one positive point")
        elif self.d_pair_values is not None:
            if not self.d_pair_values or any(value < 1 for value in self.d_pair_values):
                raise ValueError("d_pair_values must contain positive points")
        elif self.min_d_pair < 1 or self.max_d_pair < self.min_d_pair or self.d_pair_step < 1:
            raise ValueError("d_pair sweep must contain at least one positive point")
        return self


def is_inference_mode(mode: str) -> bool:
    return mode == "inference"


def mode_label(mode: str) -> str:
    return "inference" if is_inference_mode(mode) else "training"


class ImplementationSpec(NamedTuple):
    impl: ImplementationType
    ln_impl: ImplementationType | None
    label: str


class FabricLike(Protocol):
    """The three methods the benches use on ``fabric``.

    Not ``lightning.Fabric``: this harness deliberately passes a no-op shim instead (see
    ``_NoFabric`` in ``run_bench`` -- Fabric's setup_module wrapper and backward add ~110us/step
    of casts and copies that have nothing to do with the kernel under test, and compress the
    speedup ratios). Annotating the real class was therefore wrong at every call site, and it was
    the only reason this module imported lightning at all.
    """

    def launch(self) -> None: ...
    def setup_module(self, module: nn.Module) -> nn.Module: ...
    def backward(self, tensor: torch.Tensor, gradient: torch.Tensor) -> None: ...


class AccuracyFields(TypedDict, total=False):
    """The correctness columns of :class:`BenchResult`, as a `_replace(**acc)` payload.

    A bare `dict[str, float]` cannot be splatted into `_replace`: the synthesized signature
    types each field separately, and a joined `float` value matches none of them. Twenty of this
    file's type findings were this one dict shape reused at five call sites.
    """

    output_max_abs: float
    output_rel_frob: float
    output_cosine: float
    grad_max_abs: float
    grad_rel_frob: float
    grad_cosine: float


class BenchResult(NamedTuple):
    value: float
    input_dtype: str = ""
    parameter_dtype: str = ""
    execution_path: str = ""
    reference: str = ""
    output_max_abs: float | None = None
    output_rel_frob: float | None = None
    output_cosine: float | None = None
    grad_max_abs: float | None = None
    grad_rel_frob: float | None = None
    grad_cosine: float | None = None
    compiled: bool | None = None
    cudagraph: str = ""
    compile_scope: str = ""
    compiled_graphs: int = 0
    measurement_scope: str = ""
    input_shapes: str = ""
    execution_validation: str = ""


def module_miniworld_spec(raw: str) -> ImplementationSpec:
    if raw.strip().lower() == MINIWORLD_IMPL:
        # MINIWORLD, not CUEQUIVARIANCE. `resolve()` sends a CUEQUIVARIANCE request for any
        # non-trimul op to KernelBackend.PYTORCH, so the old mapping made `implementations=
        # [miniworld]` bench the pytorch reference under the "ours" label for the two module
        # benches that use this spec (transition, conditioned_transition). Silent: the sweep
        # reported plausible times (3.16 ms at L=384/d=128, vs 0.65 ms for the triton path)
        # and the autotune-capture builder recorded NOTHING, because no triton kernel ever ran.
        return ImplementationSpec(ImplementationType.MINIWORLD, None, raw)
    return parse_implementation_spec(raw)


def triton_miniworld_spec(raw: str) -> ImplementationSpec:
    if raw.strip().lower() == MINIWORLD_IMPL:
        return ImplementationSpec(ImplementationType.TRITON, None, raw)
    if raw.strip().lower() == BIAS_ONLY_V_IMPL:
        return ImplementationSpec(ImplementationType.PYTORCH, None, raw)
    return parse_implementation_spec(raw)


def parse_implementation_spec(raw: str) -> ImplementationSpec:
    key = raw.strip().lower()
    if key == MINIWORLD_IMPL:
        return ImplementationSpec(ImplementationType.CUTE, None, raw)
    if key in {impl.value for impl in ImplementationType}:
        return ImplementationSpec(ImplementationType(key), None, key)
    if key in {"triton_pytorch_ln", "triton-ln-pytorch", "triton_ln_pytorch"}:
        return ImplementationSpec(ImplementationType.TRITON, ImplementationType.PYTORCH, raw)
    if key in {
        "triton_ln",
        "triton_kernel_ln",
        "triton_dispatch_ln",
        "triton-ln-kernel",
        "triton-ln-dispatch",
    }:
        return ImplementationSpec(ImplementationType.TRITON, ImplementationType.CUDA, raw)
    msg = f"Unknown implementation spec: {raw!r}"
    raise ValueError(msg)


def bench_memory(func: Callable, warmup: int = 3, rep: int = 10) -> dict[str, float]:
    memories = []

    for i in range(warmup + rep):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize(DEVICE)

        start_mem = torch.cuda.memory_allocated(DEVICE)
        func()
        torch.cuda.synchronize(DEVICE)
        peak_mem = torch.cuda.max_memory_allocated(DEVICE)

        delta_mb = (peak_mem - start_mem) / 1024 / 1024

        if i >= warmup:
            memories.append(delta_mb)

    return {
        "median_mb": float(np.median(memories)),
        "mean_mb": float(np.mean(memories)),
        "min_mb": float(np.min(memories)),
        "max_mb": float(np.max(memories)),
        "std_mb": float(np.std(memories)),
    }


def bench_time(
    func: Callable,
    warmup: int = 10,
    rep: int = 100,
    grad_to_none: list | None = None,
) -> dict[str, float]:
    quantiles = [0.5, 0.2, 0.8]
    median, p20, p80 = triton.testing.do_bench(
        func,
        warmup=warmup,
        rep=rep,
        quantiles=quantiles,
        grad_to_none=grad_to_none or [],
    )

    return {
        "median_ms": median,
        "p20_ms": p20,
        "p80_ms": p80,
    }


def measured_result(
    *, conf: BenchConfig, func: Callable, grad_to_none: list, params: list,
    is_train: bool, input_dtype: str, parameter_dtype: str,
    execution_path: str, reference: str,
) -> BenchResult:
    """Measure one callable and attach observed execution, never copied request flags."""
    if conf.metric == "memory" and conf.cudagraph != "disabled":
        raise UnsupportedBenchmark("graph memory measurement is not implemented; select cudagraph=disabled")
    if conf.cudagraph == "graphed" and is_train:
        raise UnsupportedBenchmark(
            "graphed training requires tensor-returning callables, but this harness times backward closures; "
            "select cudagraph=manual")
    scope = ("backward" if conf.target.endswith("_bwd") else "forward") if conf.level == "kernel" else (
        "forward_backward" if is_train else "forward")
    import json

    actual_shapes = input_shapes_of(func)
    execution_checks = {}
    # For pure kernels, compare the newly compiled callable with the exact eager
    # callable whose target-specific reference metrics were computed above.
    eager_snapshot = None
    if conf.level == "kernel" and conf.compile:
        original = getattr(func, "_torchdynamo_orig_callable", func)
        with torch.no_grad() if not is_train else contextlib.nullcontext():
            eager_snapshot = snapshot_outputs(original())
    # Module forwards were compiled before construction of their timed training/inference step.
    # Pure kernel launchers must compile the actual timed callable here.
    if conf.level == "kernel" and conf.compile and not hasattr(func, "_benchmark_compile_probe"):
        func = compile_for_benchmark(func)
    if is_train and grad_to_none:
        training_func = func

        def func():
            # Capture the same fresh-gradient step that eager timing performs.
            # Clearing outside graph.replay would not change the captured accumulation ops.
            for tensor in grad_to_none:
                tensor.grad = None
            return training_func()

    if not is_train:
        inner = func

        def func():
            with torch.no_grad():
                return inner()

    if os.environ.get("BENCH_MARK_STEP") == "1":
        inner_step = func

        def func():
            torch.compiler.cudagraph_mark_step_begin()
            return inner_step()

    # Probe outside the timing window. Compile tracing/reference calls do not count;
    # evidence is supplied only by an executed Inductor executable in this scope.
    captured_outputs = [None]

    def checked_step():
        output = func()
        captured_outputs[0] = ({"result": output, "gradients": tuple(t.grad for t in grad_to_none)}
                               if is_train else output)
        return output

    with observe_execution() as evidence:
        checked_step()
        execution_checks["finite_tensors"] = check_finite_outputs(captured_outputs[0])
        if eager_snapshot is not None:
            execution_checks["compiled_vs_eager"] = check_execution_outputs(captured_outputs[0], eager_snapshot)
            del eager_snapshot
        if conf.cudagraph in {"manual", "graphed"}:
            before_capture = snapshot_outputs(captured_outputs[0])
            graph = capture_cudagraph(checked_step, params, is_train=is_train)
            graph.replay()
            torch.cuda.synchronize()
            execution_checks["graph_replay"] = check_execution_outputs(captured_outputs[0], before_capture)
            del before_capture
            timed = graph.replay
            actual_graph = "manual"
        else:
            timed = func
            actual_graph = "disabled"
    require_compile_evidence(conf.compile, evidence)
    if actual_graph == "disabled":
        captured_outputs[0] = None
        for tensor in grad_to_none:
            tensor.grad = None
    if conf.metric == "memory":
        value = bench_memory(timed)["median_mb"]
    else:
        value = bench_time(timed, grad_to_none=[] if actual_graph != "disabled" else grad_to_none)["median_ms"]
    return BenchResult(
        value=value, input_dtype=input_dtype, parameter_dtype=parameter_dtype,
        execution_path=execution_path, reference=reference, compiled=evidence.compiled,
        cudagraph=actual_graph, compile_scope="+".join(sorted(evidence.scopes)),
        compiled_graphs=len(evidence.graphs), measurement_scope=scope, input_shapes=actual_shapes,
        execution_validation=json.dumps(execution_checks, sort_keys=True),
    )


def actual_compiled_flag(result: BenchResult | None) -> bool | None:
    """Unknown or failed executions cannot inherit a requested compile flag."""
    return None if result is None else result.compiled


_CAPTURE_STREAM: "torch.cuda.Stream | None" = None


def capture_stream() -> "torch.cuda.Stream":
    """The one non-default stream that graph capture and its forward graph must share.

    Ported from cf68bd2c, which fixed this on `origin/mpnn` and was never merged here.

    A CUDA graph captures on a non-default stream, and the autograd engine replays every backward
    op on the stream its FORWARD op ran on. Build the forward on the default stream and capture
    elsewhere and CUDA rejects the capture: `cudaErrorStreamCaptureIsolation`, "operation would
    make the legacy stream depend on a capturing blocking stream". That is why adaln_bwd,
    transition_b2b_bwd and gemm_epilogue_bwd -- every target that times `torch.autograd.grad` --
    produced no captured number at all, while layernorm_bwd (a pure function, nothing routes it
    back to the default stream) was fine.

    And a dead capture leaves the default generator registered to it, so every LATER row in the
    process fails on RNG with "Offset increment outside graph capture" instead. One bad capture
    loses the whole sweep and reports a misleading error for every row after the first: on the
    A100 run each of the three targets failed row 1 with cudaErrorStreamCaptureInvalidated and
    rows 2-12 with the RNG message.
    """
    global _CAPTURE_STREAM
    if _CAPTURE_STREAM is None:
        _CAPTURE_STREAM = torch.cuda.Stream()
    return _CAPTURE_STREAM


def forward_stream(conf) -> "contextlib.AbstractContextManager":
    """Build a bench's forward on the capture stream, unless no graph will be captured.

    cf68bd2c did this inside `_bwd_autograd_result`, which then took the `build` callable. That
    signature is gone -- the three backward benches construct `out` themselves -- so it moves to
    the one place every bench function is invoked. Keeping the forward off the default stream is
    what the fix needs; where the construction sits is not.
    """
    if getattr(conf, "cudagraph", "disabled") == "disabled":
        return contextlib.nullcontext()
    side = capture_stream()
    side.wait_stream(torch.cuda.current_stream())
    return torch.cuda.stream(side)


def capture_cudagraph(step: Callable, params: list, is_train: bool,
                      warmup_iters: int = 8) -> "torch.cuda.CUDAGraph":
    """Capture `step` (the existing training_step = fwd + fabric.backward, or inference_step = fwd)
    in a per-shape CUDA graph and return it; replay reruns the captured kernels with zero
    host/launch overhead — the deployment regime for graph-break cute/triton kernels. Reusing the
    harness's own step keeps the backward path consistent (fabric.backward, required by the
    fabric/precision strategy). Training: params get static .grad buffers (accumulated on replay,
    fine for timing). Module-scoped — `step` excludes the optimizer. Inputs must be the same static
    tensors each replay (the harness reuses one pair/dy/mask)."""
    if is_train:
        for p in params:
            p.grad = torch.zeros_like(p)
    # Autotune-capture builds: prime Triton autotune on the DEFAULT stream FIRST, so every
    # kernel's `_bench` runs eagerly and is recorded by the cache builder. Forward kernels tune
    # fine from the side-stream warmup below, but BACKWARD-only kernels (transition/attention
    # bwd, split bwd, …) otherwise first tune inside the graph capture — where do_bench can't run
    # — and are silently skipped. A couple of eager fwd(+bwd) iters here fixes that; it's a no-op
    # for normal timing runs (guarded on the capture patch being installed).
    try:
        from miniworld_engine.autotune import capture as _cap
        _capturing = _cap._orig_bench is not None
    except Exception:
        _capturing = False
    if _capturing:
        for _ in range(2):
            if is_train:
                step()
            else:
                with torch.no_grad():
                    step()
        torch.cuda.synchronize()
    side = capture_stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup_iters):
            if is_train:
                step()
            else:
                with torch.no_grad():
                    step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if is_train:
        # The SAME stream the forward was built on; `torch.cuda.graph` left to
        # itself picks an internal one and reintroduces the split.
        with torch.cuda.graph(graph, stream=side):
            step()
    else:
        with torch.cuda.graph(graph, stream=side), torch.no_grad():
            step()
    return graph


def as_bench_result(value: float) -> BenchResult:
    return BenchResult(value=value)


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    actual_f = actual.detach().float().reshape(-1)
    expected_f = expected.detach().float().reshape(-1)
    diff = actual_f - expected_f
    max_abs = float(diff.abs().max().item())
    rel_frob = float(diff.norm().div(expected_f.norm().clamp_min(1e-20)).item())
    cosine = float(
        actual_f.dot(expected_f).div(actual_f.norm() * expected_f.norm() + 1e-20).item(),
    )
    return max_abs, rel_frob, cosine


@torch.compiler.disable()
@torch.no_grad()
def _miniworld_inference(
    pair: torch.Tensor,
    w_left: torch.Tensor,
    w_left_gate: torch.Tensor,
    w_right: torch.Tensor,
    w_right_gate: torch.Tensor,
    w_gate: torch.Tensor,
    w_out: torch.Tensor,
    w_out_nn: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    eps: float,
    packed_left_right: torch.Tensor,
    row_mask: torch.Tensor | None,
) -> torch.Tensor:
    from miniworld_engine.kernels.trimul_inproj.cute.inference import (
        trimul_inproj_inference,
    )

    if pair.shape[-1] <= 128:
        return trimul_inproj_inference(
            pair,
            w_left,
            w_left_gate,
            w_right,
            w_right_gate,
            w_gate,
            w_out,
            norm_in_weight,
            norm_in_bias,
            norm_out_weight,
            norm_out_bias,
            eps,
            packed_left_right,
            row_mask,
        )

    from miniworld_engine.kernels.layernorm.triton.main import (
        triton_layernorm,
        triton_layernorm_masked,
    )
    from miniworld_engine.kernels.trimul_inproj.cute.back_split import (
        trimul_back_split,
    )
    from miniworld_engine.kernels.trimul_inproj.cute.launch import (
        trimul_inproj_cute_forward,
    )

    batch, left_len, right_len, d_pair = pair.shape
    flat_pair = pair.reshape(batch * left_len * right_len, d_pair)
    if row_mask is None:
        x_normed = triton_layernorm(flat_pair, norm_in_weight, norm_in_bias, eps)
    else:
        x_normed = triton_layernorm_masked(
            flat_pair,
            norm_in_weight,
            norm_in_bias,
            eps,
            row_mask,
        )
    x_normed = x_normed.view(batch, left_len, right_len, d_pair)
    left, right, _gate = trimul_inproj_cute_forward(
        x_normed,
        w_left,
        w_left_gate,
        w_right,
        w_right_gate,
        None,
        bdll_direct=True,
        compute_gate=False,
        b_lr=packed_left_right,
    )
    triangle = torch.einsum("bdik,bdjk->bdij", left, right)
    return trimul_back_split(
        triangle,
        x_normed,
        w_out_nn,
        w_gate,
        norm_out_weight,
        norm_out_bias,
        pair,          # the residual: every trimul back half returns `pair + trimul(pair)`
        eps,
    )


class MiniWorldTriangleMultiplicationInference(nn.Module):
    def __init__(self, base: TriangleMultiplication) -> None:
        super().__init__()
        from miniworld_engine.kernels.trimul_inproj.cute import _bdll_patch
        from miniworld_engine.kernels.trimul_inproj.cute.launch import (
            prepack_lr_operand,
        )

        _bdll_patch.apply()
        _load_cute_fns()

        base = base.to(device=DEVICE, dtype=torch.bfloat16)
        self.left_weight = base.to_left.weight.T
        self.left_gate_weight = base.to_left_gate.weight.T
        self.right_weight = base.to_right.weight.T
        self.right_gate_weight = base.to_right_gate.weight.T
        self.gate_weight = base.to_gate.weight.T.contiguous()
        self.out_weight = base.to_out.weight.T.contiguous()
        self.out_weight_nn = base.to_out.weight
        self.norm_in_weight = base.ln_pair.weight
        self.norm_in_bias = base.ln_pair.bias
        self.norm_out_weight = base.ln_out.weight
        self.norm_out_bias = base.ln_out.bias
        self.eps = base.ln_pair.eps
        self.packed_left_right = prepack_lr_operand(
            self.left_weight,
            self.left_gate_weight,
            self.right_weight,
            self.right_gate_weight,
        )

    def forward(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        pair = pair.to(torch.bfloat16)
        row_mask = None
        if mask is not None:
            row_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).reshape(-1).to(pair.dtype)
        return _miniworld_inference(
            pair,
            self.left_weight,
            self.left_gate_weight,
            self.right_weight,
            self.right_gate_weight,
            self.gate_weight,
            self.out_weight,
            self.out_weight_nn,
            self.norm_in_weight,
            self.norm_in_bias,
            self.norm_out_weight,
            self.norm_out_bias,
            self.eps,
            self.packed_left_right,
            row_mask,
        )


class MiniWorldTriangleMultiplicationTraining(nn.Module):
    def __init__(self, base: TriangleMultiplication) -> None:
        super().__init__()
        from miniworld_engine.kernels.trimul_inproj.cute.v6_training_merged import (
            V6TriMulMerged as V6TriMul,
        )

        self.impl = V6TriMul(base.to(torch.bfloat16))

    def forward(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.impl(pair.to(torch.bfloat16), mask)


def triangle_multiplication_path(implementation: str, mode: str, d_pair: int) -> str:
    if implementation == MINIWORLD_IMPL and is_inference_mode(mode):
        if d_pair <= 128:
            return "miniworld.trimul_inproj_inference"
        return "miniworld.forward_only_front+split_back"
    if implementation == MINIWORLD_IMPL:
        return "miniworld.v6_training_merged"
    if implementation == DTV1_IMPL:
        return "dtv1.fused_triangle_multiplicative_update"
    return implementation


@contextlib.contextmanager
def _paired_trimul_dropout(model, reference, pair, probability):
    """Give accuracy calls identical representable scales; never used by the timer.

    A common seed cannot align FP32 reference and compiled BF16 RNG streams. Bind
    the same per-layer scale temporarily, then restore the production generators
    before measurement warmup/compilation and all timed calls.
    """
    from unittest.mock import patch

    with contextlib.ExitStack() as stack:
        if probability:
            for actual_layer, reference_layer in zip(model.layers, reference.layers, strict=True):
                shape = (pair.shape[0], 1, pair.shape[2], pair.shape[3])
                keep = torch.rand(shape, device=pair.device, dtype=pair.dtype) > probability
                scale = keep.to(pair.dtype) / (1.0 - probability)

                def generate(x, p, scale=scale):
                    return scale.to(x.dtype)

                for layer in (actual_layer, reference_layer):
                    stack.enter_context(patch.object(layer, "_make_drop_row_scale", generate))
        yield


def bench_module_triangle_multiplication(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
    bidirectional: bool = False,
):
    # single-dir TriangleMultiplication, or the bidirectional (outgoing+incoming) variant — the
    # only differences are the base module, the dt-v1 baseline fn, and the miniworld layer class;
    # the whole correctness + timing (incl. CUDA-graph) tail below is shared.
    if implementation == DTV1_IMPL:
        # DTv1's input LN calls torch.ops directly, before its output LN imports
        # the vendor wrapper that registers these operators in a fresh process.
        importlib.import_module("cuequivariance_ops_torch.fused_layer_norm_torch")
    base_cls = BidirectionalTriangleMultiplication if bidirectional else TriangleMultiplication
    if bidirectional and conf.trimul_direction != "outgoing":
        raise UnsupportedBenchmark("bidirectional already includes both directions")

    def make_layer(index, **kwargs):
        if not bidirectional:
            kwargs["outgoing"] = (index % 2 == 0 if conf.trimul_direction == "alternating"
                                  else conf.trimul_direction == "outgoing")
        return base_cls(conf.d_pair, **kwargs)
    torch.manual_seed(0)
    layer_states = []
    for layer_index in range(conf.n_layers):
        base = make_layer(layer_index)
        for linear in (
            base.to_left,
            base.to_left_gate,
            base.to_right,
            base.to_right_gate,
            base.to_gate,
            base.to_out,
        ):
            nn.init.normal_(linear.weight, std=conf.d_pair**-0.5)
        layer_states.append(base.state_dict())

    class MultiTriangleMultiplication(nn.Module):
        def __init__(self, raw_implementation: str) -> None:
            super().__init__()
            self.raw_implementation = raw_implementation
            if raw_implementation == DTV1_IMPL:
                self.layers = nn.ModuleList(
                    [make_layer(layer_index, p_drop=conf.dropout) for layer_index in range(len(layer_states))],
                )
                for layer, state in zip(self.layers, layer_states, strict=True):
                    layer.load_state_dict(state)
                return
            if raw_implementation == MINIWORLD_IMPL:
                # Use the real production module (implementation=MINIWORLD): its per-GPU
                # dispatch runs the sm100-native cute path on B200 (tcgen05 front + split
                # sm100 out-projection), correct at every d. The prior hand-wired wrappers
                # (MiniWorld*Inference/Training, BidirV6TriMul) called the H100 quack /
                # back_split kernels, which are numerically WRONG on sm_100 (out-cosine
                # ~0.05-0.7 vs pytorch) and assert "SM90 only" at d>=256.
                # cute path is bf16-only (asserts on fp32 weights); pin bf16 like the
                # old wrappers did. load_state_dict casts the fp32 reference state.
                self.layers = nn.ModuleList(
                    [
                        make_layer(
                            layer_index,
                            implementation=ImplementationType.MINIWORLD,
                            p_drop=conf.dropout,
                        ).to(
                            torch.bfloat16,
                        )
                        for layer_index in range(len(layer_states))
                    ],
                )
                for layer, state in zip(self.layers, layer_states, strict=True):
                    layer.load_state_dict(state)
                return

            spec = parse_implementation_spec(raw_implementation)
            if bidirectional:
                self.layers = nn.ModuleList(
                    [make_layer(layer_index, implementation=spec.impl, p_drop=conf.dropout)
                     for layer_index in range(len(layer_states))],
                )
            else:
                self.layers = nn.ModuleList(
                    [
                        make_layer(
                            layer_index,
                            implementation=spec.impl,
                            ln_implementation=spec.ln_impl or ImplementationType.PYTORCH,
                            p_drop=conf.dropout,
                        )
                        for layer_index in range(len(layer_states))
                    ],
                )
            for layer, state in zip(self.layers, layer_states, strict=True):
                layer.load_state_dict(state)

        def forward(
            self,
            pair: torch.Tensor,
            mask: torch.Tensor | None,
        ) -> torch.Tensor:
            for layer in self.layers:
                if self.raw_implementation != DTV1_IMPL:
                    pair = layer(pair, mask)
                    continue
                mask_2d = None
                if mask is not None:
                    mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                # `self.layers` is an nn.ModuleList, so iterating it yields `Module` and every
                # `layer.to_left` reads as `Tensor | Module` -- twenty findings for one loop
                # variable. Both classes that can be in here (TriangleMultiplication and its
                # bidirectional sibling) carry the same projection and norm names.
                tm = cast("TriangleMultiplication", layer)
                p_in = torch.cat([tm.to_left.weight, tm.to_right.weight], dim=0)
                g_in = torch.cat([tm.to_left_gate.weight, tm.to_right_gate.weight], dim=0)
                # The miniworld TriangleMultiplication now ALWAYS adds the residual (it is
                # unconditional — see the module). The dtv1 baseline is the raw op, so add the
                # residual explicitly here to keep the per-layer stack semantics identical for a
                # fair speed/correctness comparison against the residual-inclusive pytorch ref.
                if bidirectional:
                    update = fused_bidirectional_dtv1(
                        pair, mask_2d,
                        norm_in_weight=tm.ln_pair.weight, norm_in_bias=tm.ln_pair.bias,
                        p_in_weight=p_in, g_in_weight=g_in,
                        norm_out_weight=tm.ln_out.weight, norm_out_bias=tm.ln_out.bias,
                        p_out_weight=tm.to_out.weight, g_out_weight=tm.to_gate.weight,
                        h=tm.d_hidden, eps=tm.ln_pair.eps,
                    )
                else:
                    update = fused_triangle_multiplicative_update_dtv1(
                        pair, direction="outgoing" if tm.outgoing else "incoming", mask=mask_2d,
                        norm_in_weight=tm.ln_pair.weight, norm_in_bias=tm.ln_pair.bias,
                        p_in_weight=p_in, g_in_weight=g_in,
                        norm_out_weight=tm.ln_out.weight, norm_out_bias=tm.ln_out.bias,
                        p_out_weight=tm.to_out.weight, g_out_weight=tm.to_gate.weight,
                        eps=tm.ln_pair.eps,
                    )
                if self.training and tm.p_drop > 0:
                    update = update * tm._make_drop_row_scale(pair, tm.p_drop)
                pair = pair + update
            return pair

    # Full bf16 for EVERY impl. The old code fed only miniworld a bf16 `pair` and left the others at
    # fp32, so pytorch/triton/cuequiv were timed as fp32 kernels -- an unfair, much slower floor
    # (triton 15.8ms fp32 vs 2.8ms bf16). The fix is the bf16 input for all of them: the custom ops
    # compute in their input dtype, and the torch path follows its inputs. `fabric` here is the
    # no-op `_NoFabric` shim -- there is NO autocast and no fp32 master anywhere in this harness.
    # The module cast below is likewise only PARTLY a cast: `_Fp32ParamsMixin._apply`
    # (modules/primitives.py) deliberately pins norm affine params to fp32 through any
    # `.to(bfloat16)` (gamma at 1.0 stagnates in bf16 -- its ULP exceeds an Adam step), which is why
    # `parameter_dtype` still records float32. Trunk weights become bf16; norm gammas stay fp32.
    bf16 = conf.precision != FP32_PRECISION
    model = MultiTriangleMultiplication(implementation).to(DEVICE)
    model = model.to(torch.bfloat16 if bf16 else torch.float32)
    model.train(not is_inference_mode(conf.mode))
    # Real deployment = compiled kernels, then graph-captured. The old guard compiled the model ONLY
    # when cudagraph=="disabled" and captured the EAGER model under a graph -- a leftover from
    # compile_wrap="disable", where a graph break crashed manual capture mid-stream. The default wrap
    # is custom_op now (no breaks), so compile+capture is the regime that actually ships; compile
    # whenever asked and let the capture below wrap the compiled module.
    if conf.compile:
        compile_module_for_benchmark(model)
    if implementation != MINIWORLD_IMPL:
        model = fabric.setup_module(model)

    ref_model = MultiTriangleMultiplication(ImplementationType.PYTORCH.value).to(DEVICE)
    ref_model.train(not is_inference_mode(conf.mode))
    if conf.compile:
        compile_module_for_benchmark(ref_model)
    ref_model = fabric.setup_module(ref_model)

    pair_dtype = torch.bfloat16 if bf16 else torch.float32
    torch.manual_seed(1)
    pair = torch.randn(
        1,
        seq_len,
        seq_len,
        conf.d_pair,
        device=DEVICE,
        dtype=pair_dtype,
    )
    dy = torch.randn_like(pair)
    pair.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def inference_step() -> torch.Tensor:
        return model(pair, mask)

    def training_step() -> torch.Tensor:
        y = inference_step()
        fabric.backward(y, dy)
        return y

    def correctness() -> AccuracyFields:
        pair_impl = pair.detach().clone().requires_grad_(not is_inference_mode(conf.mode))
        pair_ref = pair.detach().float().clone().requires_grad_(not is_inference_mode(conf.mode))
        dy_ref = dy.detach().float()
        if is_inference_mode(conf.mode):
            with torch.no_grad():
                actual = model(pair_impl, mask)
                expected = ref_model(pair_ref, mask)
            out_max, out_rel, out_cos = tensor_metrics(actual, expected)
            return {
                "output_max_abs": out_max,
                "output_rel_frob": out_rel,
                "output_cosine": out_cos,
            }

        actual = model(pair_impl, mask)
        expected = ref_model(pair_ref, mask)
        fabric.backward(actual, dy)
        fabric.backward(expected, dy_ref)
        out_max, out_rel, out_cos = tensor_metrics(actual, expected)
        assert pair_impl.grad is not None
        assert pair_ref.grad is not None
        grad_max, grad_rel, grad_cos = tensor_metrics(pair_impl.grad, pair_ref.grad)
        return {
            "output_max_abs": out_max,
            "output_rel_frob": out_rel,
            "output_cosine": out_cos,
            "grad_max_abs": grad_max,
            "grad_rel_frob": grad_rel,
            "grad_cosine": grad_cos,
        }

    with _paired_trimul_dropout(model, ref_model, pair, conf.dropout):
        accuracy = correctness()
    for item in [pair, *list(model.parameters()), *list(ref_model.parameters())]:
        item.grad = None
    # Inference under no_grad (this bench does its own capture, so it can't lean on
    # measured_result's central guard): a grad-enabled forward routes trimul's grad-keyed dispatch
    # to the save-activation TRAINING kernels. Wrap only the timed func -- training_step must keep
    # grad, and it reuses inference_step, so no_grad cannot live inside inference_step itself.
    if is_inference_mode(conf.mode):
        def func() -> torch.Tensor:
            with torch.no_grad():
                return inference_step()
    else:
        func = training_step
    grad_to_none = [pair, *list(model.parameters())]
    return measured_result(
        conf=conf, func=func, grad_to_none=grad_to_none,
        params=list(model.parameters()), is_train=not is_inference_mode(conf.mode),
        input_dtype=str(pair.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=("dtv1.fused_bidirectional_triangle_multiplicative_update"
                        if bidirectional and implementation == DTV1_IMPL else
                        "dtv1.fused_triangle_multiplicative_update"
                        if implementation == DTV1_IMPL else
                        "modules.triangle_multiplication.bidirectional.BidirectionalTriangleMultiplication"
                        if bidirectional else "modules.triangle_multiplication.module.TriangleMultiplication"),
        reference=ImplementationType.PYTORCH.value,
    )._replace(**accuracy)


def bench_module_triangle_attention(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
):
    spec = triton_miniworld_spec(implementation)

    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiTriangleAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    TriangleAttention(
                        conf.d_pair,
                        implementation=spec.impl,
                        use_self_attention=True,
                        p_drop=conf.dropout,
                    )
                    for _ in range(conf.n_layers)
                ],
            )

        def forward(
            self,
            pair: torch.Tensor,
            mask: torch.Tensor | None,
        ) -> torch.Tensor:
            for layer in self.layers:
                pair = layer(pair, mask)
            return pair

    # A zero output projection hides dropout's effect and gives zero upstream
    # gradients. Use reproducible nonzero weights, as the trimul comparison does.
    torch.manual_seed(0)
    model = MultiTriangleAttention()
    for layer in model.layers:
        for linear in layer.modules():
            if isinstance(linear, nn.Linear):
                nn.init.normal_(linear.weight, std=conf.d_pair**-0.5)
    model = model.to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:  # compile the kernels, then capture (real regime); custom_op has no breaks
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(pair)
    pair.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def inference_step() -> torch.Tensor:
        return model(pair, mask)

    def training_step() -> torch.Tensor:
        y = inference_step()
        fabric.backward(y, dy)
        return y

    func = inference_step if is_inference_mode(conf.mode) else training_step
    execution_path = {
        ImplementationType.PYTORCH: "module.reference.torch",
        ImplementationType.TRITON: (
            "modules.triangle_attention.full."
            "layernorm+qkv_bias_projection+triton_triangle_attention_pair_bias+gate_out"
        ),
        ImplementationType.CUEQUIVARIANCE: "cuequivariance_torch.triangle_attention",
    }.get(spec.impl, spec.impl.value)
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=[pair, *list(model.parameters())],
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(pair.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=execution_path,
        reference="module.reference.torch",
    )


def bench_module_transition(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
):
    spec = module_miniworld_spec(implementation)
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiTransition(nn.Module):
        def __init__(self, layer_spec: ImplementationSpec) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    Transition(conf.d_pair, implementation=layer_spec.impl)
                    for _ in range(conf.n_layers)
                ],
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(0)
    layer_states = [
        Transition(conf.d_pair, implementation=ImplementationType.PYTORCH).state_dict()
        for _ in range(conf.n_layers)
    ]

    model = MultiTransition(spec).to(DEVICE)
    for layer, state in zip(model.layers, layer_states, strict=True):
        layer.load_state_dict(state)
    model.to(dtype=dtype)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:  # compile the kernels, then capture (real regime); custom_op has no breaks
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    ref_spec = ImplementationSpec(ImplementationType.PYTORCH, None, "pytorch")
    ref_model = MultiTransition(ref_spec).to(DEVICE)
    for layer, state in zip(ref_model.layers, layer_states, strict=True):
        layer.load_state_dict(state)
    ref_model.to(dtype=dtype)
    ref_model.train(not is_inference_mode(conf.mode))
    if conf.compile:  # ref compiled too, regardless of graph, so the comparison is apples-to-apples
        compile_module_for_benchmark(ref_model)
    ref_model = fabric.setup_module(ref_model)

    torch.manual_seed(1)
    x = torch.randn(1, seq_len, seq_len, conf.d_pair, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(x)
    x.requires_grad = True

    def inference_step() -> torch.Tensor:
        return model(x)

    def training_step() -> torch.Tensor:
        y = inference_step()
        fabric.backward(y, dy)
        return y

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [x, *list(model.parameters())]

    def correctness() -> AccuracyFields:
        x_impl = x.detach().clone().requires_grad_(not is_inference_mode(conf.mode))
        x_ref = x.detach().clone().requires_grad_(not is_inference_mode(conf.mode))
        dy_ref = dy.detach().clone()
        if is_inference_mode(conf.mode):
            with torch.no_grad():
                actual = model(x_impl)
                expected = ref_model(x_ref)
            out_max, out_rel, out_cos = tensor_metrics(actual, expected)
            return {
                "output_max_abs": out_max,
                "output_rel_frob": out_rel,
                "output_cosine": out_cos,
            }

        actual = model(x_impl)
        expected = ref_model(x_ref)
        fabric.backward(actual, dy)
        fabric.backward(expected, dy_ref)
        out_max, out_rel, out_cos = tensor_metrics(actual, expected)
        assert x_impl.grad is not None
        assert x_ref.grad is not None
        grad_max, grad_rel, grad_cos = tensor_metrics(x_impl.grad, x_ref.grad)
        return {
            "output_max_abs": out_max,
            "output_rel_frob": out_rel,
            "output_cosine": out_cos,
            "grad_max_abs": grad_max,
            "grad_rel_frob": grad_rel,
            "grad_cosine": grad_cos,
        }

    accuracy = correctness()
    for item in [x, *list(model.parameters()), *list(ref_model.parameters())]:
        item.grad = None

    execution_path = (
        "module.reference.torch"
        if spec.impl in {ImplementationType.PYTORCH, ImplementationType.CUEQUIVARIANCE}
        else "modules.transition.module.Transition"
    )
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(x.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=execution_path,
        reference="module.reference.torch",
    )._replace(**accuracy)


def bench_module_conditioned_transition(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
):
    spec = module_miniworld_spec(implementation)
    if spec.impl not in {
        ImplementationType.PYTORCH,
        ImplementationType.TRITON,
        ImplementationType.CUEQUIVARIANCE,
        ImplementationType.MINIWORLD,
    }:
        raise UnsupportedBenchmark(f"conditioned_transition does not implement {implementation!r}")

    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiConditionedTransition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    # The token side: `d_single_token` (768) conditioned on `d_single` (384), which
                    # is what the model's `token_dit` builds and what AlphaFold-3 calls c_token and
                    # c_s. This read `d_hidden=d_pair(128), d_cond=d_single_token(768)` -- the two
                    # roles swapped and the wrong widths, a combination the model never builds.
                    ConditionedTransition(
                        d_hidden=conf.d_single_token,
                        d_cond=conf.d_single,
                        implementation=spec.impl,
                        dtype=dtype,
                    )
                    for _ in range(conf.n_layers)
                ],
            )

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x, cond)
            return x

    model = MultiConditionedTransition().to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    # requires_grad ONLY when the mode is training. AdaptiveLayerNorm and ConditionedTransition
    # both pick their kernel by asking whether anything in the graph carries a gradient -- an input
    # that always requires one routes the INFERENCE bench through the training forward, which saves
    # activations for a backward that never runs. The smoke run showed it: `mode=inference` launched
    # adaln_epilogue_saveact_triton and layernorm_fwd_saveact_strided_triton, both save-activation
    # kernels. Training keeps it: these are mid-network blocks, so dx is real work the model does.
    # `conf.precision`, like every other module bench. This one ignored it twice over: it never
    # passed `dtype` to the constructor, whose default is fp32, and it built its inputs with no
    # dtype at all -- so `precision=bf16-mixed` still measured fp32 end to end. The model runs
    # bf16, and this family's kernels run either.
    wants_grad = not is_inference_mode(conf.mode)
    x = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_single_token,
        device=DEVICE,
        dtype=dtype,
        requires_grad=wants_grad,
    )
    cond = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_single,
        device=DEVICE,
        dtype=dtype,
        requires_grad=wants_grad,
    )
    dy = torch.randn_like(x)

    def inference_step() -> torch.Tensor:
        # Under no_grad, like every other module bench here (see the triangle_multiplication one).
        # Without it the module still saves activations for a backward this mode never runs:
        # `dispatch.needs_backward` asks whether gradients are RECORDABLE, and outside no_grad a
        # module that nobody called .eval() on, holding parameters that require grad, says yes.
        # The smoke run showed the cost -- `mode=inference` launching *_saveact_* kernels.
        with torch.no_grad():
            return model(x, cond)

    def training_step() -> torch.Tensor:
        y = model(x, cond)
        fabric.backward(y, dy)
        return y

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [x, cond, *list(model.parameters())]
    if spec.impl in {ImplementationType.TRITON,
                     ImplementationType.MINIWORLD}:
        if is_inference_mode(conf.mode):
            # The dispatch reads d_hidden, and d_hidden is `d_single_token`. This branched on
            # `conf.d_pair` -- the pair width, which this module never sees -- so at the token
            # width it recorded the fused b2b path while the composed one ran. The threshold is
            # imported rather than re-typed: a bench that keeps its own copy of a dispatch constant
            # is a second place for it to drift.
            from miniworld_engine.kernels.conditioned_transition.triton.dispatch import (
                ATOM_D_MAX,
            )

            execution_path = (
                "kernels.conditioned_transition.triton.inference"
                if conf.d_single_token <= ATOM_D_MAX
                else "kernels.conditioned_transition.triton.composed"
            )
        else:
            execution_path = "kernels.conditioned_transition.triton.training"
    else:
        execution_path = "module.reference.torch"
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(x.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=execution_path,
        reference="module.reference.torch",
    )


def bench_module_adaptive_layernorm(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
):
    spec = triton_miniworld_spec(implementation)
    implementation_type = spec.impl
    if implementation_type not in {
        ImplementationType.PYTORCH,
        ImplementationType.TRITON,
    }:
        raise UnsupportedBenchmark(f"adaptive_layernorm does not implement {implementation!r}")

    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiAdaptiveLayerNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    # The token side, like the ConditionedTransition bench: AdaptiveLayerNorm is
                    # constructed inside ConditionedTransition and AugmentedAttentionPairBias with
                    # the block's own (d_single, d_cond), so the model builds it at 768/384 and at
                    # 128/128 and at nothing else. `d_pair` (128) for BOTH was the pair width, which
                    # this module never sees -- adaln normalises the SINGLE representation.
                    AdaptiveLayerNorm(
                        d_hidden=conf.d_single_token,
                        d_cond=conf.d_single,
                        implementation=implementation_type,
                    )
                    for _ in range(conf.n_layers)
                ],
            )

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x, cond)
            return x

    model = MultiAdaptiveLayerNorm().to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    # requires_grad ONLY when the mode is training. AdaptiveLayerNorm and ConditionedTransition
    # both pick their kernel by asking whether anything in the graph carries a gradient -- an input
    # that always requires one routes the INFERENCE bench through the training forward, which saves
    # activations for a backward that never runs. The smoke run showed it: `mode=inference` launched
    # adaln_epilogue_saveact_triton and layernorm_fwd_saveact_strided_triton, both save-activation
    # kernels. Training keeps it: these are mid-network blocks, so dx is real work the model does.
    wants_grad = not is_inference_mode(conf.mode)
    x = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_single_token,
        device=DEVICE,
        dtype=dtype,
        requires_grad=wants_grad,
    )
    cond = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_single,
        device=DEVICE,
        dtype=dtype,
        requires_grad=wants_grad,
    )
    dy = torch.randn_like(x)

    def inference_step() -> torch.Tensor:
        # Under no_grad, like every other module bench here (see the triangle_multiplication one).
        # Without it the module still saves activations for a backward this mode never runs:
        # `dispatch.needs_backward` asks whether gradients are RECORDABLE, and outside no_grad a
        # module that nobody called .eval() on, holding parameters that require grad, says yes.
        # The smoke run showed the cost -- `mode=inference` launching *_saveact_* kernels.
        with torch.no_grad():
            return model(x, cond)

    def training_step() -> torch.Tensor:
        y = model(x, cond)
        fabric.backward(y, dy)
        return y

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [x, cond, *list(model.parameters())]
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(x.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        # What the module ACTUALLY dispatches to. `kernels.adaln.triton.main` named a file
        # that no longer exists (it was split into inference.py / ln_strided.py), and it named
        # one path where the module picks between two: AdaptiveLayerNorm.forward calls
        # `adaln_train` when anything in the graph carries a gradient and `adaln_inference`
        # otherwise, and those are two different kernel files with different kernels.
        execution_path=(
            "module.reference.torch"
            if implementation_type == ImplementationType.PYTORCH
            else ("kernels.adaln.triton.inference" if is_inference_mode(conf.mode)
                  else "kernels.adaln.triton.training")
        ),
        reference="module.reference.torch",
    )


def bench_module_augmented_attention_token(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
):
    spec = triton_miniworld_spec(implementation)
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class BiasOnlyValue(AugmentedAttentionPairBias):
        """`out = softmax(bias) @ v` -- see BIAS_ONLY_V_IMPL.

        The whole forward is overridden rather than just the core: to_query and to_key are the
        point of the ablation, and a core-only override would still pay for them.
        """

        def forward(self, single, cond, pair, mask=None, *, compute_dtype=None):
            # Imported here, not at module scope: this file carries `torch` and `nn` only, and
            # an implementation reaches for what it needs at its own branch.
            from miniworld_engine.modules.functional import sigmoid_gate

            single = self.ada_ln_in(single, cond)
            pair = self.ln_pair(pair)
            value, gate = self.to_value(single), self.to_gate(single)
            bias = self.to_bias(pair)                                   # (B, L, L, H)
            n_aug, batch, len_res = value.shape[:3]
            hidden = value.shape[-1] // self.n_head
            value = value.view(n_aug, batch, len_res, self.n_head, hidden)
            # No `a` on the bias side of the einsum: the softmax is computed once and reused
            # across the augmentation samples, where the full path computes one per sample.
            logits = bias.permute(0, 3, 1, 2)
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
            attention = torch.softmax(logits, dim=-1)  # (B, H, L, L)
            out = torch.einsum("bhij,abjhd->abihd", attention, value)
            out = out.flatten(-2)                                        # (A, B, L, H*D)
            out = sigmoid_gate(gate, out)
            out = self.to_out(out)
            return sigmoid_gate(self.to_scale(cond), out)

    cls = (BiasOnlyValue if implementation.strip().lower() == BIAS_ONLY_V_IMPL
           else AugmentedAttentionPairBias)

    class MultiTokenAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                cls(d_single=conf.d_single_token, d_cond=conf.d_single,
                    d_pair=conf.d_pair, n_head=16, implementation=spec.impl)
                for _ in range(conf.n_layers)
            ])

        def forward(self, single, cond, pair, mask):
            for layer in self.layers:
                single = layer(single, cond, pair, mask)
            return single

    model = MultiTokenAttention().to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))

    if conf.compile:
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair, device=DEVICE, dtype=dtype)
    single = torch.randn(
        conf.n_augment, 1, seq_len, conf.d_single_token, device=DEVICE, dtype=dtype
    )
    cond = torch.randn(
        conf.n_augment, 1, seq_len, conf.d_single, device=DEVICE, dtype=dtype
    )
    dy_single = torch.randn_like(single)
    pair.requires_grad = True
    single.requires_grad = True
    cond.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def inference_step() -> torch.Tensor:
        return model(single, cond, pair, mask)

    def training_step() -> torch.Tensor:
        out_single = inference_step()
        fabric.backward(out_single, dy_single)
        return out_single

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [pair, single, cond, *list(model.parameters())]
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(single.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=(
            "module.reference.torch"
            if spec.impl in {ImplementationType.PYTORCH, ImplementationType.CUEQUIVARIANCE}
            else "modules.augmented_attention.module.AugmentedAttentionPairBias"
        ),
        reference="module.reference.torch",
    )


def bench_module_swa_atom_attention(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
):
    """ESMFold2 SWA atom attention core: Wqkv -> qk-RMSNorm -> 3D RoPE -> windowed attention
    (FA4/FA2, SDPA fallback) -> sigmoid gate -> out_proj. The adaLN modulate and SwiGLU FFN that
    wrap it into a full block live in the consumer's SWAAtomBlock, not here.

    Runs at the atom length (``seq_len * 8``), the granularity the atom stack operates on. The
    windowed attention needs a flash backend to run at that length -- SDPA's [N, S, S] band mask
    is 24 GiB at S=8192 -- so on a card/install without one this reports NaN rather than OOM.
    """
    if implementation.strip().lower() not in {MINIWORLD_IMPL, "pytorch"}:
        raise UnsupportedBenchmark("swa_atom_attention implements pytorch and miniworld")
    swa_impl = ImplementationType(implementation.strip().lower())
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16
    n_head = 4

    class MultiSWA(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                SWA3DRoPEAttention(conf.d_single_atom, n_head, half_window=64, implementation=swa_impl)
                for _ in range(conf.n_layers)
            ])

        def forward(self, x, ap):
            for layer in self.layers:
                x = layer(x, ap)
            return x

    model = MultiSWA().to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    atom_len = seq_len * 8
    n = conf.n_augment
    half = conf.d_single_atom // n_head // 2
    x = torch.randn(n, atom_len, conf.d_single_atom, device=DEVICE, dtype=dtype)
    # cos/sin are per batch element (no augmentation axis) -- B=1 here -- and fp32 for angle
    # precision; build_attention_params expands them over the num_aug=n axis. Values are random:
    # the shapes drive the kernels, and the perf does not depend on the angles.
    cos = torch.randn(1, atom_len, half, device=DEVICE, dtype=torch.float32)
    sin = torch.randn(1, atom_len, half, device=DEVICE, dtype=torch.float32)
    valid_lengths = (torch.rand(n, atom_len, device=DEVICE) > conf.mask_prob).sum(-1)
    valid = torch.arange(atom_len, device=DEVICE)[None, :] < valid_lengths[:, None]
    ap = build_attention_params(cos, sin, valid, num_aug=n)
    dy = torch.randn_like(x)
    x.requires_grad = True

    def inference_step() -> torch.Tensor:
        return model(x, ap)

    def training_step() -> torch.Tensor:
        y = inference_step()
        fabric.backward(y, dy)
        return y

    func = inference_step if is_inference_mode(conf.mode) else training_step
    from miniworld_engine.modules.swa_atom_attention.module import _flash_backend

    backend = _flash_backend(DEVICE)
    execution_path = f"modules.swa_atom_attention.SWA3DRoPEAttention[{backend or 'unavailable'}]"
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=[x, *list(model.parameters())],
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(x.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=execution_path,
        reference="",
    )


def bench_module_augmented_attention_atom(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: FabricLike,
):
    spec = triton_miniworld_spec(implementation)
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiAtomAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                AugmentedAttentionPairBias(
                    d_single=conf.d_single_atom, d_cond=conf.d_single_atom,
                    d_pair=conf.d_pair_atom, n_head=4, implementation=spec.impl,
                ) for _ in range(conf.n_layers)
            ])

        def forward(self, single, cond, pair, mask):
            for layer in self.layers:
                single = layer(single, cond, pair, mask)
            return single

    model = MultiAtomAttention().to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))

    if conf.compile:
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    atom_len = seq_len * 8
    pair = torch.randn(1, atom_len, atom_len, conf.d_pair_atom, device=DEVICE, dtype=dtype)
    single = torch.randn(
        conf.n_augment, 1, atom_len, conf.d_single_atom, device=DEVICE, dtype=dtype
    )
    cond = torch.randn(
        conf.n_augment, 1, atom_len, conf.d_single_atom, device=DEVICE, dtype=dtype
    )
    dy_single = torch.randn_like(single)
    pair.requires_grad = True
    single.requires_grad = True
    cond.requires_grad = True
    mask = torch.rand(1, atom_len, device=DEVICE) > conf.mask_prob

    def inference_step() -> torch.Tensor:
        return model(single, cond, pair, mask)

    def training_step() -> torch.Tensor:
        out_single = inference_step()
        fabric.backward(out_single, dy_single)
        return out_single

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [pair, single, cond, *list(model.parameters())]
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(single.dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=(
            "module.reference.torch"
            if spec.impl in {ImplementationType.PYTORCH, ImplementationType.CUEQUIVARIANCE}
            else "modules.augmented_attention.module.AugmentedAttentionPairBias"
        ),
        reference="module.reference.torch",
    )


def bench_module_triangle_multiplication_bidirectional(conf, seq_len, implementation, fabric):
    return bench_module_triangle_multiplication(conf, seq_len, implementation, fabric, bidirectional=True)


# =============================================================================================
# KERNEL FUNCTION-OPERATION benches. One target == one compute-operation; the folder name is the
# operation's intrinsic nature, NOT a module. Each function benches ALL implementations of that op
# as ROWS (dispatch on `implementation`), incl. deprecated/abandoned variants + a pytorch ref,
# swept over L (seq_len) and d (d_pair). Forward and backward are SEPARATE operations/folders.
# Timing is value-independent; correctness re-inits gating/zero weights so cosine is meaningful.
# Unknown/unsupported `implementation` -> nan row; a raised exception -> status=failed (caught by
# the harness main loop). Forward + standalone (pure-fn) backward ops time a pure launcher
# (is_train=False, cudagraph captures under no_grad); autograd-only backward ops time
# torch.autograd.grad on a pre-built graph (is_train=True, retain_graph).
# =============================================================================================
BF16 = torch.bfloat16


def _acc_fwd(actual: torch.Tensor, expected: torch.Tensor) -> AccuracyFields:
    mx, rel, cos = tensor_metrics(actual, expected)
    return {"output_max_abs": mx, "output_rel_frob": rel, "output_cosine": cos}


def _acc_grad(actual: torch.Tensor, expected: torch.Tensor) -> AccuracyFields:
    mx, rel, cos = tensor_metrics(actual, expected)
    return {"grad_max_abs": mx, "grad_rel_frob": rel, "grad_cosine": cos}


def _flat(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).float() for t in items])


def _fwd_result(conf, kfn, args, *, acc, path, ref, dtype, parameter_dtype=None):
    """Time a pure forward launcher ``kfn(*args)`` (is_train=False) + attach correctness ``acc``."""
    import json

    shapes = json.dumps({f"arg{i}": list(arg.shape) for i, arg in enumerate(args)
                         if isinstance(arg, torch.Tensor)})
    return measured_result(
        conf=conf, func=lambda: kfn(*args), grad_to_none=[], params=[], is_train=False,
        input_dtype=dtype, parameter_dtype=dtype if parameter_dtype is None else parameter_dtype, execution_path=path, reference=ref,
    )._replace(**acc, input_shapes=shapes)


def _bwd_autograd_result(conf, out, leaves, dy, ref_grads, *, path, ref, dtype):
    """Backward-only timing via ``torch.autograd.grad`` on a pre-built forward graph ``out``.
    Compare every requested gradient. is_train=True so capture keeps grad on."""
    import json

    if conf.compile:
        raise UnsupportedBenchmark("compiled backward over a prebuilt eager graph is unsupported")
    def kfn():
        return torch.autograd.grad(out, leaves, dy, retain_graph=True)
    grads = kfn()
    acc = _acc_grad(_flat(list(grads)), _flat(list(ref_grads)))
    shapes = json.dumps({"grad_output": list(dy.shape),
                         "leaves": [list(leaf.shape) for leaf in leaves]})
    return measured_result(
        conf=conf, func=kfn, grad_to_none=[], params=[], is_train=True,
        input_dtype=dtype, parameter_dtype=dtype, execution_path=path, reference=ref,
    )._replace(**acc, input_shapes=shapes)


# ---- FORWARD operations -----------------------------------------------------------------------
def bench_kernel_dual_gemm_epilogue(conf, seq_len, implementation, fabric):
    """Gated dual-GEMM in-projection (trimul front): left=(x@WL)*sigma(x@WLg), right=..., gate=sigma(x@Wg).
    Rows: pytorch, trimul_inproj_cute, tm1_cute, triton_tm1. Variants without a gate compare
    left|right only.

    `trimul_front_triton` and `trimul_front_sm100` are gone: 38575f1a deleted the five fronts
    nothing reaches, and their modules with them, so both rows raised ModuleNotFoundError on every
    shape of every sweep -- 18 rows a run, reported as ordinary bench failures."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)

    def _w():
        return (torch.randn(D, D, device=DEVICE, dtype=dtype) * (D**-0.5)).contiguous()

    wl, wlg, wr, wrg, wg = _w(), _w(), _w(), _w(), _w()

    def _x():
        torch.manual_seed(1)
        return torch.randn(1, L, L, D, device=DEVICE, dtype=dtype).contiguous()

    def ref_lr(x):
        xf = x.reshape(L * L, D)
        return (xf @ wl) * torch.sigmoid(xf @ wlg), (xf @ wr) * torch.sigmoid(xf @ wrg)

    def ref_gate(x):
        return torch.sigmoid(x.reshape(L * L, D) @ wg)

    def bdll_to_md(t):
        return t.permute(0, 2, 3, 1).reshape(L * L, -1)

    if implementation == "pytorch":
        def run(x):
            left, right = ref_lr(x)
            return left, right, ref_gate(x)
        path = "pytorch"
    elif implementation == "trimul_inproj_cute":
        from miniworld_engine.kernels.trimul_inproj.cute.launch import (
            trimul_inproj_cute_forward,
        )

        def run(x):
            left, right, gate = trimul_inproj_cute_forward(x, wl, wlg, wr, wrg, wg, compute_gate=True)
            assert gate is not None      # compute_gate=True; None is the compute_gate=False shape
            return bdll_to_md(left), bdll_to_md(right), gate.reshape(L * L, D)
        path = "kernels.trimul_inproj.cute.launch"
    elif implementation == "tm1_cute":
        from miniworld_engine.kernels.tm1.cute.launch import tm1_cute_forward

        def run(x):
            left, right = tm1_cute_forward(x, wl, wlg, wr, wrg, out_layout="bdll")
            return bdll_to_md(left), bdll_to_md(right)
        path = "kernels.tm1.cute.launch"
    elif implementation == "triton_tm1":
        from miniworld_engine.kernels.tm1.triton.main import triton_tm1

        def run(x):
            # The pair activation, NOT its (M, D) flattening: TritonTM1Function reads
            # `token_key(length_of(x.shape))` before its own rearrange, so a pre-flattened
            # (M, D) is refused outright by the shape_key guard. drivers/trimul_inproj says the same.
            left, right = triton_tm1(x, wl, wlg, wr, wrg)
            return left.reshape(L * L, D), right.reshape(L * L, D)
        path = "kernels.tm1.triton.main"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    xc = _x()
    res = run(xc)
    outs_a, outs_e = [res[0], res[1]], list(ref_lr(xc))
    if len(res) == 3:
        outs_a.append(res[2])
        outs_e.append(ref_gate(xc))
    acc = _acc_fwd(_flat(outs_a), _flat(outs_e))
    return _fwd_result(conf, run, (_x(),), acc=acc, path=path, ref="pytorch", dtype=tname)


def bench_kernel_gemm_epilogue(conf, seq_len, implementation, fabric):
    """Fused LayerNorm+Linear (GEMM w/ LN epilogue): Y = LN(x) @ W^T. N=K=d. Rows: pytorch,
    layernorm_linear_triton, layernorm_linear_cute(M1), layernorm_linear_cute_fused(M2), layernorm_linear_te."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    import torch.nn.functional as F

    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    lw = torch.randn(D, device=DEVICE, dtype=dtype)
    lb = torch.randn(D, device=DEVICE, dtype=dtype)
    w = (torch.randn(D, D, device=DEVICE, dtype=dtype) * (D**-0.5)).contiguous()
    eps = 1e-5

    def _x():
        # Pair-shaped, not pre-flattened: `layernorm_linear_triton` flattens to (M, K) itself and
        # reads the shape key off what it was handed, so a (M, D) input is refused by the
        # shape_key guard -- every `layernorm_linear_triton` row came back `failed`. Each backend
        # below that genuinely wants a matrix flattens it explicitly, so all rows still measure
        # the same numbers on the same data. drivers/layernorm_linear's `layernorm_linear_fwd_triton` carries
        # the same note.
        torch.manual_seed(1)
        return torch.randn(1, L, L, D, device=DEVICE, dtype=dtype).contiguous()

    def ref(x):
        return F.linear(F.layer_norm(x, (D,), lw, lb, eps), w)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch"
    elif implementation == "layernorm_linear_triton":
        from miniworld_engine.kernels.layernorm_linear.interface import (
            layernorm_linear_triton,
        )
        kfn = lambda x: layernorm_linear_triton(x, lw, lb, w, None, eps)
        path = "kernels.layernorm_linear.triton.fused"
    elif implementation == "layernorm_linear_cute":
        from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
            layernorm_linear_cute,
        )
        kfn = lambda x: layernorm_linear_cute(
            x.reshape(-1, D), lw, lb, w, None, eps).reshape(x.shape)
        path = "kernels.layernorm_linear.cute.gemm_layernorm_linear"
    elif implementation == "layernorm_linear_cute_fused":
        from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
            layernorm_linear_cute_fused,
        )
        kfn = lambda x: layernorm_linear_cute_fused(
            x.reshape(-1, D), lw, lb, w, None, eps).reshape(x.shape)
        path = "kernels.layernorm_linear.cute.gemm_layernorm_linear_fused"
    elif implementation == "layernorm_linear_te":
        from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
            layernorm_linear_te_fn,
        )
        # Flattened here, with length=L passed explicitly: TE wants a matrix, and once it is one
        # the pair length is no longer readable from the shape.
        kfn = lambda x: layernorm_linear_te_fn(
            x.reshape(-1, D), lw, lb, w, None, eps, length=L).reshape(x.shape)
        path = "kernels.layernorm_linear.triton.te_style"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = _acc_fwd(kfn(_x()), ref(_x()))
    return _fwd_result(conf, kfn, (_x(),), acc=acc, path=path, ref="pytorch", dtype=tname)


def bench_kernel_transition_b2b(conf, seq_len, implementation, fabric):
    """SwiGLU MLP (transition, back-to-back): out = squeeze(silu(LN(x)@Wa)*(LN(x)@Wb)). Rows: pytorch,
    triton_transition_fused, cute_transition_fused, transition_b2b_ktiled(unverified)."""
    if conf.compile and implementation == "transition_b2b_ktiled":
        raise UnsupportedBenchmark(
            "transition_b2b_ktiled has no opaque compile entry; its direct Triton launcher "
            "fails Dynamo tracing. Use compile=False for this exact implementation")
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.transition import Transition

    D, L, n = conf.d_pair, seq_len, 4
    torch.manual_seed(0)
    ref_mod = Transition(D, n=n, implementation=ImplementationType.PYTORCH).to(DEVICE).to(dtype)
    for lin in (ref_mod.expand_a, ref_mod.expand_b, ref_mod.squeeze):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
    lw, lb = ref_mod.ln_in.weight, ref_mod.ln_in.bias
    wa, wb, wsq = ref_mod.expand_a.weight, ref_mod.expand_b.weight, ref_mod.squeeze.weight
    eps = ref_mod.ln_in.eps

    def _x():
        torch.manual_seed(1)
        return torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)

    # The Transition op INCLUDES the residual -- `triton_transition_fused` folds `+x` into its
    # squeeze epilogue and there is no flag to turn that off. So the reference has to include it
    # too, and the two rows below that compute the raw op (`cute_transition_fused` has no residual
    # epilogue; `transition_b2b_ktiled` is the K>128 tile, not the Transition entry) get an
    # explicit `+ x` so all four rows are timed and scored on the SAME function.
    ref_fn = lambda x: ref_mod._torch_forward(x) + x
    if implementation == "pytorch":
        kfn, path = ref_fn, "module.reference.torch"
    elif implementation == "triton_transition_fused":
        from miniworld_engine.kernels import triton_transition_fused
        kfn = lambda x: triton_transition_fused(x, lw, lb, wa, wb, wsq, n, eps)
        path = "kernels.transition.triton.fused"
    elif implementation == "cute_transition_fused":
        from miniworld_engine.kernels import cute_transition_fused
        kfn = lambda x: cute_transition_fused(x, lw, lb, wa, wb, wsq, n, eps) + x
        path = "kernels.transition.cute.fused"
    elif implementation == "transition_b2b_ktiled":
        from miniworld_engine.kernels.transition.triton.fused import (
            transition_b2b_ktiled,
        )
        kfn = lambda x: transition_b2b_ktiled(
            x.reshape(L * L, D), lw, lb, wa, wb, wsq, eps).reshape(1, L, L, D) + x
        path = "kernels.transition.triton.fused.b2b_ktiled"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = _acc_fwd(kfn(_x()), ref_fn(_x()))
    return _fwd_result(conf, kfn, (_x(),), acc=acc, path=path, ref="module.reference.torch", dtype=tname)


def bench_kernel_layernorm(conf, seq_len, implementation, fabric):
    """LayerNorm forward: y = LN(x)*w + b. Rows: pytorch, triton_layernorm, layernorm_dispatch,
    quack_cute, triton_layernorm_lowreg(dep)."""
    import torch.nn.functional as F

    D, L = conf.d_pair, seq_len
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    torch.manual_seed(0)
    w = torch.randn(D, device=DEVICE, dtype=dtype)
    b = torch.randn(D, device=DEVICE, dtype=dtype)
    eps = 1e-5

    def _x():
        torch.manual_seed(1)
        return torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)

    def ref(x):
        return F.layer_norm(x, (D,), w, b, eps)

    if implementation == "pytorch":
        kfn, path = ref, "torch.nn.functional.layer_norm"
    elif implementation == "triton_layernorm":
        from miniworld_engine.kernels import triton_layernorm
        kfn = lambda x: triton_layernorm(x, w, b, eps)
        path = "kernels.layernorm.triton.main"
    elif implementation == "layernorm_dispatch":
        from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
        kfn = lambda x: layernorm_kernel(x, w, b, eps)
        path = "kernels.layernorm.interface"
    elif implementation == "quack_cute":
        from miniworld_engine.kernels.layernorm.cute.quack_adapter import (
            quack_layernorm_fwd,
        )
        kfn = lambda x: quack_layernorm_fwd(x, w, b, eps)
        path = "kernels.layernorm.cute.quack"
    elif implementation == "triton_layernorm_lowreg":
        from miniworld_engine.kernels.layernorm.triton.lowreg import (
            triton_layernorm_lowreg,
        )
        kfn = lambda x: triton_layernorm_lowreg(x, w, b, eps)
        path = "kernels.layernorm.triton.lowreg"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = _acc_fwd(kfn(_x()), ref(_x()))
    return _fwd_result(conf, kfn, (_x(),), acc=acc, path=path,
                       ref="torch.nn.functional.layer_norm", dtype=tname)


def bench_kernel_adaln(conf, seq_len, implementation, fabric):
    """Adaptive LayerNorm forward: y = sigma(scale)*LN(x) + bias, scale/bias = Linear(LN(cond)).
    Rows: pytorch, adaln_inference, adaln_lnfold. `triton_adaln` and `adaln_fused3` are
    gone: no module dispatched to either, so they were deleted with main.py and fused3.py."""
    from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
    from miniworld_engine.modules.exceptions import ImplementationType

    # The ATOM side: the model's `atom_dit` builds both adaln and ConditionedTransition at
    # d_hidden = d_cond = 128 (AlphaFold-3's c_atom). The number was already right; the name
    # was not -- `d_pair` is the PAIR width, which neither module ever sees. The TOKEN side
    # (768/384) is covered by the module benches.
    D, L = conf.d_single_atom, seq_len
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    torch.manual_seed(0)
    ref_mod = AdaptiveLayerNorm(D, D, implementation=ImplementationType.PYTORCH).to(DEVICE).to(dtype)
    for lin in (ref_mod.to_scale, ref_mod.to_bias):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
        if lin.bias is not None:
            torch.nn.init.normal_(lin.bias, std=D**-0.5)
    clw = ref_mod.ln_cond.weight
    sw, sb, bw = ref_mod.to_scale.weight, ref_mod.to_scale.bias, ref_mod.to_bias.weight
    ex, ec = ref_mod.ln_in.eps, ref_mod.ln_cond.eps

    # SINGLE, not pair. This bench used to build `(1, L, L, D)` -- 147,456 rows at L=384 -- and it
    # is the only adaln-vs-torch measurement in the repo that showed a win (5-6x on every card).
    # Nothing in src/ hands adaln a 4-D pair activation: the two construction sites,
    # augmented_attention/module.py:62 and conditioned_transition/module.py:79, both pass
    # `(B, L, D)`. So the win was measured on a shape the model does not run, while the module
    # bench -- on the shape it does -- has adaln LOSING at inference on all five cards.
    #
    # It also collided in the cache. adaln is `level=atom` and keys on `atom_key(length_of(shape))`
    # = `shape[-2]` floored, so the pair bench at L=384 and the module bench at L=384 resolved to
    # the SAME entry with 12x different row counts. That is the failure `shape_key.BOTH_ROWS`
    # documents at 1.73x, and `level=atom` has no protection from it because a pair launch was
    # assumed impossible -- it was the bench, not the model, that made one.
    def _xc():
        torch.manual_seed(1)
        # `(A, 1, L, D)`, the shape production hands adaln: augmented_attention/module.py:166
        # passes `single`, which carries the augmentation axis. NOT `(1, L, L, D)` -- that was a
        # pair activation nothing constructs -- and not `(1, L, D)` either, which is the same L at
        # 1/A the rows. `length_of` is `shape[-2]` and reads L from all three, so all three land in
        # one atom bucket at wildly different row counts; only this one is the row count the model
        # actually launches. bench_module_adaptive_layernorm builds the same shape.
        return (torch.randn(conf.n_augment, 1, L, D, device=DEVICE, dtype=dtype),
                torch.randn(conf.n_augment, 1, L, D, device=DEVICE, dtype=dtype))

    if implementation == "pytorch":
        kfn, path = (lambda x, c: ref_mod(x, c)), "module.reference.torch"
    elif implementation == "adaln_inference":
        from miniworld_engine.kernels.adaln.triton.inference import adaln_inference
        kfn = lambda x, c: adaln_inference(x, c, clw, sw, sb, bw, ex, ec)
        path = "kernels.adaln.triton.inference"
    elif implementation == "adaln_lnfold":
        from miniworld_engine.kernels.adaln.triton.inference import (
            adaln_inference_lnfold,
        )
        from miniworld_engine.kernels.layernorm_linear.cute import fold_for_gemm
        _wcat = torch.cat([sw, bw], dim=0).contiguous()
        _bcat = torch.cat([sb, sb.new_zeros(D)], dim=0).contiguous()
        _pf = fold_for_gemm(_wcat, clw, clw.new_zeros(clw.shape), _bcat, w2_dtype=dtype)
        kfn = lambda x, c: adaln_inference_lnfold(
            x, c, clw, sw, sb, bw, ex, ec, weight_cat=_wcat, bias_cat=_bcat, prefolded=_pf)
        path = "kernels.adaln.triton.inference.lnfold"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    xc, cc = _xc()
    acc = _acc_fwd(kfn(xc, cc), ref_mod(xc, cc))
    return _fwd_result(conf, kfn, _xc(), acc=acc, path=path, ref="module.reference.torch", dtype=tname)


def bench_kernel_triangle_attention(conf, seq_len, implementation, fabric):
    """Triangle self-attention: softmax(QK^T*d^-0.5 + pair_bias)*V. q,k,v:(1,H,L,L,dh) bias:(1,H,L,L).
    Rows: pytorch(SDPA), triton_triangle_attention, triton_triangle_attention_miniworld(dep), triton_triangle_attention_perf(dep)."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    if conf.d_pair < 32 or conf.d_pair % 32:
        raise UnsupportedBenchmark("attention width must be a positive multiple of head width 32")
    import torch.nn.functional as F

    L, dh = seq_len, 32
    H = max(1, conf.d_pair // dh)

    def mk():
        torch.manual_seed(1)
        q = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=dtype)
        k = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=dtype)
        v = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=dtype)
        bias = torch.randn(1, H, L, L, device=DEVICE, dtype=dtype)
        return q, k, v, bias

    def ref(q, k, v, bias):
        qf, kf, vf = (t.reshape(H, L, L, dh) for t in (q, k, v))
        mask = bias.reshape(H, 1, L, L)
        return F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask).reshape(1, H, L, L, dh)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch.sdpa"
    elif implementation == "triton_triangle_attention":
        from miniworld_engine.kernels import triton_triangle_attention_pair_bias as fn
        kfn = lambda q, k, v, b: fn(q, k, v, b)
        path = "kernels.triangle_attention.triton.main"
    elif implementation == "triton_triangle_attention_atomic":
        from miniworld_engine.kernels.triangle_attention.triton.atomic import (
            triton_triangle_attention_pair_bias as fn,
        )
        kfn = lambda q, k, v, b: fn(q, k, v, b)
        path = "kernels.triangle_attention.triton.atomic"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = {}
    try:
        qc = mk()
        acc = _acc_fwd(kfn(*qc), ref(*qc))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch.sdpa", dtype=tname, parameter_dtype="")


def bench_kernel_bias_only_attention(conf, seq_len, implementation, fabric):
    """Bias-only attention: out[i,j,d]=sum_k softmax_k(bias[j,k])*v[i,k,d]. v:(1,H,L,L,dh) bias:(1,H,L,L).
    Rows: pytorch, triton_bias_only_attention."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    if conf.d_pair < 32 or conf.d_pair % 32:
        raise UnsupportedBenchmark("attention width must be a positive multiple of head width 32")
    import torch.nn.functional as F

    L, dh = seq_len, 32
    H = max(1, conf.d_pair // dh)

    def mk():
        torch.manual_seed(1)
        v = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=dtype)
        bias = torch.randn(1, H, L, L, device=DEVICE, dtype=dtype)
        return v, bias

    def ref(v, bias):
        p = F.softmax(bias.float(), dim=-1).to(v.dtype)
        return torch.einsum("bhjk,bhikd->bhijd", p, v)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch.einsum"
    elif implementation == "triton_bias_only_attention":
        from miniworld_engine.kernels import triton_bias_only_attention
        kfn = lambda v, b: triton_bias_only_attention(v, b)
        path = "kernels.bias_only_attention.triton.main"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch.einsum", dtype=tname, parameter_dtype="")


def bench_kernel_augmented_attention(conf, seq_len, implementation, fabric):
    """Augmented pair-bias attention: softmax(q.k*d^-0.5 + bias)*v. q,k,v:(A,1,L,H,dh) bias:(1,L,L,H).
    Rows: pytorch, triton_augmented_attention, augmented_attention_memory_efficient."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    if conf.d_pair < 32 or conf.d_pair % 32:
        raise UnsupportedBenchmark("attention width must be a positive multiple of head width 32")
    import torch.nn.functional as F

    L, A, H, dh = seq_len, conf.n_augment, conf.d_pair // 32, 32

    def mk():
        torch.manual_seed(1)
        q = torch.randn(A, 1, L, H, dh, device=DEVICE, dtype=dtype)
        k = torch.randn(A, 1, L, H, dh, device=DEVICE, dtype=dtype)
        v = torch.randn(A, 1, L, H, dh, device=DEVICE, dtype=dtype)
        bias = torch.randn(1, L, L, H, device=DEVICE, dtype=dtype)
        return q, k, v, bias

    def ref(q, k, v, bias):
        att = torch.einsum("abihd,abjhd->abhij", q * (dh**-0.5), k)
        att = att + bias.permute(0, 3, 1, 2)[None]
        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        return torch.einsum("abhij,abjhd->abihd", att, v)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch.einsum"
    elif implementation == "triton_augmented_attention":
        from miniworld_engine.kernels import triton_augmented_attention_pair_bias
        # The dispatch wrapper's default: the compute-efficient backend in triton/main.py. The
        # memory-efficient one is the augmented_attention_memory_efficient row below.
        kfn = lambda q, k, v, b: triton_augmented_attention_pair_bias(q, k, v, b)
        path = "kernels.augmented_attention.triton.main"
    elif implementation == "augmented_attention_memory_efficient":
        from miniworld_engine.kernels.augmented_attention.triton.memory_efficient import (
            triton_augmented_attention_pair_bias as fn,
        )
        kfn = lambda q, k, v, b: fn(q, k, v, b)
        path = "kernels.augmented_attention.triton.memory_efficient"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch.einsum", dtype=tname, parameter_dtype="")


def bench_kernel_fused_ln_mask(conf, seq_len, implementation, fabric):
    """Fused LayerNorm+mask: out = LN(x)*mask (per-row scale). Rows: pytorch, fused_ln_mask."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    import torch.nn.functional as F

    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    w = torch.randn(D, device=DEVICE, dtype=dtype)
    b = torch.randn(D, device=DEVICE, dtype=dtype)
    eps = 1e-5

    def mk():
        torch.manual_seed(1)
        x = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
        mask = (torch.rand(1, L, L, device=DEVICE) > conf.mask_prob).to(dtype)
        return x, mask

    def ref(x, mask):
        return F.layer_norm(x, (D,), w, b, eps) * mask[..., None]

    if implementation == "pytorch":
        kfn, path = ref, "pytorch"
    elif implementation == "fused_ln_mask":
        from miniworld_engine.kernels.fused_ln_mask.cute.fused_ln_mask import (
            fused_ln_mask,
        )
        kfn = lambda x, m: fused_ln_mask(x, w, b, m, eps)
        path = "kernels.fused_ln_mask.cute"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch", dtype=tname)


def bench_kernel_gemm_gate(conf, seq_len, implementation, fabric):
    """Gated output projection (tm2 back half): out = sigma(xg@Wg^T)*(xo@Wp^T). Rows: pytorch,
    tm2_cute, triton_tm2."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    wg = (torch.randn(D, D, device=DEVICE, dtype=dtype) * (D**-0.5)).contiguous()  # (N,K)
    wp = (torch.randn(D, D, device=DEVICE, dtype=dtype) * (D**-0.5)).contiguous()

    def mk():
        torch.manual_seed(1)
        return (torch.randn(1, L, L, D, device=DEVICE, dtype=dtype),
                torch.randn(1, L, L, D, device=DEVICE, dtype=dtype))

    def ref(xg, xo):
        return torch.sigmoid(xg @ wg.t()) * (xo @ wp.t())

    if implementation == "pytorch":
        kfn, path = ref, "pytorch"
    elif implementation == "tm2_cute":
        # cuequiv wrapper deleted 2026-08-04 — this now benches OUR from-scratch tm2 kernel.
        from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import (
            tm2_dual_from_scratch,
        )
        kfn = lambda xg, xo: tm2_dual_from_scratch(xg, xo, wg, wp)  # (wg/wp are (N,K))
        path = "kernels.tm2.cute"
    elif implementation == "triton_tm2":
        from miniworld_engine.kernels.tm2.triton.main import triton_tm2
        wgt, wpt = wg.t().contiguous(), wp.t().contiguous()  # kernel computes x@W (K,N form)
        # Pre-flatten, and TritonTM2Function's `length_of(x.shape)` sees M = L*L and refuses.
        kfn = lambda xg, xo: triton_tm2(xg, xo, wgt, wpt).reshape(1, L, L, D)
        path = "kernels.tm2.triton.main"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch", dtype=tname)


def bench_kernel_conditioned_transition_tail(conf, seq_len, implementation, fabric):
    """Post-adaLN conditioned-transition tail: out=squeeze(silu(x@Wa)*(x@Wb)); y=sigma(cond@Wsc+b)*out.
    Rows: pytorch, triton_cond_transition.

    Honours ``conf.precision``, like every other kernel bench in this file. It used to hardcode
    fp32 three times over -- ``.float()`` on the module, ``dtype=torch.float32`` in ``_xc()`` and
    a literal ``dtype="float32"`` in the result -- while ``bench_policy`` sweeps this target at
    BOTH precisions (its registry family declares ``bf16|fp32``). So the bf16 row and the fp32 row
    were the same fp32 measurement recorded twice, and the bf16 run wrote its numbers to an
    artifact named ``..._bf16-mixed_...``. The op is tuned at both precisions -- the A100 cache
    holds ``bfloat16|...`` and ``float32|...`` entries for ``cond_transition_fwd_b2b_triton`` --
    so the bf16 half was declared, cached, and never measured."""
    from miniworld_engine import kernels
    from miniworld_engine.modules.conditioned_transition.module import (
        ConditionedTransition,
    )
    from miniworld_engine.modules.exceptions import ImplementationType

    # The atom side, as in the adaln kernel benches. n is the transition's expansion factor:
    # the model sets `condition.transition_n: 2`, so 4 measured a shape twice as wide as the
    # one the model launches, and the driver tunes for n=2.
    D, L, n = conf.d_single_atom, seq_len, 2
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    torch.manual_seed(0)
    ref_mod = ConditionedTransition(
        D, D, n=n, implementation=ImplementationType.PYTORCH).to(DEVICE).to(dtype)
    for lin in (ref_mod.expand_a, ref_mod.expand_b, ref_mod.squeeze):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
    wa, wb, ws = ref_mod.expand_a.weight, ref_mod.expand_b.weight, ref_mod.squeeze.weight
    wsc, bsc = ref_mod.to_scale.weight, ref_mod.to_scale.bias

    def _xc():
        torch.manual_seed(1)
        # `(A, 1, L, D)`, the shape the module hands this family -- see bench_kernel_adaln. This
        # built `(1, L, L, D)`, a pair activation with M = L*L, and passed `length=L` into an
        # atom-level family: at L=512 it asked for the config tuned at 512 rows to run 262,144.
        return (torch.randn(conf.n_augment, 1, L, D, device=DEVICE, dtype=dtype),
                torch.randn(conf.n_augment, 1, L, D, device=DEVICE, dtype=dtype))

    # The kernel is the POST-AdaLN tail: ConditionedTransition.forward runs `ada_ln_in` first and
    # only then calls it. Normalize once here so both sides see the same input.
    ref_fn = lambda x, c: ref_mod._reference(x, c)
    if implementation == "pytorch":
        kfn, path = ref_fn, "module.reference.torch"
    elif implementation == "triton_cond_transition":
        raw = kernels.cond_transition_inference_dispatch
        # `length=L`: the dispatch takes the pre-flatten L explicitly precisely because its
        # caller has already flattened to (M, K). Omitting it left the inner path calling
        # `length_of` on a 2-D shape, which the guard refuses.
        kfn = lambda x, c: raw(
            x.reshape(-1, D), c.reshape(-1, D),
            wa, wb, ws, wsc, bsc, length=L).reshape(conf.n_augment, 1, L, D)
        path = "kernels.conditioned_transition.triton"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    xc, cc = _xc()
    with torch.no_grad():
        normalized = ref_mod.ada_ln_in(xc, cc).detach()
    acc = _acc_fwd(kfn(normalized, cc), ref_fn(normalized, cc))
    return _fwd_result(conf, kfn, (normalized, cc), acc=acc, path=path, ref="module.reference.torch", dtype=tname)


# ---- BACKWARD operations (pure-function launchers; cudagraph-safe) ----------------------------
def bench_kernel_layernorm_bwd(conf, seq_len, implementation, fabric):
    """LayerNorm backward: (dy,x,w,mean,rstd)->(dx,dw,db). Rows: pytorch(pure), triton_atomic,
    triton_persistent. CSV metrics cover dx/dw/db together; logs also report each gradient."""
    import json

    from miniworld_engine import settings
    D, L = conf.d_pair, seq_len
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    weight_precision = conf.layernorm_weight_precision
    weight_dtype = (dtype if weight_precision is None else
                    torch.float32 if weight_precision == FP32_PRECISION else BF16)
    tname = str(dtype).replace("torch.", "")
    wname = str(weight_dtype).replace("torch.", "")
    torch.manual_seed(0)
    w = torch.randn(D, device=DEVICE, dtype=weight_dtype)
    torch.manual_seed(1)
    # The pair activation (1, L, L, D), NOT its (M, D) flattening. `_bwd_*_impl` reshapes
    # internally and reads `both_key(length_of(x.shape))` off the 4-D shape, so a pre-flattened
    # (M, D) is refused by the shape_key guard -- 18 of this file's bench rows came back
    # `failed: shape (..., D) is already flattened`. drivers/layernorm fixed this; the bench did not.
    x = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    dy = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    eps = 1e-5
    xf = x.reshape(-1, D).float()
    dyf = dy.reshape(-1, D)
    mean = xf.mean(-1)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)

    def torch_bwd():
        xhat = (xf - mean[:, None]) * rstd[:, None]
        dxhat = dyf.float() * w.float()
        dx = rstd[:, None] * (dxhat - dxhat.mean(-1, keepdim=True)
                              - xhat * (dxhat * xhat).mean(-1, keepdim=True))
        dwt = (dyf.float() * xhat).sum(0)
        dbt = dyf.float().sum(0)
        return dx.to(dtype), dwt.to(weight_dtype), dbt.to(weight_dtype)

    if implementation == "pytorch":
        kfn, path = torch_bwd, "pytorch"
    elif implementation in {"triton_atomic", "triton_persistent"}:
        # No `triton_partial`: 3d5a0a2c deleted the partial backward path and `_bwd_partial_impl`
        # with it -- `_VALID_BWD_PATHS` is {"persistent", "atomic", "cuda"}. The import named it
        # anyway, so all three rows died on the import, not just the one that no longer exists.
        from miniworld_engine.kernels.layernorm.compile_native import _dispatch_bwd

        # Use the same opaque entry as production autograd in every execution regime.
        # Tracing the private launchers enters shape-key construction and graph-breaks.
        kfn = lambda: _dispatch_bwd(dy, x, w, mean, rstd)
        path = f"kernels.layernorm.compile_native._dispatch_bwd[{implementation}]"
    elif implementation == "cuda":
        # Hand-CUDA vectorized backward; the shipped dispatch routes bf16 128<=N<=512 here.
        # Outside that gate the dispatch keeps triton, so report NaN (not applicable).
        if dtype is not BF16 or weight_dtype is not BF16 or not (128 <= D <= 512):
            raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")
        from miniworld_engine.kernels.layernorm.compile_native import _dispatch_bwd

        kfn = lambda: _dispatch_bwd(dy, x, w, mean, rstd)
        path = "kernels.layernorm.compile_native._dispatch_bwd[cuda]"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    pinned_path = {"triton_atomic": "atomic", "triton_persistent": "persistent",
                   "cuda": "cuda"}.get(implementation)
    previous_path = settings.current().layernorm_bwd_path
    if pinned_path is not None:
        settings.configure(layernorm_bwd_path=pinned_path)
    try:
        if conf.compile:
            kfn = compile_for_benchmark(kfn, fullgraph=True)
        actual, expected = kfn(), torch_bwd()
        gradients = {}
        for name, got, want in zip(("dx", "dw", "db"), actual, expected, strict=True):
            maximum, relative, cosine = tensor_metrics(got, want)
            gradients[name] = {"max_abs": maximum, "rel_frob": relative, "cosine": cosine,
                               "dtype": str(got.dtype).replace("torch.", ""),
                               "reference_dtype": str(want.dtype).replace("torch.", "")}
        print("LAYERNORM_BWD_GRADIENTS " + json.dumps({
            "implementation": implementation, "seq_len": L, "d_pair": D,
            "input_dtype": tname, "parameter_dtype": wname, "compiled": conf.compile,
            "cudagraph": conf.cudagraph, "gradients": gradients}), flush=True)
        acc = _acc_grad(_flat(list(actual)), _flat(list(expected)))
        return measured_result(
            conf=conf, func=kfn, grad_to_none=[], params=[], is_train=False,
            input_dtype=tname, parameter_dtype=wname, execution_path=path, reference="pytorch",
        )._replace(**acc)
    finally:
        if pinned_path is not None:
            settings.configure(layernorm_bwd_path=previous_path)


def bench_kernel_gemm_gate_bwd(conf, seq_len, implementation, fabric):
    """Gate-elementwise backward: bwd of y=sigma(x_n@Wg)*proj -> (d_proj, dx_n, dWg). Rows: pytorch(pure),
    gate_elem_bwd. Cosine on concatenated grads."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    wg = (torch.randn(D, D, device=DEVICE, dtype=dtype) * (D**-0.5)).contiguous()  # (K,N)
    torch.manual_seed(1)
    x_n = torch.randn(L * L, D, device=DEVICE, dtype=dtype)
    proj = torch.randn(L * L, D, device=DEVICE, dtype=dtype)
    dy = torch.randn(L * L, D, device=DEVICE, dtype=dtype)
    gate = torch.sigmoid(x_n.float() @ wg.float()).to(dtype)

    def torch_bwd():
        g, dyf, pf = gate.float(), dy.float(), proj.float()
        d_proj = dyf * g
        d_glog = dyf * pf * g * (1 - g)
        dx = d_glog @ wg.float().t()
        dwg = x_n.float().t() @ d_glog
        return d_proj.to(dtype), dx.to(dtype), dwg.to(dtype)

    if implementation == "pytorch":
        kfn, path = torch_bwd, "pytorch"
    elif implementation == "gate_elem_bwd":
        from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
            gate_elem_bwd,
        )
        # This isolated gate benchmark has no dropout; the production custom op
        # requires an explicit row scale and sequence length even for p_drop=0.
        dropscale = torch.ones((L, D), device=DEVICE, dtype=dtype)
        kfn = lambda: gate_elem_bwd(dy, x_n, proj, gate, wg, dropscale, L)
        path = "kernels.trimul_inproj.triton.gate_elem"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    import json

    acc = _acc_grad(_flat(list(kfn())), _flat(list(torch_bwd())))
    shapes = json.dumps({"dy": list(dy.shape), "x_n": list(x_n.shape), "proj": list(proj.shape), "weight": list(wg.shape)})
    return measured_result(
        conf=conf, func=kfn, grad_to_none=[], params=[], is_train=False,
        input_dtype=tname, parameter_dtype=tname, execution_path=path, reference="pytorch",
    )._replace(**acc, input_shapes=shapes)


def bench_kernel_dual_gemm_epilogue_bwd(conf, seq_len, implementation, fabric):
    """Gated dual-GEMM front backward: (d_left,d_right)->dx_n + 4 weight grads. Rows: pytorch(pure),
    front_bwd_fused. Cosine on concatenated (dx_n|dWL|dWLg|dWR|dWRg)."""
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    D, L, H = conf.d_pair, seq_len, conf.d_pair
    torch.manual_seed(0)

    def _w():
        return (torch.randn(D, H, device=DEVICE, dtype=dtype) * (D**-0.5)).contiguous()

    WL, WLg, WR, WRg = _w(), _w(), _w(), _w()
    torch.manual_seed(1)
    x_n = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    d_left = torch.randn(1, L, L, H, device=DEVICE, dtype=dtype)
    d_right = torch.randn(1, L, L, H, device=DEVICE, dtype=dtype)

    def torch_bwd():
        xf = x_n.reshape(L * L, D).float()
        dl, dr = d_left.reshape(L * L, H).float(), d_right.reshape(L * L, H).float()
        pL, gL = xf @ WL.float(), torch.sigmoid(xf @ WLg.float())
        pR, gR = xf @ WR.float(), torch.sigmoid(xf @ WRg.float())
        d_pL, d_gL = dl * gL, dl * pL * gL * (1 - gL)
        d_pR, d_gR = dr * gR, dr * pR * gR * (1 - gR)
        dxn = (d_pL @ WL.float().t() + d_gL @ WLg.float().t()
               + d_pR @ WR.float().t() + d_gR @ WRg.float().t())
        return (dxn.reshape(1, L, L, D).to(dtype), (xf.t() @ d_pL).to(dtype), (xf.t() @ d_gL).to(dtype),
                (xf.t() @ d_pR).to(dtype), (xf.t() @ d_gR).to(dtype))

    if implementation == "pytorch":
        kfn, path = torch_bwd, "pytorch"
    elif implementation == "front_bwd_fused":
        from miniworld_engine.kernels.trimul_inproj.triton.back_fused import (
            front_bwd_fused,
        )
        xf = x_n.reshape(L * L, D)
        gLlog, pL = xf @ WLg, xf @ WL
        gRlog, pR = xf @ WRg, xf @ WR
        left_il = torch.stack([gLlog, pL], dim=-1).reshape(L * L, 2 * H)
        right_il = torch.stack([gRlog, pR], dim=-1).reshape(L * L, 2 * H)
        preact = torch.cat([left_il, right_il], dim=-1).reshape(
            1, L, L, 4 * H).permute(0, 3, 1, 2).contiguous()
        dlb = d_left.permute(0, 3, 1, 2).contiguous()
        drb = d_right.permute(0, 3, 1, 2).contiguous()
        kfn = lambda: front_bwd_fused(dlb, drb, preact, x_n, WL, WLg, WR, WRg)
        path = "kernels.trimul_inproj.triton.back_fused"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")

    import json

    acc = _acc_grad(_flat(list(kfn())), _flat(list(torch_bwd())))
    shapes = json.dumps({"x_n": list(x_n.shape), "d_left": list(d_left.shape), "d_right": list(d_right.shape)})
    return measured_result(
        conf=conf, func=kfn, grad_to_none=[], params=[], is_train=False,
        input_dtype=tname, parameter_dtype=tname, execution_path=path, reference="pytorch",
    )._replace(**acc, input_shapes=shapes)


# ---- BACKWARD operations (autograd; backward-only timing via autograd.grad) -------------------
def bench_kernel_adaln_bwd(conf, seq_len, implementation, fabric):
    """adaLN backward (autograd, backward-only). Rows: pytorch, adaln_train.
    Cosine on dx vs pytorch autograd. `triton_adaln` / `adaln_fused3` are gone -- no module
    reached them.

    THE LEAF SET IS THE MEASUREMENT. `AdaLNTrainFn.backward` is ONE autograd node: it always
    produces all six gradients (dx, dcond, dlnw, dWs, dsb, dWb) because autograd cannot prune
    inside a node. The torch reference is a graph of ~8 nodes, so asking `autograd.grad` for a
    SUBSET prunes the rest -- with `[x, cond]` it never runs the two wgrad GEMMs, the ln_cond
    gamma reduction or the scale-bias sum. GPU time and launch count per backward, A100,
    A=32, d=128, sum of kernel self-times from the torch profiler:

        leaves              reference (L=384 / L=1024)   ours (L=384 / L=1024)
        [x, cond]           0.079 / 0.186 ms,  8 kern    0.059 / 0.135 ms, 8 kern
        [x, cond, *params]  0.181 / 0.449 ms, 15 kern    0.059 / 0.135 ms, 8 kern

    Ours does not change -- it never had the choice -- and the reference more than doubles.
    What this bench reported, `bench_kernel adaln_bwd --no-build`, before -> after the fix:

        L        384    512    640    768    896   1024
        before  0.40x  0.52x  0.49x  0.53x  0.51x  0.47x
        after   0.77x  0.78x  0.79x  0.80x  0.84x  1.00x

    Roughly half the "2x loss" was the pruning; the rest is host dispatch -- see the return.
    """
    if conf.compile:
        raise UnsupportedBenchmark(
            "backward-only autograd uses a prebuilt eager graph; compiled backward is unsupported")
    from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
    from miniworld_engine.modules.exceptions import ImplementationType

    # The ATOM side: the model's `atom_dit` builds both adaln and ConditionedTransition at
    # d_hidden = d_cond = 128 (AlphaFold-3's c_atom). The number was already right; the name
    # was not -- `d_pair` is the PAIR width, which neither module ever sees. The TOKEN side
    # (768/384) is covered by the module benches.
    D, L = conf.d_single_atom, seq_len
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    torch.manual_seed(0)
    ref_mod = AdaptiveLayerNorm(D, D, implementation=ImplementationType.PYTORCH).to(DEVICE).to(dtype)
    for lin in (ref_mod.to_scale, ref_mod.to_bias):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
        if lin.bias is not None:
            torch.nn.init.normal_(lin.bias, std=D**-0.5)
    clw = ref_mod.ln_cond.weight
    sw, sb, bw = ref_mod.to_scale.weight, ref_mod.to_scale.bias, ref_mod.to_bias.weight
    ex, ec = ref_mod.ln_in.eps, ref_mod.ln_cond.eps
    torch.manual_seed(1)
    # `(A, 1, L, D)` -- the production shape, see bench_kernel_adaln.
    x0 = torch.randn(conf.n_augment, 1, L, D, device=DEVICE, dtype=dtype)
    c0 = torch.randn(conf.n_augment, 1, L, D, device=DEVICE, dtype=dtype)
    dy = torch.randn(conf.n_augment, 1, L, D, device=DEVICE, dtype=dtype)
    xr, cr = x0.clone().requires_grad_(True), c0.clone().requires_grad_(True)
    ref_mod(xr, cr).backward(dy)
    ref_grads = []
    for tensor in (xr, cr, clw, sw, sb, bw):
        if tensor is None:
            continue
        gradient = tensor.grad
        if gradient is None:
            raise RuntimeError("reference backward did not produce every requested gradient")
        ref_grads.append(gradient.detach().clone())

    x, c = x0.clone().requires_grad_(True), c0.clone().requires_grad_(True)
    if implementation == "pytorch":
        out, path = ref_mod(x, c), "module.reference.torch"
    elif implementation == "adaln_train":
        from miniworld_engine.kernels.adaln.triton.training import adaln_train
        out = adaln_train(x, c, clw, sw, sb, bw, ex, ec)
        path = "kernels.adaln.triton.training"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")
    # Every leaf `adaln_train`'s node differentiates, so both rows are asked for the same six
    # gradients. `ln_in` is `elementwise_affine=False` and `ln_cond`/`to_bias` are bias-free, so
    # these four ARE the module's whole parameter set -- the reference is not being charged for
    # anything the kernel skips.
    leaves = [t for t in (x, c, clw, sw, sb, bw) if t is not None]
    # WHAT THIS NUMBER IS. At the atom width both sides are CPU-DISPATCH bound, not kernel bound:
    # at L=1024 the reference costs 0.470 ms of host enqueue against 0.449 ms of GPU, ours
    # 0.565 ms against 0.135 ms, and a back-to-back event bracket sits on the enqueue time for
    # both -- the GPU is starved. That is why both columns are nearly FLAT in L while the tokens
    # grow 7x. On the GPU work itself ours wins 3.3x; the bench metric is the Python cost of ~11
    # eager launches (two of them `torch.library` custom ops wrapping Triton) against torch's
    # ~15 C++ autograd nodes. `--cudagraph manual` would remove it from both sides, but a graph
    # capture of `autograd.grad` fails here (`cudaErrorStreamCaptureInvalidated`; every row of
    # the committed a6000 adaln_bwd table is that failure), so there is no graphed variant yet.
    return _bwd_autograd_result(conf, out, leaves, dy, ref_grads, path=path,
                                ref="module.reference.torch", dtype=tname)


def bench_kernel_transition_b2b_bwd(conf, seq_len, implementation, fabric):
    """Transition backward (autograd, backward-only). Rows: pytorch, triton_transition_fused,
    cute_transition_fused. Cosine on dx vs pytorch autograd."""
    if conf.compile:
        raise UnsupportedBenchmark(
            "backward-only autograd uses a prebuilt eager graph; compiled backward is unsupported")
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.transition import Transition

    D, L, n = conf.d_pair, seq_len, 4
    torch.manual_seed(0)
    ref_mod = Transition(D, n=n, implementation=ImplementationType.PYTORCH).to(DEVICE).to(dtype)
    for lin in (ref_mod.expand_a, ref_mod.expand_b, ref_mod.squeeze):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
    lw, lb = ref_mod.ln_in.weight, ref_mod.ln_in.bias
    wa, wb, wsq = ref_mod.expand_a.weight, ref_mod.expand_b.weight, ref_mod.squeeze.weight
    eps = ref_mod.ln_in.eps
    torch.manual_seed(1)
    x0 = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    dy = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    # Residual on both sides: the Transition op includes `+x` and there is no flag to turn it
    # off, so the reference carries it too. It contributes an identity `+dy` to ref_dx, which is
    # exactly what the fused backward's `_finalize_dx` adds -- comparing a residual kernel to a
    # residual-free reference would score that identity as error.
    xr = x0.clone().requires_grad_(True)
    (ref_mod._torch_forward(xr) + xr).backward(dy)
    ref_grads = []
    for tensor in (xr, lw, lb, wa, wb, wsq):
        if tensor is None:
            continue
        gradient = tensor.grad
        if gradient is None:
            raise RuntimeError("reference backward did not produce every requested gradient")
        ref_grads.append(gradient.detach().clone())

    x = x0.clone().requires_grad_(True)
    if implementation == "pytorch":
        out, path = ref_mod._torch_forward(x) + x, "module.reference.torch"
    elif implementation == "triton_transition_fused":
        from miniworld_engine.kernels import triton_transition_fused
        out = triton_transition_fused(x, lw, lb, wa, wb, wsq, n, eps)
        path = "kernels.transition.triton.fused"
    elif implementation == "cute_transition_fused":
        from miniworld_engine.kernels import cute_transition_fused
        out = cute_transition_fused(x, lw, lb, wa, wb, wsq, n, eps) + x
        path = "kernels.transition.cute.fused"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")
    # EVERY leaf, not just x. `torch.autograd.grad(out, leaves, ...)` prunes what no leaf needs,
    # and the two sides prune differently: `TritonTransitionFusedFunction.backward` is ONE autograd
    # node, so it always returns dx, dgamma, dbeta, dWa, dWb, dWs -- autograd cannot reach inside a
    # node -- while the torch reference is a graph whose wgrad GEMMs simply do not run when no
    # weight is a leaf. Asking for `[x]` charged us for six gradients and the reference for one,
    # and reported the difference as our kernel being slow. Same defect, same fix, as
    # bench_kernel_adaln_bwd, where it cost 0.40x -> 0.77x.
    return _bwd_autograd_result(conf, out, [x, lw, lb, wa, wb, wsq], dy, ref_grads, path=path,
                                ref="module.reference.torch", dtype=tname)


def bench_kernel_gemm_epilogue_bwd(conf, seq_len, implementation, fabric):
    """LayerNorm+Linear backward (autograd, backward-only). Rows: pytorch, layernorm_linear_te,
    layernorm_linear_cute. Cosine on dx vs pytorch autograd."""
    if conf.compile:
        raise UnsupportedBenchmark(
            "backward-only autograd uses a prebuilt eager graph; compiled backward is unsupported")
    if conf.precision == FP32_PRECISION and implementation != "pytorch":
        raise UnsupportedBenchmark(
            f"{conf.target}/{implementation}: this benchmark backend supports BF16 inputs only")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    import torch.nn.functional as F

    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    # requires_grad on the weights, and they are leaves below. Without it the torch reference
    # computes NO weight gradient at all, while `LayerNormLinearTEFn.backward` -- one autograd
    # node -- always produces dW, dgamma and dbeta. The bench timed our three extra outputs
    # against a reference excused from them. See bench_kernel_adaln_bwd.
    lw = torch.randn(D, device=DEVICE, dtype=dtype, requires_grad=True)
    lb = torch.randn(D, device=DEVICE, dtype=dtype, requires_grad=True)
    w = (torch.randn(D, D, device=DEVICE, dtype=dtype) * (D**-0.5)).contiguous().requires_grad_(True)
    eps = 1e-5
    torch.manual_seed(1)
    x0 = torch.randn(L * L, D, device=DEVICE, dtype=dtype)
    dy = torch.randn(L * L, D, device=DEVICE, dtype=dtype)
    xr = x0.clone().requires_grad_(True)
    F.linear(F.layer_norm(xr, (D,), lw, lb, eps), w).backward(dy)
    ref_grads = []
    for tensor in (xr, lw, lb, w):
        if tensor is None:
            continue
        gradient = tensor.grad
        if gradient is None:
            raise RuntimeError("reference backward did not produce every requested gradient")
        ref_grads.append(gradient.detach().clone())

    x = x0.clone().requires_grad_(True)
    if implementation == "pytorch":
        out, path = F.linear(F.layer_norm(x, (D,), lw, lb, eps), w), "pytorch.autograd"
    elif implementation == "layernorm_linear_te":
        from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
            layernorm_linear_te_fn,
        )
        out = layernorm_linear_te_fn(x, lw, lb, w, None, eps, length=L)  # x is (L*L, D)
        path = "kernels.layernorm_linear.triton.te_style"
    elif implementation == "layernorm_linear_cute":
        from miniworld_engine.kernels.layernorm_linear.autograd import (
            layernorm_linear_fn,
        )
        out = layernorm_linear_fn(x, lw, lb, w, None, eps, length=L)  # x is (L*L, D)
        path = "kernels.layernorm_linear.autograd.cute"
    else:
        raise UnsupportedBenchmark(f"{conf.target}/{implementation}: unsupported configuration")
    return _bwd_autograd_result(conf, out, [x, lw, lb, w], dy, ref_grads, path=path,
                                ref="pytorch.autograd", dtype=tname)


# A kernel target is named after the kernel FAMILY in `src/miniworld_engine/kernels/registry.csv`
# that it benches -- `triangle_attention`, `bias_only_attention`, `augmented_attention`,
# `fused_ln_mask`, `layernorm`, `adaln`, `conditioned_transition` (whose target benches the
# post-AdaLN `_tail` of the family). The four exceptions bench a fused OP SHAPE that several
# families implement rather than one family -- `dual_gemm_epilogue`, `gemm_epilogue`, `gemm_gate`,
# `transition_b2b` -- and are named after the shape. No abbreviations: the name is spelled the way
# the engine spells it. A `_bwd` suffix marks the backward-only bench of the same op. The key is
# also the directory: `benchmarks/kernels/<target>/` (see `target_dir`), and the bench function is
# `bench_kernel_<target>` (asserted below).
KERNEL_TARGETS = {
    # kernel function-operations (benchmarks/kernels/<op>/): forward
    "dual_gemm_epilogue": bench_kernel_dual_gemm_epilogue,
    "gemm_epilogue": bench_kernel_gemm_epilogue,
    "transition_b2b": bench_kernel_transition_b2b,
    "layernorm": bench_kernel_layernorm,
    "adaln": bench_kernel_adaln,
    "triangle_attention": bench_kernel_triangle_attention,
    "bias_only_attention": bench_kernel_bias_only_attention,
    "augmented_attention": bench_kernel_augmented_attention,
    "fused_ln_mask": bench_kernel_fused_ln_mask,
    "gemm_gate": bench_kernel_gemm_gate,
    "conditioned_transition_tail": bench_kernel_conditioned_transition_tail,
    # kernel function-operations: backward
    "layernorm_bwd": bench_kernel_layernorm_bwd,
    "gemm_gate_bwd": bench_kernel_gemm_gate_bwd,
    "dual_gemm_epilogue_bwd": bench_kernel_dual_gemm_epilogue_bwd,
    "adaln_bwd": bench_kernel_adaln_bwd,
    "transition_b2b_bwd": bench_kernel_transition_b2b_bwd,
    "gemm_epilogue_bwd": bench_kernel_gemm_epilogue_bwd,
}

def bench_module_dit(conf, seq_len, implementation, fabric):
    """Pair-bias DiT block (token by default, atom for the dit_atom target): augmented attention (pair bias) + conditioned transition, both
    residuals explicit. `modules/dit`.

    A block, not a part, because a per-part result does not compose: every kernel here is an
    opaque `custom_op`, so a per-part bench pays its launch overhead once and a block pays it
    once per part -- while `torch.compile` fuses ACROSS parts in the reference and cannot fuse
    across our opaque ops. Both effects only land here.
    """
    from miniworld_engine.modules.dit import DiTBlock

    atom = conf.target == "dit_atom"
    length = seq_len * 8 if atom else seq_len
    d_single = conf.d_single_atom if atom else conf.d_single_token
    d_cond = conf.d_single_atom if atom else conf.d_single
    d_pair = conf.d_pair_atom if atom else conf.d_pair
    n_head = 4 if atom else 16

    spec = triton_miniworld_spec(implementation)
    if spec.impl not in {ImplementationType.PYTORCH, ImplementationType.TRITON,
                         ImplementationType.MINIWORLD}:
        raise UnsupportedBenchmark(f"dit does not implement {implementation!r}")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([
                DiTBlock(d_single=d_single, d_cond=d_cond,
                         d_pair=d_pair, n_head=n_head, implementation=spec.impl)
                for _ in range(conf.n_layers)])

        def forward(self, single, cond, pair, mask=None):
            for layer in self.layers:
                single = layer(single, cond, pair, mask)
            return single

    model = MultiDiT().to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:
        compile_module_for_benchmark(model)
    model = fabric.setup_module(model)

    wants_grad = not is_inference_mode(conf.mode)
    single = torch.randn(conf.n_augment, 1, length, d_single,
                         device=DEVICE, dtype=dtype, requires_grad=wants_grad)
    cond = torch.randn(conf.n_augment, 1, length, d_cond,
                       device=DEVICE, dtype=dtype, requires_grad=wants_grad)
    pair = torch.randn(1, length, length, d_pair, device=DEVICE, dtype=dtype,
                       requires_grad=wants_grad)
    mask = torch.rand(1, length, device=DEVICE) > conf.mask_prob
    dy = torch.randn_like(single)

    def inference_step():
        with torch.no_grad():
            return model(single, cond, pair, mask)

    def training_step() -> torch.Tensor:
        y = model(single, cond, pair, mask)
        fabric.backward(y, dy)
        return y

    return measured_result(
        conf=conf,
        func=inference_step if is_inference_mode(conf.mode) else training_step,
        grad_to_none=[single, cond, pair, *list(model.parameters())],
        params=list(model.parameters()),
        is_train=wants_grad,
        input_dtype=str(dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=("modules.dit.DiTBlock" if spec.impl != ImplementationType.PYTORCH
                        else "module.reference.torch"),
        reference="module.reference.torch",
    )


def bench_module_dit_atom(conf, seq_len, implementation, fabric):
    """Full pair-bias atom DiT, at 8 * seq_len and atom widths; shares the token algorithm."""
    return bench_module_dit(conf, seq_len, implementation, fabric)


def bench_swa_component_audit(conf, seq_len, implementation, fabric):
    """Paired SWA sub-operation training and one-at-a-time full-block substitutions.

    Uses the standard compiler, correctness checks, timer, and CSV provenance. All
    substitutions are local benchmark subclasses; production dispatch is untouched.
    """
    import copy

    import torch.nn.functional as F

    from miniworld_engine import kernels, ops
    from miniworld_engine.kernels.rope.interface import qk_norm_rope_3d
    from miniworld_engine.modules.swa_atom_attention.module import apply_rotary_emb_3d
    from miniworld_engine.modules.swa_dit import SWADiTBlock

    names = {"modulation", "rope", "swiglu", "residual", "sigmoid_gate"}
    requested = set(conf.swa_kernels if conf.swa_kernels is not None else names)
    if requested - names:
        raise ValueError(f"unknown SWA kernels: {requested - names}")
    spec = triton_miniworld_spec(implementation)
    if spec.impl not in {ImplementationType.PYTORCH, ImplementationType.MINIWORLD, ImplementationType.TRITON}:
        raise UnsupportedBenchmark("SWA audit supports pytorch and miniworld")
    enabled = requested if spec.impl != ImplementationType.PYTORCH else set()
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16
    n, s, width, heads = conf.n_augment, seq_len * 8, conf.d_single_atom, 4
    component = conf.swa_component
    if conf.n_layers != 1 or not conf.compile or conf.mode != "training":
        raise ValueError("SWA audit requires one block, compile=true, mode=training")

    def torch_rope(q, k, cos, sin):
        eps = torch.finfo(torch.float32).eps
        q = F.rms_norm(q.float(), (q.shape[-1],), eps=eps).to(q.dtype)
        k = F.rms_norm(k.float(), (k.shape[-1],), eps=eps).to(k.dtype)
        return apply_rotary_emb_3d(q, cos, sin), apply_rotary_emb_3d(k, cos, sin)

    class AuditAttention(SWA3DRoPEAttention):
        audit_kernels: frozenset[str]

        def forward(self, x, ap):
            n, s = x.shape[:2]
            cos, sin, used, cu, maximum, valid = ap
            qkv = self.Wqkv(x).view(n, s, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)
            fn = qk_norm_rope_3d if "rope" in self.audit_kernels else torch_rope
            q, k = fn(q, k, cos, sin)
            out = self._flash_window(q, k, v, cu, used, maximum, valid, n, s).reshape(n, s, -1)
            gate = self.gate_proj(x)
            out = kernels.sigmoid_gate_fused(gate, out) if "sigmoid_gate" in self.audit_kernels else out * torch.sigmoid(gate)
            return self.out_proj(out)

    class AuditBlock(SWADiTBlock):
        audit_kernels: frozenset[str]

        def forward(self, x, cond, ap):
            if "modulation" in self.audit_kernels:
                activated = self.adaln_modulation[0](cond)
                projection = self.adaln_modulation[1]
                assert isinstance(projection, nn.Linear)
                sh_a, sc_a, g_a, sh_f, sc_f, g_f = projection.weight.chunk(6, dim=0)
                a, ga = ops.rms_norm_modulation(x, activated, sc_a, sh_a, g_a, eps=torch.finfo(torch.float32).eps)
            else:
                sha, sca, ga, shf, scf, gf = self.adaln_modulation(cond).chunk(6, dim=-1)
                a = self.attn_norm(x) * (1 + sca) + sha
            branch = self.attn(a, ap)
            x = ops.gated_residual(x, ga, branch) if "residual" in self.audit_kernels else x + ga * branch
            if "modulation" in self.audit_kernels:
                f, gf = ops.rms_norm_modulation(x, activated, sc_f, sh_f, g_f, eps=torch.finfo(torch.float32).eps)
            else:
                f = self.ffn_norm(x) * (1 + scf) + shf
            branch = self.ffn(f)
            return ops.gated_residual(x, gf, branch) if "residual" in self.audit_kernels else x + gf * branch

    class Component(nn.Module):
        audit_kernels: frozenset[str]

        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, *inputs):
            engine = component in self.audit_kernels
            if component == "modulation":
                x, cond = inputs
                activated = self.base.adaln_modulation[0](cond)
                shift, scale, gate = self.base.adaln_modulation[1].weight.chunk(6, dim=0)[:3]
                if engine:
                    return ops.rms_norm_modulation(x, activated, scale, shift, gate, eps=torch.finfo(torch.float32).eps)
                return (self.base.attn_norm(x) * (1 + F.linear(activated, scale)) + F.linear(activated, shift), F.linear(activated, gate))
            if component == "rope":
                return (qk_norm_rope_3d if engine else torch_rope)(*inputs)
            if component == "swiglu":
                return self.base.ffn(inputs[0])
            if component == "residual":
                x, gate, branch = inputs
                return ops.gated_residual(x, gate, branch) if engine else x + gate * branch
            if component == "sigmoid_gate":
                gate, branch = inputs
                return kernels.sigmoid_gate_fused(gate, branch) if engine else torch.sigmoid(gate) * branch
            raise ValueError(component)

    base = AuditBlock(width, width, heads).to(device=DEVICE, dtype=dtype)
    base.attn.__class__ = AuditAttention
    if conf.swa_active_gates:
        with torch.no_grad():
            projection = base.adaln_modulation[1]
            assert isinstance(projection, nn.Linear)
            projection.weight.normal_(std=0.01)
    model = base if component == "block" else Component(base)

    def configure(model, selected):
        for module in model.modules():
            module.audit_kernels = frozenset(selected)
            if hasattr(module, "implementation"):
                module.implementation = ImplementationType.PYTORCH
        base = model if isinstance(model, AuditBlock) else model.base
        base.ffn.implementation = ImplementationType.MINIWORLD if "swiglu" in selected else ImplementationType.PYTORCH

    configure(model, enabled)
    reference = copy.deepcopy(model)
    configure(reference, set())
    if component == "block":
        # Independent production PyTorch forward, not a second copy of the audit formula.
        reference.__class__ = SWADiTBlock
        reference.attn.__class__ = SWA3DRoPEAttention
    x = torch.randn(n, s, width, device=DEVICE, dtype=dtype, requires_grad=True)
    cond = torch.randn_like(x, requires_grad=True)
    angles = torch.randn(1, s, width // heads // 2, device=DEVICE)
    cos, sin = angles.cos(), angles.sin()
    lengths = (torch.rand(n, s, device=DEVICE) > conf.mask_prob).sum(-1)
    valid = torch.arange(s, device=DEVICE)[None] < lengths[:, None]
    ap = build_attention_params(cos, sin, valid, num_aug=n)
    cos, sin = ap[:2]
    if component == "block":
        inputs = (x, cond, ap)
    elif component == "modulation":
        inputs = (x, cond)
    elif component == "rope":
        # Preserve production Q/K's strided views into one packed QKV tensor.
        packed = torch.randn(n, s, 3, heads, width // heads, device=DEVICE, dtype=dtype)
        q, k, _ = packed.unbind(2)
        inputs = (q.detach().requires_grad_(), k.detach().requires_grad_(), cos, sin)
    elif component == "swiglu":
        inputs = (x,)
    elif component == "residual":
        inputs = (x, cond, torch.randn_like(x, requires_grad=True))
    else:
        inputs = (x, cond)
    # Only differentiable tensor inputs need independent gradient storage.
    ref_inputs = tuple(t.detach().clone().requires_grad_() if isinstance(t, torch.Tensor) and t.requires_grad else t for t in inputs)
    leaves = [t for t in inputs if isinstance(t, torch.Tensor) and t.requires_grad]
    ref_leaves = [t for t in ref_inputs if isinstance(t, torch.Tensor) and t.requires_grad]
    model.train()
    reference.train()
    compile_module_for_benchmark(model, fullgraph=True)
    compile_module_for_benchmark(reference, fullgraph=True)
    y, ry = model(*inputs), reference(*ref_inputs)
    outputs = y if isinstance(y, tuple) else (y,)
    references = ry if isinstance(ry, tuple) else (ry,)
    grads = tuple(torch.randn_like(t) for t in outputs)
    torch.autograd.backward(outputs, grads)
    torch.autograd.backward(references, grads)
    output_metrics, grad_metrics = [], []
    for a, b in zip(outputs, references, strict=True):
        torch.testing.assert_close(a, b, atol=.04 if dtype == torch.bfloat16 else .005, rtol=.04 if dtype == torch.bfloat16 else .01)
        output_metrics.append(tensor_metrics(a, b))
    for a, b in zip([*leaves, *model.parameters()], [*ref_leaves, *reference.parameters()], strict=True):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
            continue
        metrics = tensor_metrics(a.grad, b.grad)
        assert metrics[1] < (.03 if dtype == torch.bfloat16 else .005), metrics
        assert bool(torch.isfinite(a.grad).all())
        grad_metrics.append(metrics)
    acc = {"output_max_abs": max(t[0] for t in output_metrics), "output_rel_frob": max(t[1] for t in output_metrics),
           "output_cosine": min(t[2] for t in output_metrics), "grad_max_abs": max(t[0] for t in grad_metrics),
           "grad_rel_frob": max(t[1] for t in grad_metrics), "grad_cosine": min(t[2] for t in grad_metrics)}
    del reference, ref_inputs, ref_leaves, y, ry, outputs, references
    for t in [*leaves, *model.parameters()]:
        t.grad = None

    def step():
        out = model(*inputs)
        torch.autograd.backward(out if isinstance(out, tuple) else (out,), grads)
        return out

    result = measured_result(conf=conf, func=step, grad_to_none=[*leaves, *model.parameters()],
                             params=list(model.parameters()), is_train=True, input_dtype=str(dtype).replace("torch.", ""),
                             parameter_dtype=parameter_dtype_of(model),
                             execution_path=f"swa_audit:{component}:kernels={','.join(sorted(enabled))}:active_gates={conf.swa_active_gates}",
                             reference="fullgraph_pytorch_same_weights_inputs")
    return result._replace(**acc)


def bench_module_swa_dit(conf, seq_len, implementation, fabric):
    """ESMFold2 atom DiT: RMSNorm + adaLN-Zero gates, windowed 3D-RoPE attention, SwiGLU.
    `modules/swa_dit`. Runs at the atom length (`seq_len * 8`), like the swa_atom_attention
    kernel bench, and reports NaN without a flash backend for the same reason.
    """
    if conf.swa_component != "block" or conf.swa_kernels is not None or conf.swa_active_gates:
        return bench_swa_component_audit(conf, seq_len, implementation, fabric)
    from miniworld_engine.modules.swa_atom_attention.module import (
        build_attention_params,
    )
    from miniworld_engine.modules.swa_dit import SWADiTBlock

    spec = triton_miniworld_spec(implementation)
    if spec.impl not in {ImplementationType.PYTORCH, ImplementationType.TRITON,
                         ImplementationType.MINIWORLD}:
        raise UnsupportedBenchmark(f"swa_dit does not implement {implementation!r}")
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16
    n_head = 4

    class MultiSWADiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([
                SWADiTBlock(d_atom=conf.d_single_atom, d_cond=conf.d_single_atom,
                            n_head=n_head, implementation=spec.impl)
                for _ in range(conf.n_layers)])

        def forward(self, x, cond, ap):
            for layer in self.layers:
                x = layer(x, cond, ap)
            return x

    model = MultiSWADiT().to(device=DEVICE, dtype=dtype)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:
        compile_module_for_benchmark(model, fullgraph=True)
    model = fabric.setup_module(model)

    atom_len, n = seq_len * 8, conf.n_augment
    half = conf.d_single_atom // n_head // 2
    wants_grad = not is_inference_mode(conf.mode)
    x = torch.randn(n, atom_len, conf.d_single_atom, device=DEVICE, dtype=dtype,
                    requires_grad=wants_grad)
    cond = torch.randn(n, atom_len, conf.d_single_atom, device=DEVICE, dtype=dtype,
                       requires_grad=wants_grad)
    cos = torch.randn(1, atom_len, half, device=DEVICE, dtype=torch.float32)
    sin = torch.randn(1, atom_len, half, device=DEVICE, dtype=torch.float32)
    valid_lengths = (torch.rand(n, atom_len, device=DEVICE) > conf.mask_prob).sum(-1)
    valid = torch.arange(atom_len, device=DEVICE)[None, :] < valid_lengths[:, None]
    ap = build_attention_params(cos, sin, valid, num_aug=n)
    dy = torch.randn_like(x)

    def inference_step():
        with torch.no_grad():
            return model(x, cond, ap)

    def training_step() -> torch.Tensor:
        y = model(x, cond, ap)
        fabric.backward(y, dy)
        return y

    from miniworld_engine.modules.swa_atom_attention.module import _flash_backend

    backend = _flash_backend(DEVICE)
    return measured_result(
        conf=conf,
        func=inference_step if is_inference_mode(conf.mode) else training_step,
        grad_to_none=[x, cond, *list(model.parameters())],
        params=list(model.parameters()),
        is_train=wants_grad,
        input_dtype=str(dtype).replace("torch.", ""),
        parameter_dtype=parameter_dtype_of(model),
        execution_path=f"modules.swa_dit.SWADiTBlock[{backend or 'unavailable'}]",
        reference="module.reference.torch",
    )

# A module target is named after the production module it benches, spelled as the engine spells
# it. The key is the directory `benchmarks/modules/<target>/` and the function is
# `bench_module_<target>` (asserted below). Names may repeat KERNEL_TARGETS keys on purpose: the
# module and the kernel it is built out of are two different benches of the same op, told apart
# by `level`, not by mangling one of the two names.
MODULE_TARGETS = {
    "triangle_multiplication": bench_module_triangle_multiplication,
    "swa_atom_attention": bench_module_swa_atom_attention,
    "triangle_multiplication_bidirectional": bench_module_triangle_multiplication_bidirectional,
    "triangle_attention": bench_module_triangle_attention,
    "transition": bench_module_transition,
    "conditioned_transition": bench_module_conditioned_transition,
    "adaptive_layernorm": bench_module_adaptive_layernorm,
    "augmented_attention_token": bench_module_augmented_attention_token,
    "augmented_attention_atom": bench_module_augmented_attention_atom,
    "dit": bench_module_dit,
    "dit_atom": bench_module_dit_atom,
    "swa_dit": bench_module_swa_dit,
}

# The naming rules above are checked here, not just written down: a target whose function is named
# off-convention (or a kernel target parked in MODULE_TARGETS) fails at import, which is the only
# way a convention survives the next addition. NOT asserted: that the two key sets are disjoint --
# a shared name is the point of having a level. `triangle_attention` is already both a kernel
# target and a module target, and `transition`/`layernorm`/... become both the day someone adds
# the missing side.
for _name, _fn in KERNEL_TARGETS.items():
    assert _fn.__name__ == f"bench_kernel_{_name}", (
        f"kernel target {_name!r} must map to bench_kernel_{_name}, got {_fn.__name__}"
    )
for _name, _fn in MODULE_TARGETS.items():
    assert _fn.__name__ == f"bench_module_{_name}", (
        f"module target {_name!r} must map to bench_module_{_name}, got {_fn.__name__}"
    )
del _name, _fn


def targets_for(level: str) -> dict[str, Callable[..., Any]]:
    """The target->bench-function table of one level. The two levels are separate namespaces."""
    # cast, not an annotation on the tables themselves: the import-time convention check below
    # reads each entry's `__name__`, which a `Callable` does not have.
    return cast("dict[str, Callable[..., Any]]",
                KERNEL_TARGETS if level == "kernel" else MODULE_TARGETS)


def target_impls(level: str, target: str) -> tuple[str, ...]:
    """Every implementation ``target`` (of ``level``) knows how to bench.

    Read out of this module's own source rather than kept as a second list: the names live in each
    bench function's ``implementation == "..."`` chain, and any hand-maintained copy drifts the
    moment an implementation is added, renamed or dropped -- which is exactly the drift that let a
    sweep call itself a kernel test while exercising one implementation per target.

    Module-level targets do not use that chain; they parse the name into an ``ImplementationType``,
    so their set is the enum plus the two aliases ``module_miniworld_spec`` understands.
    """
    import ast as _ast
    import inspect as _inspect

    # A module enum is not a support matrix: advertising all enum values caused
    # unsupported labels and duplicate reference implementations in ordinary `all` runs.
    if level == "module":
        supported = {
            "triangle_multiplication": ("pytorch", "triton", "miniworld", "cuequivariance", "cute", "dtv1"),
            "triangle_multiplication_bidirectional": ("pytorch", "triton", "miniworld", "cuequivariance", "cute", "dtv1"),
            "triangle_attention": ("pytorch", "triton", "miniworld", "cuequivariance"),
            "transition": ("pytorch", "triton", "miniworld", "cute"),
            "conditioned_transition": ("pytorch", "triton", "miniworld"),
            "adaptive_layernorm": ("pytorch", "triton", "miniworld"),
            "augmented_attention_token": ("pytorch", "triton", "miniworld"),
            "augmented_attention_atom": ("pytorch", "triton", "miniworld"),
            "swa_atom_attention": ("pytorch", "miniworld"),
            "dit": ("pytorch", "triton", "miniworld"),
            "dit_atom": ("pytorch", "triton", "miniworld"),
            "swa_dit": ("pytorch", "triton", "miniworld"),
        }
        if set(supported) != set(MODULE_TARGETS):
            raise RuntimeError("module benchmark implementation matrix is incomplete")
        return supported.get(target, ())
    fn = targets_for(level).get(target)
    if fn is None:
        return ()
    tree = _ast.parse(_inspect.getsource(fn))
    out: list[str] = []
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Compare) and isinstance(node.left, _ast.Name)
                and node.left.id == "implementation"):
            continue
        for cmp in node.comparators:
            vals = ([cmp] if isinstance(cmp, _ast.Constant)
                    else list(cmp.elts) if isinstance(cmp, (_ast.Tuple, _ast.List, _ast.Set))
                    else [])
            for v in vals:
                if isinstance(v, _ast.Constant) and isinstance(v.value, str) and v.value not in out:
                    out.append(v.value)
    if out:
        return tuple(out)
    # Module-level targets have no `implementation == "..."` chain; their set is the enum plus
    # `miniworld`. Labels that only some targets define are read from the source (their constant
    # is named in the function that supports them), so the list cannot drift: offering a label
    # everywhere once benched the same path twice under two names.
    src = _inspect.getsource(fn)
    names = [*[m.value for m in ImplementationType], MINIWORLD_IMPL]
    if "BIAS_ONLY_V_IMPL" in src:
        names.append(BIAS_ONLY_V_IMPL)
    return tuple(names)


#: (target, implementation) -> the minimum GPU architecture that implementation needs.
#:
#: `bench_kernel all` offers every implementation a target defines to whatever card is running, so
#: an A100 sweep spent 99 of its 307 failed rows launching H100/B200 kernels that answer with
#: `AssertionError: SM90 (H100) only`, `NotImplementedError: Gemm Sm80 is not implemented yet` or
#: `OpError: expects arch to be sm_90a`. That is a declaration the harness already has for kernels
#: -- `registry.csv`'s `arch`, gated by `run_all.meets_arch` -- and did not have for bench
#: implementations. Neither derivation works: an implementation's `path=` string names a function
#: or a package as often as a file (22 of 34 map), and its imports go through the flat
#: `miniworld_engine.kernels` re-export (16 of 32). So it is declared, and a test holds every key
#: against the real target and implementation names so it cannot drift into naming nothing.
#:
#: Every entry below is the message that card actually produced, not a guess. An implementation
#: absent from this table is offered everywhere, which is the behaviour that was there before.
IMPL_MIN_ARCH: dict[tuple[str, str], str] = {
    # AssertionError: first version targets SM90 (H100)
    ("adaln", "adaln_lnfold"): "sm90",
    ("gemm_epilogue", "layernorm_linear_cute"): "sm90",
    # AssertionError: SM90 (H100) only
    ("gemm_epilogue", "layernorm_linear_cute_fused"): "sm90",
    ("transition", "cute"): "sm90",
    ("transition_b2b", "cute_transition_fused"): "sm90",
    # OpError: expects arch to be sm_90a, but got sm_80
    ("gemm_gate", "tm2_cute"): "sm90",
    # NotImplementedError: Gemm Sm80 is not implemented yet
    ("dual_gemm_epilogue", "tm1_cute"): "sm90",
    ("dual_gemm_epilogue", "trimul_inproj_cute"): "sm90",
    # Both module variants reach the CuTe GEMM path; Ampere reports
    # NotImplementedError: Gemm Sm80 is not implemented yet.
    ("triangle_multiplication", "cute"): "sm90",
    ("triangle_multiplication_bidirectional", "cute"): "sm90",
}


def min_arch_for(target: str, implementation: str) -> str | None:
    """The declared minimum arch for this (target, implementation), or None if unrestricted.

    The `_bwd` fallback lives here rather than in `runnable_here` so that every caller resolves
    the entry the same way. It did not, and the skip message indexed IMPL_MIN_ARCH directly:
    `gemm_epilogue_bwd` gated correctly through the fallback and then died formatting the line
    that says so -- KeyError, hydra aborted the target, and the run produced no rows for it at
    all, which is worse than the noise the gate was added to remove.
    """
    want = IMPL_MIN_ARCH.get((target, implementation))
    if want is None and target.endswith("_bwd"):
        want = IMPL_MIN_ARCH.get((target.removesuffix("_bwd"), implementation))
    return want


def runnable_here(target: str, implementation: str) -> bool:
    """Can this card run this implementation? Unlisted implementations always can.

    A `*_bwd` target falls back to its forward sibling's entry. `transition_b2b_bwd` benches the
    backward of the kernels `transition_b2b` benches, so an implementation the card cannot run one
    way it cannot run the other -- and keying both separately means every future entry has to be
    written twice, which is exactly how `cute_transition_fused` stayed ungated on the backward
    target after the forward one was fixed.
    """
    from miniworld_engine.autotune.run_all import meets_arch

    want = min_arch_for(target, implementation)
    return True if want is None else meets_arch({"arch": want})


def expand_implementations(level: str, target: str, requested: list[str]) -> list[str]:
    """Resolve the sentinel ``all`` to every implementation of ``target``."""
    if [r.strip().lower() for r in requested] != ["all"]:
        return requested
    impls = target_impls(level, target)
    if not impls:
        msg = f"implementations=all: no implementations found for {level} target {target!r}"
        raise ValueError(msg)
    # `all` means "every implementation this CARD can run". One it cannot is not a result: it
    # occupies a row, reports an architecture assertion as a bench failure, and buries the
    # failures that are defects. Asking for one BY NAME still runs it and still reports what the
    # kernel says -- the filter is on the sentinel, not on the implementation.
    here = [i for i in impls if runnable_here(target, i)]
    skipped = [i for i in impls if i not in here]
    if skipped:
        print(f"=== {target}: skipping {len(skipped)} implementation(s) this card cannot run: "
              f"{', '.join(f'{i} (needs {min_arch_for(target, i)})' for i in skipped)}",
              flush=True)
    return here


_KERNELS_ROOT = _REPO_ROOT / "benchmarks" / "kernels"
_MODULES_ROOT = _REPO_ROOT / "benchmarks" / "modules"


def target_dir(level: str, target: str) -> Path:
    """Where ``target``'s configs/artifacts/results live.

    The mapping is the identity, not a lookup table: `level` IS the directory
    (`benchmarks/kernels/` or `benchmarks/modules/`) and `target` IS the folder name under it,
    for every target with no exceptions. The hand-written dict this replaced is what let two
    targets (`augmented_attention_token`/`_atom`) quietly share one folder and disambiguate their
    tables by filename prefix instead.
    """
    root = _KERNELS_ROOT if level == "kernel" else _MODULES_ROOT
    return root / target


# Triton autotuner objects live in the per-op `triton/main.py` of each kernel. Keyed by MODULE
# target: a module composes several kernel families, so its summary has to gather all of them. A
# kernel target benches one op directly and needs no such list -- and must not borrow a module's,
# now that the two namespaces share names (`triangle_attention` is both).
AUTOTUNE_MODULES = {
    "triangle_multiplication": [
        "miniworld_engine.modules.triangle_multiplication.baseline_dtv1",
        "miniworld_engine.kernels.layernorm.triton.main",
        "miniworld_engine.kernels.tm1.triton.main",
        "miniworld_engine.kernels.tm2.triton.main",
    ],
    "triangle_multiplication_bidirectional": [
        "miniworld_engine.modules.triangle_multiplication.baseline_dtv1",
        "miniworld_engine.kernels.layernorm.triton.main",
        "miniworld_engine.kernels.tm1.triton.main",
        "miniworld_engine.kernels.tm2.triton.main",
        "miniworld_engine.kernels.trimul_inproj.triton.back_fused",
    ],
    "triangle_attention": [
        "miniworld_engine.kernels.layernorm.triton.main",
        "miniworld_engine.kernels.bias_only_attention.triton.gate_out",
        "miniworld_engine.kernels.triangle_attention.triton.main",
    ],
    "transition": [
        "miniworld_engine.kernels.layernorm.triton.main",
        "miniworld_engine.kernels.transition.triton.main",
        "miniworld_engine.kernels.transition.triton.fused",
    ],
    "conditioned_transition": [
        "miniworld_engine.kernels.conditioned_transition.triton.inference",
        "miniworld_engine.kernels.conditioned_transition.triton.composed",
        "miniworld_engine.kernels.conditioned_transition.triton.training",
    ],
    "adaptive_layernorm": [
        "miniworld_engine.kernels.adaln.triton.main",
    ],
    "augmented_attention_token": [
        "miniworld_engine.kernels.layernorm.triton.main",
        "miniworld_engine.kernels.augmented_attention.triton.main",
    ],
    "augmented_attention_atom": [
        "miniworld_engine.kernels.layernorm.triton.main",
        "miniworld_engine.kernels.augmented_attention.triton.main",
    ],
}


def get_autotuners(level: str, target: str) -> dict[str, Autotuner]:
    if level != "module":
        return {}
    autotuners = {}
    for module_name in AUTOTUNE_MODULES.get(target, []):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr_name, attr_value in vars(module).items():
            if isinstance(attr_value, Autotuner):
                autotuners[f"{module.__name__}.{attr_name}"] = attr_value
    return autotuners


def format_autotune_key(key: tuple) -> str:
    return ", ".join(str(value).replace("torch.", "") for value in key)


def format_autotune_config(config: triton.Config) -> str:
    parts = [f"{name}={value}" for name, value in sorted(config.kwargs.items())]
    parts.append(f"num_warps={config.num_warps}")
    if getattr(config, "num_stages", None) is not None:
        parts.append(f"num_stages={config.num_stages}")
    if getattr(config, "num_ctas", 1) != 1:
        parts.append(f"num_ctas={config.num_ctas}")
    if getattr(config, "maxnreg", None) is not None:
        parts.append(f"maxnreg={config.maxnreg}")
    return ", ".join(parts)


def capture_autotune_state(
    level: str,
    target: str,
    cache_records: dict[str, dict[tuple, triton.Config]],
    single_config_records: dict[str, triton.Config],
    seen_autotuners: set[str],
) -> None:
    for autotuner_name, autotuner in sorted(get_autotuners(level, target).items()):
        seen_autotuners.add(autotuner_name)
        cache = getattr(autotuner, "cache", None) or {}
        if cache:
            bucket = cache_records.setdefault(autotuner_name, {})
            bucket.update(cache)
            continue

        configs = getattr(autotuner, "configs", [])
        if len(configs) == 1:
            single_config_records[autotuner_name] = configs[0]


def build_autotune_summary(
    level: str,
    target: str,
    cache_records: dict[str, dict[tuple, triton.Config]] | None = None,
    single_config_records: dict[str, triton.Config] | None = None,
    seen_autotuners: set[str] | None = None,
) -> str | None:
    cache_records = cache_records or {}
    single_config_records = single_config_records or {}
    seen_autotuners = seen_autotuners or set(get_autotuners(level, target))

    sections = []
    current_autotuners = get_autotuners(level, target)
    for autotuner_name in sorted(seen_autotuners):
        lines = [autotuner_name]
        configs = getattr(current_autotuners.get(autotuner_name), "configs", [])
        if configs:
            lines.append(f"  candidate_configs={len(configs)}")
            for index, config in enumerate(configs, start=1):
                lines.append(f"    candidate[{index}] {format_autotune_config(config)}")
        cache = cache_records.get(autotuner_name, {})
        if cache:
            for key, config in sorted(
                cache.items(),
                key=lambda item: tuple(str(value) for value in item[0]),
            ):
                lines.append(
                    f"  key=({format_autotune_key(key)}) -> "
                    f"{format_autotune_config(config)}",
                )
        elif autotuner_name in single_config_records:
            lines.append(
                "  single_config -> "
                f"{format_autotune_config(single_config_records[autotuner_name])}",
            )
        else:
            lines.append("  no cache entries captured")
        sections.append("\n".join(lines))

    if not sections:
        return None
    return "Triton autotune summary\n" + "\n\n".join(sections)


def autotune_summary_path(results_dir: Path, run_name: str) -> Path:
    return results_dir / f"{run_name}_autotune_summary.txt"


def _compile_wrap_now() -> str:
    """``settings.compile_wrap`` as it stands, which is what the kernels registered under.

    Read live rather than taken from a bench flag: the value is consumed at kernel-IMPORT time
    (see kernels._compile), so by the time a row is written it is a fact about this process, not
    a request that might not have been honoured.
    """
    from miniworld_engine import settings as _s

    return _s.current().compile_wrap


CSV_FIELDS = [
    "run_name",
    "measurement_schema",
    "run_id",
    "config_hash",
    "source_hash",
    "target_kind",
    "target",
    "device",
    "torch_version",
    "cuda_version",
    "metric",
    "unit",
    "mode",
    "compiled",
    "compile_requested",
    "cudagraph",
    "cudagraph_requested",
    "compile_scope",
    "compiled_graphs",
    "measurement_scope",
    "mode_requested",
    "input_shapes",
    "execution_validation",
    # WHICH compile_wrap produced the row. Every table committed before this column existed was
    # measured with "disable" -- a graph break at every kernel entry -- and nothing said so, which
    # is the same defect the `compiled` column had: a number whose regime is not recorded cannot
    # be compared with a number measured in another one. Read from the live settings, not from a
    # flag, because it is settings.compile_wrap at kernel-IMPORT time that decided what ran.
    "compile_wrap",
    "precision",
    "allow_tf32",
    "sweep_axis",
    "implementation",
    "implementation_type",
    "ln_implementation",
    "input_dtype",
    "parameter_dtype",
    "execution_path",
    "reference",
    "output_max_abs",
    "output_rel_frob",
    "output_cosine",
    "grad_max_abs",
    "grad_rel_frob",
    "grad_cosine",
    "n_layers",
    "n_layers_requested",
    "trimul_direction",
    "n_augment",
    "mask_prob",
    "dropout",
    "seq_len",
    "tokens",
    "batch_size",
    "d_pair",
    "d_single",
    "d_single_token",
    "d_single_atom",
    "d_pair_atom",
    "status",
    "error",
    "value",
]


def result_unit(metric: str) -> str:
    return "ms" if metric == "time" else "MiB"


def ascii_safe(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def fabric_precision(precision: int | str) -> str:
    return "32-true" if precision == FP32_PRECISION else str(precision)


def csv_row(
    *,
    conf: BenchConfig,
    run_name: str,
    device_name: str,
    seq_len: int,
    implementation: str,
    result: BenchResult | None,
    status: str = "ok",
    error: str = "",
) -> dict[str, str | int | float | bool | None]:
    if implementation == DTV1_IMPL:
        spec = None
    else:
        try:
            spec = parse_implementation_spec(implementation)
        except ValueError:
            spec = None  # kernel-bench variant label, not a module ImplementationType
    import math
    if status == "ok" and (result is None or not math.isfinite(result.value)):
        status, error = "failed", "benchmark returned no finite measurement"
        result = None
    if status == "ok" and result is not None and (
        result.compiled is None or result.cudagraph not in {"disabled", "manual", "graphed"}
        or result.measurement_scope not in {"forward", "backward", "forward_backward"}
        or result.compiled != (result.compiled_graphs > 0)
        or (result.compiled and not result.compile_scope)
    ):
        status, error = "failed", "measurement lacks observed execution evidence"
        result = None
    implementation_type = spec.impl.value if spec is not None else implementation
    if implementation == MINIWORLD_IMPL:
        implementation_type = MINIWORLD_IMPL
    return {
        "run_name": run_name,
        "measurement_schema": 2,
        "run_id": getattr(conf, "_benchmark_provenance", {}).get("run_id", ""),
        "config_hash": getattr(conf, "_benchmark_provenance", {}).get("config_hash", ""),
        "source_hash": getattr(conf, "_benchmark_provenance", {}).get("source_hash", ""),
        "target_kind": conf.level,
        "target": conf.target,
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "metric": conf.metric,
        "unit": result_unit(conf.metric),
        "mode": ("backward" if conf.target.endswith("_bwd") else "inference")
                if conf.level == "kernel" else mode_label(conf.mode),
        "mode_requested": mode_label(conf.mode),
        "compiled": actual_compiled_flag(result),
        "compile_requested": conf.compile,
        "cudagraph": "" if result is None else result.cudagraph,
        "cudagraph_requested": conf.cudagraph,
        "compile_scope": "" if result is None else result.compile_scope,
        "compiled_graphs": 0 if result is None else result.compiled_graphs,
        "measurement_scope": "" if result is None else result.measurement_scope,
        "input_shapes": "" if result is None else result.input_shapes,
        "execution_validation": "" if result is None else result.execution_validation,
        "compile_wrap": _compile_wrap_now(),
        "precision": conf.precision,
        "allow_tf32": conf.allow_tf32,
        "sweep_axis": conf.sweep_axis,
        "implementation": implementation,
        "implementation_type": implementation_type,
        "ln_implementation": "" if spec is None or spec.ln_impl is None else spec.ln_impl.value,
        "input_dtype": "" if result is None else result.input_dtype,
        "parameter_dtype": "" if result is None else result.parameter_dtype,
        "execution_path": "" if result is None else result.execution_path,
        "reference": "" if result is None else result.reference,
        "output_max_abs": None if result is None else result.output_max_abs,
        "output_rel_frob": None if result is None else result.output_rel_frob,
        "output_cosine": None if result is None else result.output_cosine,
        "grad_max_abs": None if result is None else result.grad_max_abs,
        "grad_rel_frob": None if result is None else result.grad_rel_frob,
        "grad_cosine": None if result is None else result.grad_cosine,
        "n_layers": conf.n_layers if conf.level == "module" else 1,
        "n_layers_requested": conf.n_layers,
        "trimul_direction": ("bidirectional" if conf.target == "triangle_multiplication_bidirectional"
                             else conf.trimul_direction if conf.target == "triangle_multiplication" else ""),
        "n_augment": conf.n_augment,
        "mask_prob": conf.mask_prob,
        "dropout": conf.dropout,
        "seq_len": seq_len,
        # Different targets flatten/augment different axes. input_shapes records actual tensors.
        "tokens": None,
        "batch_size": None,
        "d_pair": conf.d_pair,
        "d_single": conf.d_single,
        "d_single_token": conf.d_single_token,
        "d_single_atom": conf.d_single_atom,
        "d_pair_atom": conf.d_pair_atom,
        "status": status,
        "error": error,
        "value": None if result is None else result.value,
    }


#: The config directory used when the command line names no target -- `--help`, or a bare
#: `python bench.py`, both of which must still load something for hydra to start. It is the module
#: base every target's config was copied from.
_FALLBACK_CONFIG_DIR = "../modules/triangle_multiplication/configs"


def _target_config_path() -> str:
    """Hydra's config directory for the target named on the command line.

    ``@hydra.main``'s ``config_path`` is a decorator argument, so it used to be one module's
    directory for every run: `benchmarks/modules/triangle_multiplication/configs/bench.yaml` was
    the base of a kernel bench, of an atom bench, of everything. The other 25 targets' configs
    were files nothing loaded, and they did not agree with what ran --
    `augmented_attention_atom/configs/bench.yaml` declares a 128-384 ladder and the target was
    swept at 384-1024, which is token-scale lengths on an atom-scale op.

    The path is a pure function of `level` and `target`, and both are plain overrides, so it can
    be read off argv before hydra starts. Returned relative to THIS file, which is what hydra
    requires. A named target whose directory has no `bench.yaml` is an error, not a fall back to
    somebody else's config -- silently loading another target's ladders is the failure being
    removed here.
    """
    picked: dict[str, str] = {}
    for arg in sys.argv[1:]:
        key, sep, value = arg.partition("=")
        if sep and key in ("level", "target"):
            picked[key] = value
    level, target = picked.get("level"), picked.get("target")
    if level is None and target is None:
        return _FALLBACK_CONFIG_DIR
    if level is None or target is None:
        # One without the other picks a config by accident: `target=layernorm` alone would load
        # the module base and then look `layernorm` up in MODULE_TARGETS, which is a KeyError
        # several hundred lines later.
        msg = (f"target and level go together: got level={level!r} target={target!r}. `level` is "
               f"`kernel` or `module`, and the pair names benchmarks/<level>s/<target>/.")
        raise ValueError(msg)
    here = Path(__file__).resolve().parent
    config_dir = here.parent / f"{level}s" / target / "configs"
    if not (config_dir / "bench.yaml").is_file():
        msg = (f"no bench config for level={level} target={target}: expected "
               f"{config_dir / 'bench.yaml'}. Every target owns one -- see "
               f"tests/layout/test_bench_config_per_target.py.")
        raise FileNotFoundError(msg)
    return os.path.relpath(config_dir, here)


@hydra.main(
    config_path=_target_config_path(),
    config_name="bench",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # The config set is chosen at import time via MINIWORLD_CONFIG_DIR (see autotune.configs):
    # this module's own header imports miniworld_engine.modules, which pulls kernel modules in,
    # so selecting here would already be too late for every op registered by that chain.
    _cfgdir = str(getattr(cfg, "config_dir", "") or "")
    if _cfgdir and _cfgdir != os.environ.get("MINIWORLD_CONFIG_DIR", ""):
        msg = (f"config_dir={_cfgdir} but MINIWORLD_CONFIG_DIR="
               f"{os.environ.get('MINIWORLD_CONFIG_DIR', '')!r}: the set must be in the "
               f"environment before this process imports any kernel.")
        raise ValueError(msg)
    if _cfgdir:
        print(f"config set: {_cfgdir}", flush=True)
    conf = BenchConfig.model_validate(cfg)
    # The registry says which dtype(s) a target actually has kernels for (bench_policy). Benching
    # outside that set measures a fallback -- an fp32 run of a bf16-only fused op silently lands on
    # the torch reference and gets reported under the kernel's name -- so say so loudly rather than
    # publishing it. A warning, not an error: an exploratory run is legitimate, a silent one is not.
    from benchmarks.runners.bench_policy import declared_precisions
    allowed = declared_precisions(conf.level, conf.target)
    requested_input_precision = 32 if conf.precision == 32 else "bf16"
    if requested_input_precision not in allowed:
        print(f"NOTE: {conf.level}/{conf.target} default precision policy is {allowed}; "
              f"this exploratory run requests {conf.precision!r}. Actual dtype and execution path "
              "are recorded per implementation.", flush=True)
    if not conf.compile and conf.cudagraph == "disabled" and not conf.allow_eager:
        msg = ("Final benchmarks must run compiled or cudagraph'd. Use compile=true or "
               "cudagraph=manual|graphed, or allow_eager=true for the ref1 eager floor.")
        raise ValueError(msg)
    from miniworld_engine.autotune.capture import install_launch_recorder
    install_launch_recorder()   # coverage accounting; no effect on what or how anything is benched
    conf.implementations = expand_implementations(conf.level, conf.target, conf.implementations)
    bench_func = targets_for(conf.level)[conf.target]

    torch.backends.cuda.matmul.allow_tf32 = conf.allow_tf32
    # Benchmark harness must measure the RAW module/kernel, NOT a training-framework wrapper.
    # Lightning Fabric's setup_module wrapper + fabric.backward add ~110us/step of GPU work
    # (input/output/grad casts+copies) that has nothing to do with the kernel under test and
    # both inflates absolute latency and COMPRESSES the speedup ratios. The models are already
    # placed on-device and cast to the right dtype by each bench (bf16 or fp32), so Fabric's
    # precision/placement is redundant here. Use a no-op shim: setup_module -> identity,
    # backward -> tensor.backward. (Kernel benches never touched fabric; this fixes the module
    # benches to the same raw-measurement standard.)
    class _NoFabric:
        @staticmethod
        def launch() -> None:
            pass

        @staticmethod
        def setup_module(module):
            return module

        @staticmethod
        def backward(tensor, gradient):
            tensor.backward(gradient)

    fabric = _NoFabric()
    fabric.launch()

    # Opt-in autotune-cache BUILD hook (``settings.capture``): instrument the Triton
    # autotuner so every config benched during this sweep is recorded per (op, dtype, bucket)
    # and written to the runtime cache at the end. Pair with settings.run_autotune so the full
    # grid (not a cached top-K) is benched. No-op otherwise; never affects benchmark numbers.
    from miniworld_engine import settings as _settings
    _capture_on = (
        _settings.current().capture
        or bool(getattr(conf, "autotune_shard", ""))  # shard build turns capture on by itself
    )
    if _capture_on:
        from miniworld_engine import settings as _settings
        from miniworld_engine.autotune import capture as _capture
        _pin = (getattr(conf, "pin_gate_backend", "") or "").strip().lower()
        # A capture MUST bench the full grid: with run_autotune off, make_cache_prune narrows the
        # candidates to the committed cache's top-K, so the build re-measures its own previous
        # answer (5 configs per bucket instead of 80-1500) and can never find a better one. This
        # used to come from MINIWORLD_RUN_AUTOTUNE=1 in the launcher; tying it to capture removes
        # the chance of running a build without it.
        _pins = {"run_autotune": True, "capture": True}
        _cj = int(getattr(conf, "compile_jobs", 0) or 0)
        if _cj:
            _pins["compile_jobs"] = _cj
        if _pin:
            _pins["pin_gate_backend"] = _pin
        _concat = getattr(conf, "pin_infer_concat", None)
        if _concat is not None:
            _pins["pin_infer_concat"] = bool(_concat)
        _b2b = getattr(conf, "pin_transition_cuda_b2b", None)
        if _b2b is not None:
            _pins["transition_cuda_b2b"] = bool(_b2b)
        _settings.configure(**_pins)
        print("  [capture] full grid unlocked; "
              + ", ".join(f"{k.removeprefix('pin_')}={v}" for k, v in _pins.items()), flush=True)
        _capture.install()

    provenance = make_run_provenance(
        {"benchmark": conf.model_dump(), "settings": vars(_settings.current()),
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}, benchmark_source_hash())
    object.__setattr__(conf, "_benchmark_provenance", provenance)
    bench_args = [
        conf.target,
        f"n_layers={conf.n_layers}",
        mode_label(conf.mode),
        conf.metric,
        str(conf.precision),
    ]
    if conf.compile:
        bench_args.append("compile")
    if conf.layernorm_weight_precision is not None:
        bench_args.append(f"affine-{conf.layernorm_weight_precision}")
    if conf.cudagraph != "disabled":
        bench_args.append(f"cudagraph-{conf.cudagraph}")
    # ...and in the NAME too, not just the column: the run name is the CSV filename, so without
    # this a custom_op run silently overwrites the disable run it was meant to be compared with.
    _wrap = _compile_wrap_now()
    if _wrap != "disable":
        bench_args.append(f"wrap-{_wrap}")
    bench_args.append(conf.sweep_axis)
    bench_args.append(f"cfg-{provenance['config_hash'][:8]}-src-{provenance['source_hash'][:8]}-run-{provenance['run_id'][:12]}")
    if conf.name_suffix:
        bench_args.append(conf.name_suffix)
    run_name = "_".join(bench_args)

    gpu_name = torch.cuda.get_device_name(0)
    results_dir = target_dir(conf.level, conf.target) / "artifacts" / gpu_name
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{run_name}.csv"
    tmp_csv_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    import json
    csv_path.with_suffix(".run.json").write_text(json.dumps(provenance, indent=2, default=str))
    autotune_cache_records: dict[str, dict[tuple, triton.Config]] = {}
    autotune_single_config_records: dict[str, triton.Config] = {}
    seen_autotuners: set[str] = set()

    if conf.sweep_axis == "seq_len":
        sweep_points = [
            (seq_len, conf.d_pair)
            for seq_len in range(conf.min_seq_len, conf.max_seq_len + 1, conf.seq_len_step)
        ]
    else:
        d_pair_values = conf.d_pair_values or list(
            range(conf.min_d_pair, conf.max_d_pair + 1, conf.d_pair_step),
        )
        sweep_points = [(conf.sweep_seq_len, d_pair) for d_pair in d_pair_values]
    if not sweep_points or not conf.implementations:
        raise ValueError("benchmark request produced no measurements")
    unsuccessful_rows = 0
    with tmp_csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for seq_len, d_pair in sweep_points:
            conf.d_pair = d_pair
            for implementation in conf.implementations:
                torch.manual_seed(0)
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                status = "ok"
                error = ""
                try:
                    require_source_identity(provenance["source_hash"])
                    with forward_stream(conf):
                        result = bench_func(conf, seq_len, implementation, fabric)
                    require_source_identity(provenance["source_hash"])
                    capture_autotune_state(
                        conf.level,
                        conf.target,
                        autotune_cache_records,
                        autotune_single_config_records,
                        seen_autotuners,
                    )
                except Exception as exc:
                    result = None
                    status = "unsupported" if isinstance(exc, UnsupportedBenchmark) else "failed"
                    error = ascii_safe(f"{type(exc).__name__}: {exc}")
                row = csv_row(
                    conf=conf,
                    run_name=run_name,
                    device_name=gpu_name,
                    seq_len=seq_len,
                    implementation=implementation,
                    result=result,
                    status=status,
                    error=error,
                )
                unsuccessful_rows += int(row["status"] != "ok")
                writer.writerow(row)
                if row["status"] != "ok":
                    error = row["error"]
                    print(
                        f"{conf.target} seq_len={seq_len} d_pair={d_pair} "
                        f"implementation={implementation} failed: {error}",
                        flush=True,
                    )
                else:
                    assert result is not None
                    print(
                        f"{conf.target} seq_len={seq_len} d_pair={d_pair} "
                        f"implementation={implementation} {conf.metric}={result.value:.6g} "
                        f"{result_unit(conf.metric)}",
                        flush=True,
                    )
    tmp_csv_path.replace(csv_path)
    print(f"\nwrote {csv_path}")

    if _capture_on:
        print("\n[autotune-capture] captured configs:")
        print(_capture.precompile_summary())
        print(_capture.summary())
        shard = getattr(conf, "autotune_shard", "") or ""
        if shard:
            # Dedicated parallel builder (miniworld_engine.autotune.builder): dump this run's
            # timings to its OWN shard file instead of the in-repo cache, so many parallel
            # capture jobs never race on the committed tree. A single merge step folds shards in.
            n_ops = _capture.dump_shard(shard)
            print(f"  [shard] dumped {n_ops} ops -> {shard}")
        else:
            written = _capture.flush(top_k=5)
            for op, dtype, bucket, n, fp in written:
                print(f"  wrote {op} [{dtype}|{bucket}] ({n} configs) -> {fp}")
        _capture.reset()

    autotune_summary = build_autotune_summary(
        conf.level,
        conf.target,
        cache_records=autotune_cache_records,
        single_config_records=autotune_single_config_records,
        seen_autotuners=seen_autotuners,
    )
    # Which ops this target actually launched -- the coverage denominator for any claim that a
    # sweep "tested the kernels". Identified by config-list identity (autotune.configs.op_of).
    from miniworld_engine.autotune.capture import launched_ops
    _lo = launched_ops()
    if _lo:
        print(f"\nops launched ({len(_lo)}): " + " ".join(sorted(_lo)), flush=True)
        # Persisted, not just printed: each target is its own process, so the caller can only
        # aggregate coverage across a whole 'all' run by reading these back.
        (results_dir / f"{run_name}.ops").write_text("\n".join(sorted(_lo)) + "\n")
    if autotune_summary is None:
        print("\nNo Triton autotune configs were captured during this run.")
        if unsuccessful_rows:
            raise SystemExit(1)
        return

    print(f"\n{autotune_summary}")
    summary_path = autotune_summary_path(results_dir, run_name)
    summary_path.write_text(
        f"{autotune_summary}\n",
        encoding="ascii",
    )
    (results_dir / "autotune_summary.txt").write_text(
        f"{autotune_summary}\n",
        encoding="ascii",
    )
    print(f"wrote {summary_path}")
    if unsuccessful_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
