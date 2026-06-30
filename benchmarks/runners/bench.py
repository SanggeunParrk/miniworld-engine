# vendored + trimmed from team-gm psk/benchmark : benchmarks/runners/bench.py
# Single bench entry for miniworld-kernels. Drops the model-level
# Pairformer / DiffusionTransformer benches; keeps the kernel-wrapping layers.
import importlib
import sys
import csv
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from typing import Literal

# When run as `python benchmarks/runners/bench.py`, sys.path[0] is
# `.../benchmarks/runners`, which can
# shadow stdlib `profile`. Point it at the repo root and add `src/` so the
# `miniworld_kernels` package imports without an editable install.
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
from lightning import Fabric
from omegaconf import DictConfig
from pydantic import BaseModel
from triton.runtime.autotuner import Autotuner

from miniworld_kernels.modules import (
    AdaptiveLayerNorm,
    AugmentedAttentionPairBias,
    ConditionedTransition,
    ImplementationType,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1_bidir import (
    fused_bidirectional_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.modules.triangle_multiplication.module import _load_cute_fns

DTV1_IMPL = "dtv1"
MINIWORLD_IMPL = "miniworld"
OLD_TRITON_IMPL = "old_triton"
if not torch.cuda.is_available():
    msg = "CUDA is not available. Please run on a machine with a CUDA-capable GPU."
    raise RuntimeError(msg)
DEVICE = torch.device("cuda")
FP32_PRECISION = 32


class BenchConfig(BaseModel):
    d_single: int = 384
    d_pair: int = 128
    d_single_token: int = 768
    d_single_atom: int = 128
    d_pair_atom: int = 16

    n_layers: int = 1
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

    kernel: str
    implementations: list[str] = [
        ImplementationType.PYTORCH.value,
        ImplementationType.TRITON.value,
    ]
    mode: Literal["inference", "training"]
    metric: Literal["time", "memory"]
    compile: bool = False
    # CUDA-graph the benched MODULE (host/launch overhead removed — the deployment regime for
    # graph-break cute/triton kernels). Module-scoped (optimizer/loss stay outside). Overrides
    # `compile` (capture is on the eager module). Modes:
    #   "disabled" — no graph (compile or eager).
    #   "manual"  — manual torch.cuda.graph capture of one static shape; what BUCKETED training
    #               uses (one graph/bucket + per-step input copy_). Works for any impl.
    #   "graphed" — torch.cuda.make_graphed_callables (auto static buffers + input copy); the
    #               PAD-TO-MAX single-shape regime (e.g. fixed 384 crops / multi-GPU max-len).
    #               Training only; fabric-wrapped baselines (dtv1) may fail here (raw backward).
    cudagraph: Literal["disabled", "manual", "graphed"] = "disabled"
    allow_tf32: bool = True
    precision: Literal[32, "bf16", "bf16-mixed"] = 32
    name_suffix: str = ""


def is_inference_mode(mode: str) -> bool:
    return mode == "inference"


def mode_label(mode: str) -> str:
    return "inference" if is_inference_mode(mode) else "training"


class ImplementationSpec(NamedTuple):
    impl: ImplementationType
    ln_impl: ImplementationType | None
    label: str


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


def module_miniworld_spec(raw: str) -> ImplementationSpec:
    if raw.strip().lower() == MINIWORLD_IMPL:
        return ImplementationSpec(ImplementationType.CUEQUIVARIANCE, None, raw)
    if raw.strip().lower() == OLD_TRITON_IMPL:
        return ImplementationSpec(ImplementationType.TRITON, None, raw)
    return parse_implementation_spec(raw)


def triton_miniworld_spec(raw: str) -> ImplementationSpec:
    if raw.strip().lower() == MINIWORLD_IMPL:
        return ImplementationSpec(ImplementationType.TRITON, None, raw)
    if raw.strip().lower() == OLD_TRITON_IMPL:
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
    *,
    conf: BenchConfig,
    func: Callable,
    grad_to_none: list,
    params: list,
    is_train: bool,
    input_dtype: str,
    parameter_dtype: str,
    execution_path: str,
    reference: str,
) -> BenchResult:
    if conf.metric == "memory":
        value = bench_memory(func)["median_mb"]
    elif conf.cudagraph == "manual":
        graph = capture_cudagraph(func, params, is_train=is_train)
        value = bench_time(graph.replay, grad_to_none=grad_to_none)["median_ms"]
    elif conf.cudagraph == "graphed":
        if is_train:
            graphed = torch.cuda.make_graphed_callables(func, ())
            value = bench_time(graphed, grad_to_none=grad_to_none)["median_ms"]
        else:
            graph = capture_cudagraph(func, [], is_train=False)
            value = bench_time(graph.replay, grad_to_none=grad_to_none)["median_ms"]
    else:
        value = bench_time(func, grad_to_none=grad_to_none)["median_ms"]
    return BenchResult(
        value=value,
        input_dtype=input_dtype,
        parameter_dtype=parameter_dtype,
        execution_path=execution_path,
        reference=reference,
    )


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
    side = torch.cuda.Stream()
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
        with torch.cuda.graph(graph):
            step()
    else:
        with torch.cuda.graph(graph), torch.no_grad():
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
    from miniworld_kernels.kernels.trimul_inproj.cute.inference import (
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

    from miniworld_kernels.kernels.layernorm.triton.main import (
        triton_layernorm,
        triton_layernorm_masked,
    )
    from miniworld_kernels.kernels.trimul_inproj.cute.back_split import (
        trimul_back_split,
    )
    from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
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
        eps,
    )


class MiniWorldTriangleMultiplicationInference(nn.Module):
    def __init__(self, base: TriangleMultiplication) -> None:
        super().__init__()
        from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
        from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
            prepack_lr_operand,
        )

        _bdll_patch.apply()
        _load_cute_fns()

        base = base.to(device=DEVICE, dtype=torch.bfloat16)
        self.left_weight = base.to_left.weight.T
        self.left_gate_weight = base.to_left_gate.weight.T
        self.right_weight = base.to_right.weight.T
        self.right_gate_weight = base.to_right_gate.weight.T
        self.gate_weight = base.to_gate.weight.T
        self.out_weight = base.to_out.weight.T
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
        from miniworld_kernels.kernels.trimul_inproj.cute.v6_training_merged import (
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


def bench_triangle_multiplication(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
    bidirectional: bool = False,
):
    # single-dir TriangleMultiplication, or the bidirectional (outgoing+incoming) variant — the
    # only differences are the base module, the dt-v1 baseline fn, and the miniworld layer class;
    # the whole correctness + timing (incl. CUDA-graph) tail below is shared.
    base_cls = BidirectionalTriangleMultiplication if bidirectional else TriangleMultiplication
    torch.manual_seed(0)
    layer_states = []
    for _ in range(conf.n_layers):
        base = base_cls(conf.d_pair)
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
                    [base_cls(conf.d_pair) for _ in layer_states],
                )
                for layer, state in zip(self.layers, layer_states, strict=True):
                    layer.load_state_dict(state)
                return
            if raw_implementation == MINIWORLD_IMPL:
                layers = []
                for state in layer_states:
                    base = base_cls(conf.d_pair)
                    base.load_state_dict(state)
                    if bidirectional:
                        from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import (
                            BidirV6TriMul,
                        )
                        layers.append(BidirV6TriMul(base.to(torch.bfloat16)))
                    else:
                        layer_cls = (
                            MiniWorldTriangleMultiplicationInference
                            if is_inference_mode(conf.mode)
                            else MiniWorldTriangleMultiplicationTraining
                        )
                        layers.append(layer_cls(base))
                self.layers = nn.ModuleList(layers)
                return

            spec = parse_implementation_spec(raw_implementation)
            if bidirectional:
                self.layers = nn.ModuleList(
                    [base_cls(conf.d_pair, implementation=spec.impl) for _ in layer_states],
                )
            else:
                self.layers = nn.ModuleList(
                    [
                        TriangleMultiplication(
                            conf.d_pair,
                            implementation=spec.impl,
                            ln_implementation=spec.ln_impl or ImplementationType.PYTORCH,
                        )
                        for _ in layer_states
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
                p_in = torch.cat([layer.to_left.weight, layer.to_right.weight], dim=0)
                g_in = torch.cat([layer.to_left_gate.weight, layer.to_right_gate.weight], dim=0)
                if bidirectional:
                    pair = fused_bidirectional_dtv1(
                        pair, mask_2d,
                        norm_in_weight=layer.ln_pair.weight, norm_in_bias=layer.ln_pair.bias,
                        p_in_weight=p_in, g_in_weight=g_in,
                        norm_out_weight=layer.ln_out.weight, norm_out_bias=layer.ln_out.bias,
                        p_out_weight=layer.to_out.weight, g_out_weight=layer.to_gate.weight,
                        h=layer.d_hidden, eps=layer.ln_pair.eps,
                    )
                else:
                    pair = fused_triangle_multiplicative_update_dtv1(
                        pair, direction="outgoing", mask=mask_2d,
                        norm_in_weight=layer.ln_pair.weight, norm_in_bias=layer.ln_pair.bias,
                        p_in_weight=p_in, g_in_weight=g_in,
                        norm_out_weight=layer.ln_out.weight, norm_out_bias=layer.ln_out.bias,
                        p_out_weight=layer.to_out.weight, g_out_weight=layer.to_gate.weight,
                        eps=layer.ln_pair.eps,
                    )
            return pair

    model = MultiTriangleMultiplication(implementation).to(DEVICE)
    if conf.compile and conf.cudagraph == "disabled":   # cudagraph captures the eager module (below)
        model.compile()
    if implementation != MINIWORLD_IMPL:
        model = fabric.setup_module(model)

    ref_model = MultiTriangleMultiplication(ImplementationType.PYTORCH.value).to(DEVICE)
    if conf.compile:
        ref_model.compile()
    ref_model = fabric.setup_module(ref_model)

    pair_dtype = torch.bfloat16 if implementation == MINIWORLD_IMPL else torch.float32
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

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

    def correctness() -> dict[str, float]:
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
        grad_max, grad_rel, grad_cos = tensor_metrics(pair_impl.grad, pair_ref.grad)
        return {
            "output_max_abs": out_max,
            "output_rel_frob": out_rel,
            "output_cosine": out_cos,
            "grad_max_abs": grad_max,
            "grad_rel_frob": grad_rel,
            "grad_cosine": grad_cos,
        }

    accuracy = correctness()
    for item in [pair, *list(model.parameters()), *list(ref_model.parameters())]:
        item.grad = None
    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [pair, *list(model.parameters())]
    if conf.metric == "time" and conf.cudagraph == "manual":
        # manual capture of one static shape (bucketed-training regime); replay timed.
        graph = capture_cudagraph(
            func, [p for p in model.parameters() if p.requires_grad],
            is_train=not is_inference_mode(conf.mode),
        )
        value = bench_time(graph.replay, grad_to_none=[])["median_ms"]
    elif conf.metric == "time" and conf.cudagraph == "graphed":
        # make_graphed_callables (pad-to-max single-shape regime): auto static buffers + copy.
        if is_inference_mode(conf.mode):                       # fwd-only: manual no-grad capture
            graph = capture_cudagraph(func, [], is_train=False)
            value = bench_time(graph.replay, grad_to_none=[])["median_ms"]
        else:
            graphed = torch.cuda.make_graphed_callables(model, (pair, mask))

            def graphed_step() -> None:
                graphed(pair, mask).backward(dy)

            value = bench_time(graphed_step, grad_to_none=[])["median_ms"]
    elif conf.metric == "time":
        value = bench_time(func, grad_to_none=grad_to_none)["median_ms"]
    else:
        value = bench_memory(func)["median_mb"]
    parameter = next(model.parameters(), None)
    return BenchResult(
        value=value,
        input_dtype=str(pair.dtype).replace("torch.", ""),
        parameter_dtype="" if parameter is None else str(parameter.dtype).replace("torch.", ""),
        execution_path=triangle_multiplication_path(implementation, conf.mode, conf.d_pair),
        reference=ImplementationType.PYTORCH.value,
        **accuracy,
    )


def bench_bias_only_attention(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = triton_miniworld_spec(implementation)
    is_old_triton = implementation.strip().lower() == OLD_TRITON_IMPL
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class OldTritonBiasOnlyAttention(TriangleAttention):
        def _kernel_bias_only_attention(
            self,
            value: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            from miniworld_kernels import kernels

            return kernels.triton_bias_only_attention(value, bias)

    class MultiTriangleAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer_cls = OldTritonBiasOnlyAttention if is_old_triton else TriangleAttention
            self.layers = nn.ModuleList(
                [
                    layer_cls(
                        conf.d_pair,
                        implementation=spec.impl,
                        use_self_attention=False,
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

    model = MultiTriangleAttention().to(device=DEVICE, dtype=dtype)
    if conf.compile and conf.cudagraph == "disabled":
        model.compile()
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(pair)
    pair.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def inference_step() -> torch.Tensor:
        return model(pair, mask)

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [pair, *list(model.parameters())]
    execution_path = {
        ImplementationType.PYTORCH: "module.reference.torch",
        ImplementationType.TRITON: (
            "modules.triangle_attention.bias_only."
            "layernorm_linear+torch_bias_only_attention+gate_out"
        ),
        ImplementationType.CUEQUIVARIANCE: "cuequivariance_torch.triangle_attention",
    }.get(spec.impl, spec.impl.value)
    if is_old_triton:
        execution_path = "kernels.bias_only_attention.triton.main"
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(pair.dtype).replace("torch.", ""),
        parameter_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        execution_path=execution_path,
        reference="module.reference.torch",
    )


def bench_triangle_attention(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = triton_miniworld_spec(implementation)
    if implementation.strip().lower() == OLD_TRITON_IMPL:
        return BenchResult(
            value=float("nan"),
            status="failed",
            error="old_triton is only defined for bias_only_attention",
        )

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

    model = MultiTriangleAttention().to(device=DEVICE, dtype=dtype)
    if conf.compile and conf.cudagraph == "disabled":
        model.compile()
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(pair)
    pair.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def inference_step() -> torch.Tensor:
        return model(pair, mask)

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

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
        parameter_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        execution_path=execution_path,
        reference="module.reference.torch",
    )


def bench_transition(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = module_miniworld_spec(implementation)
    is_old_triton = implementation.strip().lower() == OLD_TRITON_IMPL

    class OldTritonTransition(Transition):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            from miniworld_kernels import kernels

            x = self.ln_in(x)
            return kernels.triton_transition(
                x,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
            )

    class MultiTransition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer_cls = OldTritonTransition if is_old_triton else Transition
            self.layers = nn.ModuleList(
                [
                    layer_cls(conf.d_pair, implementation=spec.impl)
                    for _ in range(conf.n_layers)
                ],
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    model = MultiTransition().to(DEVICE)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    x = torch.randn(1, seq_len, seq_len, conf.d_pair).to(DEVICE).to(torch.bfloat16)
    dy = torch.randn_like(x)
    x.requires_grad = True

    def inference_step() -> torch.Tensor:
        return model(x)

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [x, *list(model.parameters())]
    if spec.impl == ImplementationType.CUEQUIVARIANCE:
        if not is_inference_mode(conf.mode) and conf.d_pair >= 256:
            execution_path = "module.reference.torch.dispatch_fallback"
        else:
            execution_path = (
                "kernels.transition.cute.fused"
                if conf.d_pair >= 256
                else "kernels.transition.triton.fused"
            )
    else:
        execution_path = {
            ImplementationType.PYTORCH: "module.reference.torch",
            ImplementationType.TRITON: "kernels.transition.triton.fused",
            ImplementationType.CUDA: "kernels.transition.cuda",
            ImplementationType.CUTE: (
                "kernels.transition.cute.forward+triton.backward"
                if not is_inference_mode(conf.mode)
                else "kernels.transition.cute.fused"
            ),
        }.get(spec.impl, spec.impl.value)
    if is_old_triton:
        execution_path = "kernels.transition.triton.main"
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(x.dtype).replace("torch.", ""),
        parameter_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        execution_path=execution_path,
        reference="module.reference.torch",
    )


def bench_conditioned_transition(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = module_miniworld_spec(implementation)
    if spec.impl not in {
        ImplementationType.PYTORCH,
        ImplementationType.TRITON,
        ImplementationType.CUEQUIVARIANCE,
    }:
        return as_bench_result(float("nan"))

    class MultiConditionedTransition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    ConditionedTransition(
                        d_hidden=conf.d_pair,
                        d_cond=conf.d_single_token,
                        implementation=spec.impl,
                    )
                    for _ in range(conf.n_layers)
                ],
            )

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x, cond)
            return x

    model = MultiConditionedTransition().to(DEVICE)
    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    x = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_pair,
        device=DEVICE,
        requires_grad=True,
    )
    cond = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_single_token,
        device=DEVICE,
        requires_grad=True,
    )
    dy = torch.randn_like(x)

    def inference_step() -> torch.Tensor:
        return model(x, cond)

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [x, cond, *list(model.parameters())]
    if spec.impl in {ImplementationType.TRITON, ImplementationType.CUEQUIVARIANCE}:
        if is_inference_mode(conf.mode):
            execution_path = (
                "kernels.conditioned_transition.triton.inference"
                if conf.d_pair <= 128
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
        parameter_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        execution_path=execution_path,
        reference="module.reference.torch",
    )


def bench_adaptive_layernorm(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = triton_miniworld_spec(implementation)
    implementation_type = spec.impl
    if implementation_type not in {
        ImplementationType.PYTORCH,
        ImplementationType.TRITON,
    }:
        return as_bench_result(float("nan"))

    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiAdaptiveLayerNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    AdaptiveLayerNorm(
                        d_hidden=conf.d_pair,
                        d_cond=conf.d_pair,
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
    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    x = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_pair,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    cond = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_pair,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    dy = torch.randn_like(x)

    def inference_step() -> torch.Tensor:
        return model(x, cond)

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [x, cond, *list(model.parameters())]
    try:
        return measured_result(
            conf=conf,
            func=func,
            grad_to_none=grad_to_none,
            params=list(model.parameters()),
            is_train=not is_inference_mode(conf.mode),
            input_dtype=str(x.dtype).replace("torch.", ""),
            parameter_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
            execution_path=(
                "module.reference.torch"
                if implementation_type == ImplementationType.PYTORCH
                else "kernels.adaln.triton.main"
            ),
            reference="module.reference.torch",
        )
    except torch.cuda.OutOfMemoryError:
        return as_bench_result(float("nan"))


def bench_augmented_attention_token(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = triton_miniworld_spec(implementation)
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16
    model = AugmentedAttentionPairBias(
        d_single=conf.d_single_token,
        d_cond=conf.d_single_token,
        d_pair=conf.d_pair,
        n_head=16,
        implementation=spec.impl,
    ).to(device=DEVICE, dtype=dtype)

    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair, device=DEVICE, dtype=dtype)
    single = torch.randn(
        conf.n_augment, 1, seq_len, conf.d_single_token, device=DEVICE, dtype=dtype
    )
    cond = torch.randn(
        conf.n_augment, 1, seq_len, conf.d_single_token, device=DEVICE, dtype=dtype
    )
    dy_single = torch.randn_like(single)
    pair.requires_grad = True
    single.requires_grad = True
    cond.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def inference_step() -> torch.Tensor:
        return model(single, cond, pair, mask)

    def training_step() -> None:
        out_single = inference_step()
        fabric.backward(out_single, dy_single)

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [pair, single, cond, *list(model.parameters())]
    return measured_result(
        conf=conf,
        func=func,
        grad_to_none=grad_to_none,
        params=list(model.parameters()),
        is_train=not is_inference_mode(conf.mode),
        input_dtype=str(single.dtype).replace("torch.", ""),
        parameter_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        execution_path=(
            "module.reference.torch"
            if spec.impl == ImplementationType.PYTORCH
            else "kernels.augmented_attention.triton.main"
        ),
        reference="module.reference.torch",
    )


def bench_augmented_attention_atom(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = triton_miniworld_spec(implementation)
    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16
    model = AugmentedAttentionPairBias(
        d_single=conf.d_single_atom,
        d_cond=conf.d_single_atom,
        d_pair=conf.d_pair_atom,
        n_head=4,
        implementation=spec.impl,
    ).to(device=DEVICE, dtype=dtype)

    if conf.compile:
        model.compile()
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

    def training_step() -> None:
        out_single = inference_step()
        fabric.backward(out_single, dy_single)

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [pair, single, cond, *list(model.parameters())]
    try:
        return measured_result(
            conf=conf,
            func=func,
            grad_to_none=grad_to_none,
            params=list(model.parameters()),
            is_train=not is_inference_mode(conf.mode),
            input_dtype=str(single.dtype).replace("torch.", ""),
            parameter_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
            execution_path=(
                "module.reference.torch"
                if spec.impl == ImplementationType.PYTORCH
                else "kernels.augmented_attention.triton.main"
            ),
            reference="module.reference.torch",
        )
    except torch.cuda.OutOfMemoryError:
        return as_bench_result(float("nan"))


def bench_bidirectional_triangle_multiplication(conf, seq_len, implementation, fabric):
    return bench_triangle_multiplication(conf, seq_len, implementation, fabric, bidirectional=True)


KERNEL_MAP = {
    "triangle_multiplication": bench_triangle_multiplication,
    "triangle_multiplication_bidirectional": bench_bidirectional_triangle_multiplication,
    "bias_only_attention": bench_bias_only_attention,
    "triangle_attention": bench_triangle_attention,
    "transition": bench_transition,
    "conditioned_transition": bench_conditioned_transition,
    "adaptive_layernorm": bench_adaptive_layernorm,
    "augmented_attention_token": bench_augmented_attention_token,
    "augmented_attention_atom": bench_augmented_attention_atom,
}

TARGET_DIRS = {
    "triangle_multiplication": _REPO_ROOT / "benchmarks" / "modules" / "triangle_multiplication",
    "triangle_multiplication_bidirectional": _REPO_ROOT / "benchmarks" / "modules"
    / "triangle_multiplication_bidirectional",
    "bias_only_attention": _REPO_ROOT / "benchmarks" / "modules" / "bias_only_attention",
    "triangle_attention": _REPO_ROOT / "benchmarks" / "modules" / "triangle_attention",
    "transition": _REPO_ROOT / "benchmarks" / "modules" / "transition",
    "conditioned_transition": _REPO_ROOT / "benchmarks" / "modules"
    / "conditioned_transition",
    "adaptive_layernorm": _REPO_ROOT / "benchmarks" / "modules" / "adaptive_layernorm",
    "augmented_attention_token": _REPO_ROOT / "benchmarks" / "modules" / "augmented_attention",
    "augmented_attention_atom": _REPO_ROOT / "benchmarks" / "modules" / "augmented_attention",
}

# Triton autotuner objects live in the per-op `triton/main.py` of each kernel.
AUTOTUNE_MODULES = {
    "triangle_multiplication": [
        "miniworld_kernels.modules.triangle_multiplication.baseline_dtv1",
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.tm1.triton.main",
        "miniworld_kernels.kernels.tm2.triton.main",
    ],
    "triangle_multiplication_bidirectional": [
        "miniworld_kernels.modules.triangle_multiplication.baseline_dtv1",
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.tm1.triton.main",
        "miniworld_kernels.kernels.tm2.triton.main",
        "miniworld_kernels.kernels.trimul_inproj.triton.back_fused",
    ],
    "bias_only_attention": [
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.layernorm_linear.triton.main",
        "miniworld_kernels.kernels.bias_only_attention.triton.gate_out",
        "miniworld_kernels.kernels.bias_only_attention.triton.main",
    ],
    "triangle_attention": [
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.bias_only_attention.triton.gate_out",
        "miniworld_kernels.kernels.triangle_attention.triton.main",
    ],
    "transition": [
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.transition.triton.main",
        "miniworld_kernels.kernels.transition.triton.fused",
    ],
    "conditioned_transition": [
        "miniworld_kernels.kernels.conditioned_transition.triton.inference",
        "miniworld_kernels.kernels.conditioned_transition.triton.composed",
        "miniworld_kernels.kernels.conditioned_transition.triton.training",
    ],
    "adaptive_layernorm": [
        "miniworld_kernels.kernels.adaln.triton.main",
    ],
    "augmented_attention_token": [
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.augmented_attention.triton.main",
    ],
    "augmented_attention_atom": [
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.augmented_attention.triton.main",
    ],
}


def get_autotuners(kernel: str) -> dict[str, Autotuner]:
    autotuners = {}
    for module_name in AUTOTUNE_MODULES.get(kernel, []):
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
    kernel: str,
    cache_records: dict[str, dict[tuple, triton.Config]],
    single_config_records: dict[str, triton.Config],
    seen_autotuners: set[str],
) -> None:
    for autotuner_name, autotuner in sorted(get_autotuners(kernel).items()):
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
    kernel: str,
    cache_records: dict[str, dict[tuple, triton.Config]] | None = None,
    single_config_records: dict[str, triton.Config] | None = None,
    seen_autotuners: set[str] | None = None,
) -> str | None:
    cache_records = cache_records or {}
    single_config_records = single_config_records or {}
    seen_autotuners = seen_autotuners or set(get_autotuners(kernel))

    sections = []
    current_autotuners = get_autotuners(kernel)
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


CSV_FIELDS = [
    "run_name",
    "target_kind",
    "target",
    "device",
    "torch_version",
    "cuda_version",
    "metric",
    "unit",
    "mode",
    "compiled",
    "cudagraph",
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
    "n_augment",
    "mask_prob",
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


def target_kind(kernel: str) -> str:
    target_dir = TARGET_DIRS[kernel]
    return target_dir.parent.name.rstrip("s")


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
    elif implementation == OLD_TRITON_IMPL:
        spec = ImplementationSpec(ImplementationType.TRITON, None, implementation)
    else:
        spec = parse_implementation_spec(implementation)
    implementation_type = DTV1_IMPL if spec is None else spec.impl.value
    if implementation == MINIWORLD_IMPL:
        implementation_type = MINIWORLD_IMPL
    if implementation == OLD_TRITON_IMPL:
        implementation_type = OLD_TRITON_IMPL
    return {
        "run_name": run_name,
        "target_kind": target_kind(conf.kernel),
        "target": conf.kernel,
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "metric": conf.metric,
        "unit": result_unit(conf.metric),
        "mode": mode_label(conf.mode),
        "compiled": conf.compile,
        "cudagraph": conf.cudagraph,
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
        "n_layers": conf.n_layers,
        "n_augment": conf.n_augment,
        "mask_prob": conf.mask_prob,
        "seq_len": seq_len,
        "tokens": seq_len * seq_len,
        "batch_size": 1,
        "d_pair": conf.d_pair,
        "d_single": conf.d_single,
        "d_single_token": conf.d_single_token,
        "d_single_atom": conf.d_single_atom,
        "d_pair_atom": conf.d_pair_atom,
        "status": status,
        "error": error,
        "value": None if result is None else result.value,
    }


@hydra.main(
    config_path="../modules/triangle_multiplication/configs",
    config_name="bench",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    conf = BenchConfig.model_validate(cfg)
    if not conf.compile and conf.cudagraph == "disabled":
        msg = "Final benchmarks must run compiled or cudagraph'd. Use compile=true or cudagraph=manual|graphed."
        raise ValueError(msg)
    bench_func = KERNEL_MAP[conf.kernel]

    torch.backends.cuda.matmul.allow_tf32 = conf.allow_tf32
    fabric = Fabric(accelerator="cuda", devices=1, precision=fabric_precision(conf.precision))
    fabric.launch()

    bench_args = [
        conf.kernel,
        f"n_layers={conf.n_layers}",
        mode_label(conf.mode),
        conf.metric,
        str(conf.precision),
    ]
    if conf.compile:
        bench_args.append("compile")
    if conf.cudagraph != "disabled":
        bench_args.append(f"cudagraph-{conf.cudagraph}")
    bench_args.append(conf.sweep_axis)
    if conf.name_suffix:
        bench_args.append(conf.name_suffix)
    run_name = "_".join(bench_args)

    gpu_name = torch.cuda.get_device_name(0)
    results_dir = TARGET_DIRS[conf.kernel] / "artifacts" / gpu_name
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{run_name}.csv"
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
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for seq_len, d_pair in sweep_points:
            conf.d_pair = d_pair
            for implementation in conf.implementations:
                torch._dynamo.reset()  # noqa: SLF001
                torch.cuda.empty_cache()
                status = "ok"
                error = ""
                try:
                    result = bench_func(conf, seq_len, implementation, fabric)
                    capture_autotune_state(
                        conf.kernel,
                        autotune_cache_records,
                        autotune_single_config_records,
                        seen_autotuners,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = None
                    status = "failed"
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
                writer.writerow(row)
                if result is None:
                    print(
                        f"{conf.kernel} seq_len={seq_len} d_pair={d_pair} "
                        f"implementation={implementation} failed: {error}",
                        flush=True,
                    )
                else:
                    print(
                        f"{conf.kernel} seq_len={seq_len} d_pair={d_pair} "
                        f"implementation={implementation} {conf.metric}={result.value:.6g} "
                        f"{result_unit(conf.metric)}",
                        flush=True,
                    )
    print(f"\nwrote {csv_path}")

    autotune_summary = build_autotune_summary(
        conf.kernel,
        cache_records=autotune_cache_records,
        single_config_records=autotune_single_config_records,
        seen_autotuners=seen_autotuners,
    )
    if autotune_summary is None:
        print("\nNo Triton autotune configs were captured during this run.")
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


if __name__ == "__main__":
    main()
