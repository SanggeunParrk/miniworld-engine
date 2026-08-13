# vendored + trimmed from team-gm psk/benchmark : benchmarks/runners/bench.py
# Single bench entry for miniworld-engine. Drops the model-level
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
from lightning import Fabric
from omegaconf import DictConfig
from pydantic import BaseModel
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

    # Dedicated parallel cache builder: when set, an autotune-capture run dumps its timings to
    # THIS shard file instead of the in-repo cache (submits/build_autotune_cache.py merges shards).
    autotune_shard: str = ""
    # Pin a device-calibrated dispatch switch for the duration of a capture. The card picks
    # one side for the shapes swept here, so the other side's kernels never fire and never
    # get captured -- yet they still run in production at other shapes, with no cached
    # configs at all. A build sweeps each side explicitly. "" = let the engine decide.
    pin_gate_backend: str = ""
    # Row-broadcast dropout probability for the trimul residual epilogue. USE_DROPOUT is part of
    # those kernels' autotune KEY, so `dropout=0` and `dropout>0` are different cache buckets and
    # neither substitutes for the other. A cache built entirely at 0 leaves training -- the only
    # place dropout is live -- with no entry, and the runtime falls back to the full grid, which
    # looks like a hang. Builds must sweep both.
    dropout: float = 0.0
    #: Worker processes for parallel pre-compilation of each autotune round (0 = auto).
    compile_jobs: int = 0
    #: Pin the inference LN+proj concat fusion; None = let the engine decide. Typed bool, not str:
    #: hydra parses `+pin_infer_concat=true` as a bool and a str field rejects it.
    pin_infer_concat: bool | None = None

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
    #   "manual"  — manual torch.cuda.graph capture of one static shape; what BUCKETED training
    #               uses (one graph/bucket + per-step input copy_). Works for any impl. DEFAULT:
    #               this is the deployment regime for the graph-breaking cute/triton kernels, so
    #               it is the representative number; run cudagraph=disabled for the eager baseline.
    #   "disabled" — no graph (compile or eager); host/launch overhead included (diagnostic).
    #   "graphed" — torch.cuda.make_graphed_callables (auto static buffers + input copy); the
    #               PAD-TO-MAX single-shape regime (e.g. fixed 384 crops / multi-GPU max-len).
    #               Training only; fabric-wrapped baselines (dtv1) may fail here (raw backward).
    cudagraph: Literal["disabled", "manual", "graphed"] = "manual"
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
        # MINIWORLD, not CUEQUIVARIANCE. `resolve()` sends a CUEQUIVARIANCE request for any
        # non-trimul op to KernelBackend.PYTORCH, so the old mapping made `implementations=
        # [miniworld]` bench the pytorch reference under the "ours" label for the two module
        # benches that use this spec (transition, conditioned_transition). Silent: the sweep
        # reported plausible times (3.16 ms at L=384/d=128, vs 0.65 ms for the triton path)
        # and the autotune-capture builder recorded NOTHING, because no triton kernel ever ran.
        return ImplementationSpec(ImplementationType.MINIWORLD, None, raw)
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


def actual_compiled_flag(conf: BenchConfig) -> bool:
    if conf.kernel == "transition" and conf.cudagraph != "disabled":
        return False
    return conf.compile


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
        from miniworld_engine.autotune import capture as _cap  # noqa: PLC0415
        _capturing = _cap._orig_bench is not None  # noqa: SLF001
    except Exception:  # noqa: BLE001
        _capturing = False
    if _capturing:
        for _ in range(2):
            if is_train:
                step()
            else:
                with torch.no_grad():
                    step()
        torch.cuda.synchronize()
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
                        base_cls(
                            conf.d_pair,
                            implementation=ImplementationType.MINIWORLD,
                            p_drop=float(getattr(conf, "dropout", 0.0) or 0.0),
                        ).to(
                            torch.bfloat16,
                        )
                        for _ in layer_states
                    ],
                )
                for layer, state in zip(self.layers, layer_states, strict=True):
                    layer.load_state_dict(state)
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
                # The miniworld TriangleMultiplication now ALWAYS adds the residual (it is
                # unconditional — see the module). The dtv1 baseline is the raw op, so add the
                # residual explicitly here to keep the per-layer stack semantics identical for a
                # fair speed/correctness comparison against the residual-inclusive pytorch ref.
                if bidirectional:
                    pair = pair + fused_bidirectional_dtv1(
                        pair, mask_2d,
                        norm_in_weight=layer.ln_pair.weight, norm_in_bias=layer.ln_pair.bias,
                        p_in_weight=p_in, g_in_weight=g_in,
                        norm_out_weight=layer.ln_out.weight, norm_out_bias=layer.ln_out.bias,
                        p_out_weight=layer.to_out.weight, g_out_weight=layer.to_gate.weight,
                        h=layer.d_hidden, eps=layer.ln_pair.eps,
                    )
                else:
                    pair = pair + fused_triangle_multiplicative_update_dtv1(
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
            from miniworld_engine import kernels

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
    dtype = torch.bfloat16

    class OldTritonTransition(Transition):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            from miniworld_engine import kernels

            # The base Transition now ALWAYS adds the residual; this legacy-triton baseline is the
            # raw op, so add the residual explicitly (residual == the module input) to stay
            # comparable to the residual-inclusive pytorch reference.
            out = kernels.triton_transition(
                self.ln_in(x),
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
            )
            return x + out

    class MultiTransition(nn.Module):
        def __init__(
            self,
            layer_spec: ImplementationSpec,
            *,
            use_old_triton: bool = False,
        ) -> None:
            super().__init__()
            layer_cls = OldTritonTransition if use_old_triton else Transition
            self.layers = nn.ModuleList(
                [
                    layer_cls(conf.d_pair, implementation=layer_spec.impl)
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

    model = MultiTransition(spec, use_old_triton=is_old_triton).to(DEVICE)
    for layer, state in zip(model.layers, layer_states, strict=True):
        layer.load_state_dict(state)
    model.train(not is_inference_mode(conf.mode))
    if conf.compile and conf.cudagraph == "disabled":
        model.compile()
    model = fabric.setup_module(model)

    ref_spec = ImplementationSpec(ImplementationType.PYTORCH, None, "pytorch")
    ref_model = MultiTransition(ref_spec).to(DEVICE)
    for layer, state in zip(ref_model.layers, layer_states, strict=True):
        layer.load_state_dict(state)
    ref_model.train(not is_inference_mode(conf.mode))
    if conf.compile and conf.cudagraph == "disabled":
        ref_model.compile()
    ref_model = fabric.setup_module(ref_model)

    torch.manual_seed(1)
    x = torch.randn(1, seq_len, seq_len, conf.d_pair, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(x)
    x.requires_grad = True

    def inference_step() -> torch.Tensor:
        return model(x)

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

    func = inference_step if is_inference_mode(conf.mode) else training_step
    grad_to_none = [x, *list(model.parameters())]

    def correctness() -> dict[str, float]:
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

    if spec.impl == ImplementationType.CUEQUIVARIANCE:
        # transition has no cuequivariance kernel; MiniWorld dispatches to the fastest per d.
        # NOTE: these labels must track modules/transition/module.py's actual routing, not just
        # the Hopper/Blackwell assumption. On pre-Hopper (sm_80 / A100) MINIWORLD routes large d
        # (>=256) to the shape-general split (_old_triton_forward -> kernels.transition.triton.main),
        # NOT the cute fused path (cute is sm_90+ only and never runs here). d=128 stays fused.
        _cap0 = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        if _cap0 < 9 and conf.d_pair >= 256:  # noqa: PLR2004 -- pre-Hopper large-d -> split
            execution_path = "kernels.transition.triton.main"
        elif not is_inference_mode(conf.mode) and conf.d_pair >= 512:
            execution_path = "kernels.transition.cute.forward+triton.backward"
        elif not is_inference_mode(conf.mode) and conf.d_pair >= 256:
            execution_path = "kernels.transition.triton.fused"
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
    )._replace(**accuracy)


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
        ImplementationType.MINIWORLD,
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


def _acc_fwd(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    _, rel, cos = tensor_metrics(actual, expected)
    return {"output_rel_frob": rel, "output_cosine": cos}


def _acc_grad(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    _, rel, cos = tensor_metrics(actual, expected)
    return {"grad_rel_frob": rel, "grad_cosine": cos}


def _flat(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).float() for t in items])


def _fwd_result(conf, kfn, args, *, acc, path, ref, dtype):
    """Time a pure forward launcher ``kfn(*args)`` (is_train=False) + attach correctness ``acc``."""
    return measured_result(
        conf=conf, func=lambda: kfn(*args), grad_to_none=[], params=[], is_train=False,
        input_dtype=dtype, parameter_dtype=dtype, execution_path=path, reference=ref,
    )._replace(**acc)


def _bwd_autograd_result(conf, out, leaves, dy, ref_grad, *, path, ref, dtype):
    """Backward-only timing via ``torch.autograd.grad`` on a pre-built forward graph ``out``.
    Cosine of leaves[0]'s grad vs ``ref_grad``. is_train=True so cudagraph capture keeps grad on."""
    def kfn():
        return torch.autograd.grad(out, leaves, dy, retain_graph=True)
    g = kfn()[0]
    acc = _acc_grad(g, ref_grad)
    return measured_result(
        conf=conf, func=kfn, grad_to_none=[], params=[], is_train=True,
        input_dtype=dtype, parameter_dtype=dtype, execution_path=path, reference=ref,
    )._replace(**acc)


# ---- FORWARD operations -----------------------------------------------------------------------
def bench_kernel_dual_gemm_epil(conf, seq_len, implementation, fabric):
    """Gated dual-GEMM in-projection (trimul front): left=(x@WL)*sigma(x@WLg), right=..., gate=sigma(x@Wg).
    Rows: pytorch, trimul_front_triton, trimul_inproj_cute, tm1_cute, triton_tm1,
    trimul_front_sm100(dep). Variants without a gate compare left|right only."""
    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)

    def _w():
        return (torch.randn(D, D, device=DEVICE, dtype=BF16) * (D**-0.5)).contiguous()

    wl, wlg, wr, wrg, wg = _w(), _w(), _w(), _w(), _w()

    def _x():
        torch.manual_seed(1)
        return torch.randn(1, L, L, D, device=DEVICE, dtype=BF16).contiguous()

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
    elif implementation == "trimul_front_triton":
        from miniworld_engine.kernels.trimul_inproj.triton.front import trimul_front_triton

        def run(x):
            left, right, gate = trimul_front_triton(x, wl, wlg, wr, wrg, wg)
            return bdll_to_md(left), bdll_to_md(right), gate.reshape(L * L, D)
        path = "kernels.trimul_inproj.triton.front"
    elif implementation == "trimul_inproj_cute":
        from miniworld_engine.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward

        def run(x):
            left, right, gate = trimul_inproj_cute_forward(x, wl, wlg, wr, wrg, wg, compute_gate=True)
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
            left, right = triton_tm1(x.reshape(L * L, D), wl, wlg, wr, wrg)
            return left.reshape(L * L, D), right.reshape(L * L, D)
        path = "kernels.tm1.triton.main"
    elif implementation == "trimul_front_sm100":
        from miniworld_engine.kernels.trimul_inproj.cute.front_sm100 import trimul_front_sm100

        def run(x):
            left, right = trimul_front_sm100(x, wl, wlg, wr, wrg)
            return bdll_to_md(left), bdll_to_md(right)
        path = "kernels.trimul_inproj.cute.front_sm100"
    else:
        return as_bench_result(float("nan"))

    xc = _x()
    res = run(xc)
    outs_a, outs_e = [res[0], res[1]], list(ref_lr(xc))
    if len(res) == 3:
        outs_a.append(res[2])
        outs_e.append(ref_gate(xc))
    acc = _acc_fwd(_flat(outs_a), _flat(outs_e))
    return _fwd_result(conf, run, (_x(),), acc=acc, path=path, ref="pytorch", dtype="bfloat16")


def bench_kernel_gemm_epil(conf, seq_len, implementation, fabric):
    """Fused LayerNorm+Linear (GEMM w/ LN epilogue): Y = LN(x) @ W^T. N=K=d. Rows: pytorch,
    layernorm_linear_triton, layernorm_linear_cute(M1), layernorm_linear_cute_fused(M2), layernorm_linear_te."""
    import torch.nn.functional as F

    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    lw = torch.randn(D, device=DEVICE, dtype=BF16)
    lb = torch.randn(D, device=DEVICE, dtype=BF16)
    w = (torch.randn(D, D, device=DEVICE, dtype=BF16) * (D**-0.5)).contiguous()
    eps = 1e-5

    def _x():
        torch.manual_seed(1)
        return torch.randn(L * L, D, device=DEVICE, dtype=BF16).contiguous()

    def ref(x):
        return F.linear(F.layer_norm(x, (D,), lw, lb, eps), w)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch"
    elif implementation == "layernorm_linear_triton":
        from miniworld_engine.kernels.layernorm_linear.interface import layernorm_linear_triton
        kfn = lambda x: layernorm_linear_triton(x, lw, lb, w, None, eps)  # noqa: E731
        path = "kernels.layernorm_linear.triton.fused"
    elif implementation == "layernorm_linear_cute":
        from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
            layernorm_linear_cute,
        )
        kfn = lambda x: layernorm_linear_cute(x, lw, lb, w, None, eps)  # noqa: E731
        path = "kernels.layernorm_linear.cute.gemm_layernorm_linear"
    elif implementation == "layernorm_linear_cute_fused":
        from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
            layernorm_linear_cute_fused,
        )
        kfn = lambda x: layernorm_linear_cute_fused(x, lw, lb, w, None, eps)  # noqa: E731
        path = "kernels.layernorm_linear.cute.gemm_layernorm_linear_fused"
    elif implementation == "layernorm_linear_te":
        from miniworld_engine.kernels.layernorm_linear.te_style import layernorm_linear_te_fn
        kfn = lambda x: layernorm_linear_te_fn(x, lw, lb, w, None, eps)  # noqa: E731
        path = "kernels.layernorm_linear.te_style"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_fwd(kfn(_x()), ref(_x()))
    return _fwd_result(conf, kfn, (_x(),), acc=acc, path=path, ref="pytorch", dtype="bfloat16")


def bench_kernel_transition_b2b(conf, seq_len, implementation, fabric):
    """SwiGLU MLP (transition, back-to-back): out = squeeze(silu(LN(x)@Wa)*(LN(x)@Wb)). Rows: pytorch,
    triton_transition_fused, cute_transition_fused, transition_b2b_ktiled(unverified)."""
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.transition import Transition

    D, L, n = conf.d_pair, seq_len, 4
    torch.manual_seed(0)
    ref_mod = Transition(D, n=n, implementation=ImplementationType.PYTORCH).to(DEVICE).to(BF16)
    for lin in (ref_mod.expand_a, ref_mod.expand_b, ref_mod.squeeze):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
    lw, lb = ref_mod.ln_in.weight, ref_mod.ln_in.bias
    wa, wb, wsq = ref_mod.expand_a.weight, ref_mod.expand_b.weight, ref_mod.squeeze.weight
    eps = ref_mod.ln_in.eps

    def _x():
        torch.manual_seed(1)
        return torch.randn(1, L, L, D, device=DEVICE, dtype=BF16)

    if implementation == "pytorch":
        kfn, path = (lambda x: ref_mod(x)), "module.reference.torch"
    elif implementation == "triton_transition_fused":
        from miniworld_engine.kernels import triton_transition_fused
        kfn = lambda x: triton_transition_fused(x, lw, lb, wa, wb, wsq, n, eps)  # noqa: E731
        path = "kernels.transition.triton.fused"
    elif implementation == "cute_transition_fused":
        from miniworld_engine.kernels import cute_transition_fused
        kfn = lambda x: cute_transition_fused(x, lw, lb, wa, wb, wsq, n, eps)  # noqa: E731
        path = "kernels.transition.cute.fused"
    elif implementation == "transition_b2b_ktiled":
        from miniworld_engine.kernels.transition.triton.fused import transition_b2b_ktiled
        kfn = lambda x: transition_b2b_ktiled(  # noqa: E731
            x.reshape(L * L, D), lw, lb, wa, wb, wsq, eps).reshape(1, L, L, D)
        path = "kernels.transition.triton.fused.b2b_ktiled"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_fwd(kfn(_x()), ref_mod(_x()))
    return _fwd_result(conf, kfn, (_x(),), acc=acc, path=path, ref="module.reference.torch", dtype="bfloat16")


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
        kfn = lambda x: triton_layernorm(x, w, b, eps)  # noqa: E731
        path = "kernels.layernorm.triton.main"
    elif implementation == "layernorm_dispatch":
        from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
        kfn = lambda x: layernorm_kernel(x, w, b, eps)  # noqa: E731
        path = "kernels.layernorm.interface"
    elif implementation == "quack_cute":
        from miniworld_engine.kernels.layernorm.cute.quack_adapter import quack_layernorm_fwd
        kfn = lambda x: quack_layernorm_fwd(x, w, b, eps)  # noqa: E731
        path = "kernels.layernorm.cute.quack"
    elif implementation == "triton_layernorm_lowreg":
        from miniworld_engine.kernels.layernorm.triton.lowreg import triton_layernorm_lowreg
        kfn = lambda x: triton_layernorm_lowreg(x, w, b, eps)  # noqa: E731
        path = "kernels.layernorm.triton.lowreg"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_fwd(kfn(_x()), ref(_x()))
    return _fwd_result(conf, kfn, (_x(),), acc=acc, path=path,
                       ref="torch.nn.functional.layer_norm", dtype=tname)


def bench_kernel_adaln(conf, seq_len, implementation, fabric):
    """Adaptive LayerNorm forward: y = sigma(scale)*LN(x) + bias, scale/bias = Linear(LN(cond)).
    Rows: pytorch, adaln_inference, triton_adaln, adaln_fused3."""
    from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
    from miniworld_engine.modules.exceptions import ImplementationType

    D, L = conf.d_pair, seq_len
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

    def _xc():
        torch.manual_seed(1)
        return (torch.randn(1, L, L, D, device=DEVICE, dtype=dtype),
                torch.randn(1, L, L, D, device=DEVICE, dtype=dtype))

    if implementation == "pytorch":
        kfn, path = (lambda x, c: ref_mod(x, c)), "module.reference.torch"
    elif implementation == "adaln_inference":
        from miniworld_engine.kernels.adaln.triton.inference import adaln_inference
        kfn = lambda x, c: adaln_inference(x, c, clw, sw, sb, bw, ex, ec)  # noqa: E731
        path = "kernels.adaln.triton.inference"
    elif implementation == "adaln_lnfold":
        from miniworld_engine.kernels.adaln.triton.inference import adaln_inference_lnfold
        from miniworld_engine.kernels.layernorm_linear.cute import fold_for_gemm
        _wcat = torch.cat([sw, bw], dim=0).contiguous()
        _bcat = torch.cat([sb, sb.new_zeros(D)], dim=0).contiguous()
        _pf = fold_for_gemm(_wcat, clw, clw.new_zeros(clw.shape), _bcat, w2_dtype=dtype)
        kfn = lambda x, c: adaln_inference_lnfold(  # noqa: E731
            x, c, clw, sw, sb, bw, ex, ec, weight_cat=_wcat, bias_cat=_bcat, prefolded=_pf)
        path = "kernels.adaln.triton.inference.lnfold"
    elif implementation == "triton_adaln":
        from miniworld_engine.kernels import triton_adaptive_layer_norm
        kfn = lambda x, c: triton_adaptive_layer_norm(x, c, clw, sw, sb, bw, ex, ec)  # noqa: E731
        path = "kernels.adaln.triton.main"
    elif implementation == "adaln_fused3":
        from miniworld_engine.kernels.adaln.triton.fused3 import adaln_fused3
        kfn = lambda x, c: adaln_fused3(x, c, clw, sw, sb, bw, ex, ec)  # noqa: E731
        path = "kernels.adaln.triton.fused3"
    else:
        return as_bench_result(float("nan"))

    xc, cc = _xc()
    acc = _acc_fwd(kfn(xc, cc), ref_mod(xc, cc))
    return _fwd_result(conf, kfn, _xc(), acc=acc, path=path, ref="module.reference.torch", dtype=tname)


def bench_kernel_tri_attn(conf, seq_len, implementation, fabric):
    """Triangle self-attention: softmax(QK^T*d^-0.5 + pair_bias)*V. q,k,v:(1,H,L,L,dh) bias:(1,H,L,L).
    Rows: pytorch(SDPA), triton_tri_attn, triton_tri_attn_miniworld(dep), triton_tri_attn_perf(dep)."""
    import torch.nn.functional as F

    L, dh = seq_len, 32
    H = max(1, conf.d_pair // dh)

    def mk():
        torch.manual_seed(1)
        q = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=BF16)
        k = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=BF16)
        v = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=BF16)
        bias = torch.randn(1, H, L, L, device=DEVICE, dtype=BF16)
        return q, k, v, bias

    def ref(q, k, v, bias):
        qf, kf, vf = (t.reshape(H, L, L, dh) for t in (q, k, v))
        mask = bias.reshape(H, 1, L, L)
        return F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask).reshape(1, H, L, L, dh)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch.sdpa"
    elif implementation == "triton_tri_attn":
        from miniworld_engine.kernels import triton_triangle_attention_pair_bias as fn
        kfn = lambda q, k, v, b: fn(q, k, v, b)  # noqa: E731
        path = "kernels.triangle_attention.triton.main"
    elif implementation == "triton_tri_attn_miniworld":
        from miniworld_engine.kernels.triangle_attention.triton.miniworld import (
            triton_triangle_attention_pair_bias as fn,
        )
        kfn = lambda q, k, v, b: fn(q, k, v, b)  # noqa: E731
        path = "kernels.triangle_attention.triton.miniworld"
    elif implementation == "triton_tri_attn_perf":
        from miniworld_engine.kernels.triangle_attention.triton.perf import (
            triton_triangle_attention_pair_bias as fn,
        )
        kfn = lambda q, k, v, b: fn(q, k, v, b)  # noqa: E731
        path = "kernels.triangle_attention.triton.perf"
    else:
        return as_bench_result(float("nan"))

    acc = {}
    try:
        qc = mk()
        acc = _acc_fwd(kfn(*qc), ref(*qc))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch.sdpa", dtype="bfloat16")


def bench_kernel_bias_attn(conf, seq_len, implementation, fabric):
    """Bias-only attention: out[i,j,d]=sum_k softmax_k(bias[j,k])*v[i,k,d]. v:(1,H,L,L,dh) bias:(1,H,L,L).
    Rows: pytorch, triton_bias_attn, bias_only_fused(dep, NEGATIVE RESULT)."""
    import torch.nn.functional as F

    L, dh = seq_len, 32
    H = max(1, conf.d_pair // dh)

    def mk():
        torch.manual_seed(1)
        v = torch.randn(1, H, L, L, dh, device=DEVICE, dtype=BF16)
        bias = torch.randn(1, H, L, L, device=DEVICE, dtype=BF16)
        return v, bias

    def ref(v, bias):
        p = F.softmax(bias.float(), dim=-1).to(v.dtype)
        return torch.einsum("bhjk,bhikd->bhijd", p, v)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch.einsum"
    elif implementation == "triton_bias_attn":
        from miniworld_engine.kernels import triton_bias_only_attention
        kfn = lambda v, b: triton_bias_only_attention(v, b)  # noqa: E731
        path = "kernels.bias_only_attention.triton.main"
    elif implementation == "bias_only_fused":
        from miniworld_engine.kernels.bias_only_attention.triton.fused import bias_only_fused_fwd
        kfn = lambda v, b: bias_only_fused_fwd(v, b)  # noqa: E731
        path = "kernels.bias_only_attention.triton.fused"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch.einsum", dtype="bfloat16")


def bench_kernel_aug_attn(conf, seq_len, implementation, fabric):
    """Augmented pair-bias attention: softmax(q.k*d^-0.5 + bias)*v. q,k,v:(A,1,L,H,dh) bias:(1,L,L,H).
    Rows: pytorch, triton_aug_attn, aug_attn_compute_efficient(dep)."""
    import torch.nn.functional as F

    L, A, H, dh = seq_len, 8, 4, 32

    def mk():
        torch.manual_seed(1)
        q = torch.randn(A, 1, L, H, dh, device=DEVICE, dtype=BF16)
        k = torch.randn(A, 1, L, H, dh, device=DEVICE, dtype=BF16)
        v = torch.randn(A, 1, L, H, dh, device=DEVICE, dtype=BF16)
        bias = torch.randn(1, L, L, H, device=DEVICE, dtype=BF16)
        return q, k, v, bias

    def ref(q, k, v, bias):
        att = torch.einsum("abihd,abjhd->abhij", q * (dh**-0.5), k)
        att = att + bias.permute(0, 3, 1, 2)[None]
        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        return torch.einsum("abhij,abjhd->abihd", att, v)

    if implementation == "pytorch":
        kfn, path = ref, "pytorch.einsum"
    elif implementation == "triton_aug_attn":
        from miniworld_engine.kernels import triton_augmented_attention_pair_bias
        # memory-efficient backend explicitly (the dispatch wrapper now defaults to
        # compute-efficient); keeps this row distinct from aug_attn_compute_efficient.
        kfn = lambda q, k, v, b: triton_augmented_attention_pair_bias(q, k, v, b, compute_efficient=False)  # noqa: E731
        path = "kernels.augmented_attention.triton.main"
    elif implementation == "aug_attn_compute_efficient":
        from miniworld_engine.kernels.augmented_attention.triton.compute_efficient import (
            triton_augmented_attention_pair_bias as fn,
        )
        kfn = lambda q, k, v, b: fn(q, k, v, b)  # noqa: E731
        path = "kernels.augmented_attention.triton.compute_efficient"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch.einsum", dtype="bfloat16")


def bench_kernel_ln_mask(conf, seq_len, implementation, fabric):
    """Fused LayerNorm+mask: out = LN(x)*mask (per-row scale). Rows: pytorch, fused_ln_mask."""
    import torch.nn.functional as F

    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    w = torch.randn(D, device=DEVICE, dtype=BF16)
    b = torch.randn(D, device=DEVICE, dtype=BF16)
    eps = 1e-5

    def mk():
        torch.manual_seed(1)
        x = torch.randn(1, L, L, D, device=DEVICE, dtype=BF16)
        mask = (torch.rand(1, L, L, device=DEVICE) > 0.1).to(BF16)
        return x, mask

    def ref(x, mask):
        return F.layer_norm(x, (D,), w, b, eps) * mask[..., None]

    if implementation == "pytorch":
        kfn, path = ref, "pytorch"
    elif implementation == "fused_ln_mask":
        from miniworld_engine.kernels.fused_ln_mask.triton.main import fused_ln_mask
        kfn = lambda x, m: fused_ln_mask(x, w, b, m, eps)  # noqa: E731
        path = "kernels.fused_ln_mask.triton"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch", dtype="bfloat16")


def bench_kernel_gemm_gate(conf, seq_len, implementation, fabric):
    """Gated output projection (tm2 back half): out = sigma(xg@Wg^T)*(xo@Wp^T). Rows: pytorch,
    tm2_cute, triton_tm2."""
    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    wg = (torch.randn(D, D, device=DEVICE, dtype=BF16) * (D**-0.5)).contiguous()  # (N,K)
    wp = (torch.randn(D, D, device=DEVICE, dtype=BF16) * (D**-0.5)).contiguous()

    def mk():
        torch.manual_seed(1)
        return (torch.randn(1, L, L, D, device=DEVICE, dtype=BF16),
                torch.randn(1, L, L, D, device=DEVICE, dtype=BF16))

    def ref(xg, xo):
        return torch.sigmoid(xg @ wg.t()) * (xo @ wp.t())

    if implementation == "pytorch":
        kfn, path = ref, "pytorch"
    elif implementation == "tm2_cute":
        # cuequiv wrapper deleted 2026-08-04 — this now benches OUR from-scratch tm2 kernel.
        from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import tm2_dual_from_scratch
        kfn = lambda xg, xo: tm2_dual_from_scratch(xg, xo, wg, wp)  # noqa: E731  (wg/wp are (N,K))
        path = "kernels.tm2.cute"
    elif implementation == "triton_tm2":
        from miniworld_engine.kernels.tm2.triton.main import triton_tm2
        wgt, wpt = wg.t().contiguous(), wp.t().contiguous()  # kernel computes x@W (K,N form)
        kfn = lambda xg, xo: triton_tm2(  # noqa: E731
            xg.reshape(L * L, D), xo.reshape(L * L, D), wgt, wpt).reshape(1, L, L, D)
        path = "kernels.tm2.triton.main"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_fwd(kfn(*mk()), ref(*mk()))
    return _fwd_result(conf, kfn, mk(), acc=acc, path=path, ref="pytorch", dtype="bfloat16")


def bench_kernel_cond_transition_tail(conf, seq_len, implementation, fabric):
    """Post-adaLN conditioned-transition tail: out=squeeze(silu(x@Wa)*(x@Wb)); y=sigma(cond@Wsc+b)*out.
    fp32. Rows: pytorch, triton_cond_transition."""
    from miniworld_engine import kernels
    from miniworld_engine.modules.conditioned_transition.module import ConditionedTransition
    from miniworld_engine.modules.exceptions import ImplementationType

    D, L, n = conf.d_pair, seq_len, 4
    torch.manual_seed(0)
    ref_mod = ConditionedTransition(D, D, n=n, implementation=ImplementationType.PYTORCH).to(DEVICE).float()
    for lin in (ref_mod.expand_a, ref_mod.expand_b, ref_mod.squeeze):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
    wa, wb, ws = ref_mod.expand_a.weight, ref_mod.expand_b.weight, ref_mod.squeeze.weight
    wsc, bsc = ref_mod.to_scale.weight, ref_mod.to_scale.bias

    def _xc():
        torch.manual_seed(1)
        return (torch.randn(1, L, L, D, device=DEVICE, dtype=torch.float32),
                torch.randn(1, L, L, D, device=DEVICE, dtype=torch.float32))

    if implementation == "pytorch":
        kfn, path = (lambda x, c: ref_mod(x, c)), "module.reference.torch"
    elif implementation == "triton_cond_transition":
        raw = kernels.cond_transition_inference_dispatch
        kfn = lambda x, c: raw(  # noqa: E731
            x.reshape(-1, D), c.reshape(-1, D), wa, wb, ws, wsc, bsc).reshape(1, L, L, D)
        path = "kernels.conditioned_transition.triton"
    else:
        return as_bench_result(float("nan"))

    xc, cc = _xc()
    acc = _acc_fwd(kfn(xc, cc), ref_mod(xc, cc))
    return _fwd_result(conf, kfn, _xc(), acc=acc, path=path, ref="module.reference.torch", dtype="float32")


# ---- BACKWARD operations (pure-function launchers; cudagraph-safe) ----------------------------
def bench_kernel_layernorm_bwd(conf, seq_len, implementation, fabric):
    """LayerNorm backward: (dy,x,w,mean,rstd)->(dx,dw,db). Rows: pytorch(pure), triton_atomic,
    triton_partial, triton_persistent. Cosine on dx vs the pure-torch LN backward."""
    D, L = conf.d_pair, seq_len
    dtype = torch.float32 if conf.precision == FP32_PRECISION else BF16
    tname = str(dtype).replace("torch.", "")
    torch.manual_seed(0)
    w = torch.randn(D, device=DEVICE, dtype=dtype)
    torch.manual_seed(1)
    x = torch.randn(L * L, D, device=DEVICE, dtype=dtype)
    dy = torch.randn(L * L, D, device=DEVICE, dtype=dtype)
    eps = 1e-5
    xf = x.float()
    mean = xf.mean(-1)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)

    def torch_bwd():
        xhat = (xf - mean[:, None]) * rstd[:, None]
        dxhat = dy.float() * w.float()
        dx = rstd[:, None] * (dxhat - dxhat.mean(-1, keepdim=True)
                              - xhat * (dxhat * xhat).mean(-1, keepdim=True))
        dwt = (dy.float() * xhat).sum(0)
        dbt = dy.float().sum(0)
        return dx.to(dtype), dwt.to(dtype), dbt.to(dtype)

    if implementation == "pytorch":
        kfn, path = torch_bwd, "pytorch"
    elif implementation in {"triton_atomic", "triton_partial", "triton_persistent"}:
        from miniworld_engine.kernels.layernorm.compile_native import (
            _bwd_atomic_impl, _bwd_partial_impl, _bwd_persistent_impl,
        )
        impl_fn = {"triton_atomic": _bwd_atomic_impl, "triton_partial": _bwd_partial_impl,
                   "triton_persistent": _bwd_persistent_impl}[implementation]
        kfn = lambda: impl_fn(dy, x, w, mean, rstd)  # noqa: E731
        path = f"kernels.layernorm.compile_native.{implementation}"
    elif implementation == "cuda":
        # Hand-CUDA vectorized backward; the shipped dispatch routes bf16 128<=N<=512 here.
        # Outside that gate the dispatch keeps triton, so report NaN (not applicable).
        if dtype is not BF16 or not (128 <= D <= 512):
            return as_bench_result(float("nan"))
        from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda
        kfn = lambda: layer_norm_bwd_cuda(dy, x, w, mean, rstd)  # noqa: E731
        path = "kernels.layernorm.cuda.layer_norm_bwd_cuda"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_grad(kfn()[0], torch_bwd()[0])
    return measured_result(
        conf=conf, func=kfn, grad_to_none=[], params=[], is_train=False,
        input_dtype=tname, parameter_dtype=tname, execution_path=path, reference="pytorch",
    )._replace(**acc)


def bench_kernel_gate_bwd(conf, seq_len, implementation, fabric):
    """Gate-elementwise backward: bwd of y=sigma(x_n@Wg)*proj -> (d_proj, dx_n, dWg). Rows: pytorch(pure),
    gate_elem_bwd. Cosine on concatenated grads."""
    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    wg = (torch.randn(D, D, device=DEVICE, dtype=BF16) * (D**-0.5)).contiguous()  # (K,N)
    torch.manual_seed(1)
    x_n = torch.randn(L * L, D, device=DEVICE, dtype=BF16)
    proj = torch.randn(L * L, D, device=DEVICE, dtype=BF16)
    dy = torch.randn(L * L, D, device=DEVICE, dtype=BF16)
    gate = torch.sigmoid(x_n.float() @ wg.float()).to(BF16)

    def torch_bwd():
        g, dyf, pf = gate.float(), dy.float(), proj.float()
        d_proj = dyf * g
        d_glog = dyf * pf * g * (1 - g)
        dx = d_glog @ wg.float().t()
        dwg = x_n.float().t() @ d_glog
        return d_proj.to(BF16), dx.to(BF16), dwg.to(BF16)

    if implementation == "pytorch":
        kfn, path = torch_bwd, "pytorch"
    elif implementation == "gate_elem_bwd":
        from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_bwd
        kfn = lambda: gate_elem_bwd(dy, x_n, proj, gate, wg)  # noqa: E731
        path = "kernels.trimul_inproj.triton.gate_elem"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_grad(_flat(list(kfn())), _flat(list(torch_bwd())))
    return measured_result(
        conf=conf, func=kfn, grad_to_none=[], params=[], is_train=False,
        input_dtype="bfloat16", parameter_dtype="bfloat16", execution_path=path, reference="pytorch",
    )._replace(**acc)


def bench_kernel_dual_gemm_epil_bwd(conf, seq_len, implementation, fabric):
    """Gated dual-GEMM front backward: (d_left,d_right)->dx_n + 4 weight grads. Rows: pytorch(pure),
    front_bwd_fused. Cosine on concatenated (dx_n|dWL|dWLg|dWR|dWRg)."""
    D, L, H = conf.d_pair, seq_len, conf.d_pair
    torch.manual_seed(0)

    def _w():
        return (torch.randn(D, H, device=DEVICE, dtype=BF16) * (D**-0.5)).contiguous()

    WL, WLg, WR, WRg = _w(), _w(), _w(), _w()
    torch.manual_seed(1)
    x_n = torch.randn(1, L, L, D, device=DEVICE, dtype=BF16)
    d_left = torch.randn(1, L, L, H, device=DEVICE, dtype=BF16)
    d_right = torch.randn(1, L, L, H, device=DEVICE, dtype=BF16)

    def torch_bwd():
        xf = x_n.reshape(L * L, D).float()
        dl, dr = d_left.reshape(L * L, H).float(), d_right.reshape(L * L, H).float()
        pL, gL = xf @ WL.float(), torch.sigmoid(xf @ WLg.float())
        pR, gR = xf @ WR.float(), torch.sigmoid(xf @ WRg.float())
        d_pL, d_gL = dl * gL, dl * pL * gL * (1 - gL)
        d_pR, d_gR = dr * gR, dr * pR * gR * (1 - gR)
        dxn = (d_pL @ WL.float().t() + d_gL @ WLg.float().t()
               + d_pR @ WR.float().t() + d_gR @ WRg.float().t())
        return (dxn.reshape(1, L, L, D).to(BF16), (xf.t() @ d_pL).to(BF16), (xf.t() @ d_gL).to(BF16),
                (xf.t() @ d_pR).to(BF16), (xf.t() @ d_gR).to(BF16))

    if implementation == "pytorch":
        kfn, path = torch_bwd, "pytorch"
    elif implementation == "front_bwd_fused":
        from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_fused
        xf = x_n.reshape(L * L, D)
        gLlog, pL = xf @ WLg, xf @ WL
        gRlog, pR = xf @ WRg, xf @ WR
        left_il = torch.stack([gLlog, pL], dim=-1).reshape(L * L, 2 * H)
        right_il = torch.stack([gRlog, pR], dim=-1).reshape(L * L, 2 * H)
        preact = torch.cat([left_il, right_il], dim=-1).reshape(
            1, L, L, 4 * H).permute(0, 3, 1, 2).contiguous()
        dlb = d_left.permute(0, 3, 1, 2).contiguous()
        drb = d_right.permute(0, 3, 1, 2).contiguous()
        kfn = lambda: front_bwd_fused(dlb, drb, preact, x_n, WL, WLg, WR, WRg)  # noqa: E731
        path = "kernels.trimul_inproj.triton.back_fused"
    else:
        return as_bench_result(float("nan"))

    acc = _acc_grad(_flat(list(kfn())), _flat(list(torch_bwd())))
    return measured_result(
        conf=conf, func=kfn, grad_to_none=[], params=[], is_train=False,
        input_dtype="bfloat16", parameter_dtype="bfloat16", execution_path=path, reference="pytorch",
    )._replace(**acc)


# ---- BACKWARD operations (autograd; backward-only timing via autograd.grad) -------------------
def bench_kernel_adaln_bwd(conf, seq_len, implementation, fabric):
    """adaLN backward (autograd, backward-only). Rows: pytorch, adaln_train, triton_adaln,
    adaln_fused3. Cosine on dx vs pytorch autograd."""
    from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
    from miniworld_engine.modules.exceptions import ImplementationType

    D, L = conf.d_pair, seq_len
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
    x0 = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    c0 = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    dy = torch.randn(1, L, L, D, device=DEVICE, dtype=dtype)
    xr, cr = x0.clone().requires_grad_(True), c0.clone().requires_grad_(True)
    ref_mod(xr, cr).backward(dy)
    ref_dx = xr.grad

    x, c = x0.clone().requires_grad_(True), c0.clone().requires_grad_(True)
    if implementation == "pytorch":
        out, path = ref_mod(x, c), "module.reference.torch"
    elif implementation == "adaln_train":
        from miniworld_engine.kernels.adaln.triton.training import adaln_train
        out = adaln_train(x, c, clw, sw, sb, bw, ex, ec)
        path = "kernels.adaln.triton.training"
    elif implementation == "triton_adaln":
        from miniworld_engine.kernels import triton_adaptive_layer_norm
        out = triton_adaptive_layer_norm(x, c, clw, sw, sb, bw, ex, ec)
        path = "kernels.adaln.triton.main"
    elif implementation == "adaln_fused3":
        from miniworld_engine.kernels.adaln.triton.fused3 import adaln_fused3_train
        out = adaln_fused3_train(x, c, clw, sw, sb, bw, ex, ec)
        path = "kernels.adaln.triton.fused3"
    else:
        return as_bench_result(float("nan"))
    return _bwd_autograd_result(conf, out, [x, c], dy, ref_dx, path=path,
                                ref="module.reference.torch", dtype=tname)


def bench_kernel_transition_b2b_bwd(conf, seq_len, implementation, fabric):
    """Transition backward (autograd, backward-only). Rows: pytorch, triton_transition_fused,
    cute_transition_fused. Cosine on dx vs pytorch autograd."""
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.transition import Transition

    D, L, n = conf.d_pair, seq_len, 4
    torch.manual_seed(0)
    ref_mod = Transition(D, n=n, implementation=ImplementationType.PYTORCH).to(DEVICE).to(BF16)
    for lin in (ref_mod.expand_a, ref_mod.expand_b, ref_mod.squeeze):
        torch.nn.init.normal_(lin.weight, std=D**-0.5)
    lw, lb = ref_mod.ln_in.weight, ref_mod.ln_in.bias
    wa, wb, wsq = ref_mod.expand_a.weight, ref_mod.expand_b.weight, ref_mod.squeeze.weight
    eps = ref_mod.ln_in.eps
    torch.manual_seed(1)
    x0 = torch.randn(1, L, L, D, device=DEVICE, dtype=BF16)
    dy = torch.randn(1, L, L, D, device=DEVICE, dtype=BF16)
    xr = x0.clone().requires_grad_(True)
    ref_mod(xr).backward(dy)
    ref_dx = xr.grad

    x = x0.clone().requires_grad_(True)
    if implementation == "pytorch":
        out, path = ref_mod(x), "module.reference.torch"
    elif implementation == "triton_transition_fused":
        from miniworld_engine.kernels import triton_transition_fused
        out = triton_transition_fused(x, lw, lb, wa, wb, wsq, n, eps)
        path = "kernels.transition.triton.fused"
    elif implementation == "cute_transition_fused":
        from miniworld_engine.kernels import cute_transition_fused
        out = cute_transition_fused(x, lw, lb, wa, wb, wsq, n, eps)
        path = "kernels.transition.cute.fused"
    else:
        return as_bench_result(float("nan"))
    return _bwd_autograd_result(conf, out, [x], dy, ref_dx, path=path,
                                ref="module.reference.torch", dtype="bfloat16")


def bench_kernel_gemm_epil_bwd(conf, seq_len, implementation, fabric):
    """LayerNorm+Linear backward (autograd, backward-only). Rows: pytorch, layernorm_linear_te,
    layernorm_linear_cute. Cosine on dx vs pytorch autograd."""
    import torch.nn.functional as F

    D, L = conf.d_pair, seq_len
    torch.manual_seed(0)
    lw = torch.randn(D, device=DEVICE, dtype=BF16)
    lb = torch.randn(D, device=DEVICE, dtype=BF16)
    w = (torch.randn(D, D, device=DEVICE, dtype=BF16) * (D**-0.5)).contiguous()
    eps = 1e-5
    torch.manual_seed(1)
    x0 = torch.randn(L * L, D, device=DEVICE, dtype=BF16)
    dy = torch.randn(L * L, D, device=DEVICE, dtype=BF16)
    xr = x0.clone().requires_grad_(True)
    F.linear(F.layer_norm(xr, (D,), lw, lb, eps), w).backward(dy)
    ref_dx = xr.grad

    x = x0.clone().requires_grad_(True)
    if implementation == "pytorch":
        out, path = F.linear(F.layer_norm(x, (D,), lw, lb, eps), w), "pytorch.autograd"
    elif implementation == "layernorm_linear_te":
        from miniworld_engine.kernels.layernorm_linear.te_style import layernorm_linear_te_fn
        out = layernorm_linear_te_fn(x, lw, lb, w, None, eps)
        path = "kernels.layernorm_linear.te_style"
    elif implementation == "layernorm_linear_cute":
        from miniworld_engine.kernels.layernorm_linear.autograd import layernorm_linear_fn
        out = layernorm_linear_fn(x, lw, lb, w, None, eps)
        path = "kernels.layernorm_linear.autograd.cute"
    else:
        return as_bench_result(float("nan"))
    return _bwd_autograd_result(conf, out, [x], dy, ref_dx, path=path,
                                ref="pytorch.autograd", dtype="bfloat16")


KERNEL_MAP = {
    # kernel function-operations (benchmarks/kernels/<op>/): forward
    "dual_gemm_epil": bench_kernel_dual_gemm_epil,
    "gemm_epil": bench_kernel_gemm_epil,
    "transition_b2b": bench_kernel_transition_b2b,
    "layernorm": bench_kernel_layernorm,
    "adaln": bench_kernel_adaln,
    "tri_attn": bench_kernel_tri_attn,
    "bias_attn": bench_kernel_bias_attn,
    "aug_attn": bench_kernel_aug_attn,
    "ln_mask": bench_kernel_ln_mask,
    "gemm_gate": bench_kernel_gemm_gate,
    "cond_transition_tail": bench_kernel_cond_transition_tail,
    # kernel function-operations: backward
    "layernorm_bwd": bench_kernel_layernorm_bwd,
    "gate_bwd": bench_kernel_gate_bwd,
    "dual_gemm_epil_bwd": bench_kernel_dual_gemm_epil_bwd,
    "adaln_bwd": bench_kernel_adaln_bwd,
    "transition_b2b_bwd": bench_kernel_transition_b2b_bwd,
    "gemm_epil_bwd": bench_kernel_gemm_epil_bwd,
    # module-level benches (benchmarks/modules/<mod>/)
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

_KERNELS_ROOT = _REPO_ROOT / "benchmarks" / "kernels"
_MODULES_ROOT = _REPO_ROOT / "benchmarks" / "modules"
_KERNEL_TARGETS = [
    "dual_gemm_epil", "gemm_epil", "transition_b2b", "layernorm", "adaln", "tri_attn",
    "bias_attn", "aug_attn", "ln_mask", "gemm_gate", "cond_transition_tail",
    "layernorm_bwd", "gate_bwd", "dual_gemm_epil_bwd", "adaln_bwd", "transition_b2b_bwd",
    "gemm_epil_bwd",
]
TARGET_DIRS = {name: _KERNELS_ROOT / name for name in _KERNEL_TARGETS}
TARGET_DIRS.update({
    "triangle_multiplication": _MODULES_ROOT / "triangle_multiplication",
    "triangle_multiplication_bidirectional": _MODULES_ROOT / "triangle_multiplication_bidirectional",
    "bias_only_attention": _MODULES_ROOT / "bias_only_attention",
    "triangle_attention": _MODULES_ROOT / "triangle_attention",
    "transition": _MODULES_ROOT / "transition",
    "conditioned_transition": _MODULES_ROOT / "conditioned_transition",
    "adaptive_layernorm": _MODULES_ROOT / "adaptive_layernorm",
    "augmented_attention_token": _MODULES_ROOT / "augmented_attention",
    "augmented_attention_atom": _MODULES_ROOT / "augmented_attention",
})

# Triton autotuner objects live in the per-op `triton/main.py` of each kernel.
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
    "bias_only_attention": [
        "miniworld_engine.kernels.layernorm.triton.main",
        "miniworld_engine.kernels.layernorm_linear.triton.main",
        "miniworld_engine.kernels.bias_only_attention.triton.gate_out",
        "miniworld_engine.kernels.bias_only_attention.triton.main",
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
        try:
            spec = parse_implementation_spec(implementation)
        except ValueError:
            spec = None  # kernel-bench variant label, not a module ImplementationType
    implementation_type = spec.impl.value if spec is not None else implementation
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
        "compiled": actual_compiled_flag(conf),
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
        def setup_module(module):  # noqa: ANN001, ANN205
            return module

        @staticmethod
        def backward(tensor, gradient):  # noqa: ANN001, ANN205
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
        _settings.configure(**_pins)
        print("  [capture] full grid unlocked; "
              + ", ".join(f"{k.removeprefix('pin_')}={v}" for k, v in _pins.items()), flush=True)
        _capture.install()

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
    tmp_csv_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
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
    with tmp_csv_path.open("w", newline="", encoding="ascii") as handle:
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
    tmp_csv_path.replace(csv_path)
    print(f"\nwrote {csv_path}")

    if _capture_on:
        print("\n[autotune-capture] captured configs:")
        print(_capture.precompile_summary())
        print(_capture.summary())
        shard = getattr(conf, "autotune_shard", "") or ""
        if shard:
            # Dedicated parallel builder (submits/build_autotune_cache.py): dump this run's
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
