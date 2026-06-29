# vendored + trimmed from team-gm psk/benchmark : benchmarks/runners/bench.py
# Single bench entry for miniworld-kernels. Drops the model-level
# Pairformer / DiffusionTransformer benches; keeps the kernel-wrapping layers.
import importlib
import sys
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
    ImplementationType,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_kernels.viz import style_for

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

    kernel: str
    implementations: list[str] = [
        ImplementationType.PYTORCH.value,
        ImplementationType.TRITON.value,
    ]
    mode: Literal["inference", "training"]
    metric: Literal["time", "memory"]
    compile: bool = False
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


def parse_implementation_spec(raw: str) -> ImplementationSpec:
    key = raw.strip().lower()
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


def bench_triangle_multiplication(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = parse_implementation_spec(implementation)

    class MultiTriangleMultiplication(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    TriangleMultiplication(
                        conf.d_pair,
                        implementation=spec.impl,
                        ln_implementation=spec.ln_impl or ImplementationType.PYTORCH,
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

    model = MultiTriangleMultiplication().to(DEVICE)
    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair).to(DEVICE)
    dy = torch.randn_like(pair)
    pair.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def forward() -> torch.Tensor:
        return model(pair, mask)

    def full() -> None:
        y = forward()
        fabric.backward(y, dy)

    func = forward if is_inference_mode(conf.mode) else full
    if conf.metric == "time":
        return bench_time(func, grad_to_none=[pair])["median_ms"]
    return bench_memory(func)["median_mb"]


def bench_triangle_attention(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = parse_implementation_spec(implementation)

    class MultiTriangleAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    TriangleAttention(
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

    model = MultiTriangleAttention().to(DEVICE)
    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair).to(DEVICE)
    dy = torch.randn_like(pair)
    pair.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def forward() -> torch.Tensor:
        return model(pair, mask)

    def full() -> None:
        y = forward()
        fabric.backward(y, dy)

    func = forward if is_inference_mode(conf.mode) else full
    if conf.metric == "time":
        return bench_time(func, grad_to_none=[pair])["median_ms"]
    return bench_memory(func)["median_mb"]


def bench_transition(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = parse_implementation_spec(implementation)

    class MultiTransition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    Transition(conf.d_pair, implementation=spec.impl)
                    for _ in range(conf.n_layers)
                ],
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    model = MultiTransition().to(DEVICE)
    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    x = torch.randn(1, seq_len, seq_len, conf.d_pair).to(DEVICE).to(torch.bfloat16)
    dy = torch.randn_like(x)
    x.requires_grad = True

    def forward() -> torch.Tensor:
        return model(x)

    def full() -> None:
        y = forward()
        fabric.backward(y, dy)

    func = forward if is_inference_mode(conf.mode) else full
    if conf.metric == "time":
        return bench_time(func, grad_to_none=[x])["median_ms"]
    return bench_memory(func)["median_mb"]


def bench_adaptive_layernorm(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = parse_implementation_spec(implementation)
    implementation_type = spec.impl
    if implementation_type not in {
        ImplementationType.PYTORCH,
        ImplementationType.TRITON,
    }:
        return float("nan")

    dtype = torch.float32 if conf.precision == FP32_PRECISION else torch.bfloat16

    class MultiAdaptiveLayerNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    AdaptiveLayerNorm(
                        d_hidden=conf.d_single_token,
                        d_cond=conf.d_single_token,
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
        conf.d_single_token,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    cond = torch.randn(
        conf.n_augment,
        1,
        seq_len,
        conf.d_single_token,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    dy = torch.randn_like(x)

    def forward() -> torch.Tensor:
        return model(x, cond)

    def full() -> None:
        y = forward()
        fabric.backward(y, dy)

    func = forward if is_inference_mode(conf.mode) else full
    try:
        if conf.metric == "time":
            return bench_time(func, grad_to_none=[x, cond])["median_ms"]
        return bench_memory(func)["median_mb"]
    except torch.cuda.OutOfMemoryError:
        return float("nan")


def bench_augmented_attention_token(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = parse_implementation_spec(implementation)
    model = AugmentedAttentionPairBias(
        d_single=conf.d_single_token,
        d_cond=conf.d_single_token,
        d_pair=conf.d_pair,
        n_head=16,
        implementation=spec.impl,
    ).to(DEVICE)

    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    pair = torch.randn(1, seq_len, seq_len, conf.d_pair).to(DEVICE)
    single = torch.randn(conf.n_augment, 1, seq_len, conf.d_single_token).to(DEVICE)
    cond = torch.randn(conf.n_augment, 1, seq_len, conf.d_single_token).to(DEVICE)
    dy_single = torch.randn_like(single)
    pair.requires_grad = True
    single.requires_grad = True
    cond.requires_grad = True
    mask = torch.rand(1, seq_len, device=DEVICE) > conf.mask_prob

    def forward() -> torch.Tensor:
        return model(single, cond, pair, mask)

    def full() -> None:
        out_single = forward()
        fabric.backward(out_single, dy_single)

    func = forward if is_inference_mode(conf.mode) else full
    if conf.metric == "time":
        return bench_time(func, grad_to_none=[pair, single, cond])["median_ms"]
    return bench_memory(func)["median_mb"]


def bench_augmented_attention_atom(
    conf: BenchConfig,
    seq_len: int,
    implementation: str,
    fabric: Fabric,
):
    spec = parse_implementation_spec(implementation)
    model = AugmentedAttentionPairBias(
        d_single=conf.d_single_atom,
        d_cond=conf.d_single_atom,
        d_pair=conf.d_pair_atom,
        n_head=4,
        implementation=spec.impl,
    ).to(DEVICE)

    if conf.compile:
        model.compile()
    model = fabric.setup_module(model)

    atom_len = seq_len * 8
    pair = torch.randn(1, atom_len, atom_len, conf.d_pair_atom).to(DEVICE)
    single = torch.randn(conf.n_augment, 1, atom_len, conf.d_single_atom).to(DEVICE)
    cond = torch.randn(conf.n_augment, 1, atom_len, conf.d_single_atom).to(DEVICE)
    dy_single = torch.randn_like(single)
    pair.requires_grad = True
    single.requires_grad = True
    cond.requires_grad = True
    mask = torch.rand(1, atom_len, device=DEVICE) > conf.mask_prob

    def forward() -> torch.Tensor:
        return model(single, cond, pair, mask)

    def full() -> None:
        out_single = forward()
        fabric.backward(out_single, dy_single)

    func = forward if is_inference_mode(conf.mode) else full
    try:
        if conf.metric == "time":
            return bench_time(func, grad_to_none=[pair, single, cond])["median_ms"]
        return bench_memory(func)["median_mb"]
    except torch.cuda.OutOfMemoryError:
        return float("nan")


KERNEL_MAP = {
    "triangle_multiplication": bench_triangle_multiplication,
    "triangle_attention": bench_triangle_attention,
    "transition": bench_transition,
    "adaptive_layernorm": bench_adaptive_layernorm,
    "augmented_attention_token": bench_augmented_attention_token,
    "augmented_attention_atom": bench_augmented_attention_atom,
}

TARGET_DIRS = {
    "triangle_multiplication": _REPO_ROOT / "benchmarks" / "modules" / "triangle_multiplication",
    "triangle_attention": _REPO_ROOT / "benchmarks" / "modules" / "triangle_attention",
    "transition": _REPO_ROOT / "benchmarks" / "modules" / "transition",
    "adaptive_layernorm": _REPO_ROOT / "benchmarks" / "modules" / "adaptive_layernorm",
    "augmented_attention_token": _REPO_ROOT / "benchmarks" / "modules" / "augmented_attention",
    "augmented_attention_atom": _REPO_ROOT / "benchmarks" / "modules" / "augmented_attention",
}

# Per-implementation (colour, linestyle) come from the repo-wide canonical
# palette (miniworld_kernels.viz) so these line plots match the grouped-bar
# figures from benchmarks/runners/plot_bench.py — same backend, same colour,
# every figure.
def impl_style(impl: str):  # noqa: ANN201 - matplotlib (color, linestyle)
    return style_for(impl)

# Triton autotuner objects live in the per-op `triton/main.py` of each kernel.
AUTOTUNE_MODULES = {
    "triangle_multiplication": [
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.tm1.triton.main",
        "miniworld_kernels.kernels.tm2.triton.main",
    ],
    "triangle_attention": [
        "miniworld_kernels.kernels.triangle_attention.triton.main",
    ],
    "transition": [
        "miniworld_kernels.kernels.layernorm.triton.main",
        "miniworld_kernels.kernels.transition.triton.main",
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
    for autotuner_name in sorted(seen_autotuners):
        lines = [autotuner_name]
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


@hydra.main(
    config_path="../modules/triangle_multiplication/configs",
    config_name="bench",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    conf = BenchConfig.model_validate(cfg)
    if not conf.compile:
        msg = "Final benchmarks must run compiled. Use compile=true."
        raise ValueError(msg)
    bench_func = KERNEL_MAP[conf.kernel]

    torch.backends.cuda.matmul.allow_tf32 = conf.allow_tf32
    fabric = Fabric(accelerator="cuda", devices=1)
    fabric.launch()

    ylabel = "Time (ms)" if conf.metric == "time" else "Memory (MB)"
    bench_args = [
        conf.kernel,
        f"n_layers={conf.n_layers}",
        mode_label(conf.mode),
        conf.metric,
        str(conf.precision),
    ]
    if conf.compile:
        bench_args.append("compile")
    if conf.name_suffix:
        bench_args.append(conf.name_suffix)

    bench_config = triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=list(range(conf.min_seq_len, conf.max_seq_len + 1, conf.seq_len_step)),
        line_arg="implementation",
        line_vals=conf.implementations,
        line_names=conf.implementations,
        styles=[impl_style(impl) for impl in conf.implementations],
        ylabel=ylabel,
        plot_name="_".join(bench_args),
        args={"conf": conf, "fabric": fabric},
    )

    gpu_name = torch.cuda.get_device_name(0)
    results_dir = TARGET_DIRS[conf.kernel] / "artifacts" / gpu_name
    results_dir.mkdir(parents=True, exist_ok=True)
    autotune_cache_records: dict[str, dict[tuple, triton.Config]] = {}
    autotune_single_config_records: dict[str, triton.Config] = {}
    seen_autotuners: set[str] = set()

    @triton.testing.perf_report([bench_config])
    def bench_with_reset(**kwargs):  # noqa: ANN003, ANN202
        torch._dynamo.reset()  # noqa: SLF001
        torch.cuda.empty_cache()
        result = bench_func(**kwargs)
        capture_autotune_state(
            conf.kernel,
            autotune_cache_records,
            autotune_single_config_records,
            seen_autotuners,
        )
        return result

    bench_with_reset.run(
        print_data=True,
        save_path=results_dir,
    )

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
    (results_dir / "autotune_summary.txt").write_text(
        f"{autotune_summary}\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
