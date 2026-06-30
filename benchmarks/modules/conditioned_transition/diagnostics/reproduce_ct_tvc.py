"""Reproduce the fast ConditionedTransition training log from ct_tvc_9939/9941.

This is intentionally a diagnostic, not a final benchmark runner. It preserves the
old apples-to-apples regime:

* fp32 tensors, TF32 matmuls
* manual CUDA graph timing for MiniWorld and eager PyTorch
* torch.compile(..., mode="reduce-overhead") for the compiled PyTorch column
* fresh tensor sets for graph capture to avoid AccumulateGrad stream poisoning

Historical reference:
benchmarks/modules/transition/artifacts/repro_71fee1c_tree/_ct_tmp/ct_tvc_9939.out
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.conditioned_transition.triton.training import (
    cond_transition_train,
    set_forward_mode,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except Exception:
    pass

TensorTuple = tuple[torch.Tensor, ...]

STREAMS = (
    ("atom", 128, (2048, 4096, 8192)),
    ("token", 768, (384, 512, 768, 1024)),
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.double().reshape(-1)
    b64 = b.double().reshape(-1)
    return (a64 @ b64 / (a64.norm() * b64.norm() + 1e-12)).item()


def reference(
    x: torch.Tensor,
    cond: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    wsc: torch.Tensor,
    bsc: torch.Tensor,
) -> torch.Tensor:
    a = x @ wa.t()
    b = x @ wb.t()
    h = F.silu(a) * b
    out = h @ ws.t()
    scale = cond @ wsc.t() + bsc
    return torch.sigmoid(scale) * out


def make_tensors(
    m: int,
    d_hidden: int,
    *,
    n: int = 2,
    d_cond: int = 384,
    device: str = "cuda",
) -> TensorTuple:
    generator = torch.Generator(device).manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=generator)

    tensors = (
        randn(m, d_hidden),
        randn(m, d_cond),
        randn(n * d_hidden, d_hidden) / d_hidden**0.5,
        randn(n * d_hidden, d_hidden) / d_hidden**0.5,
        randn(d_hidden, n * d_hidden) / (n * d_hidden) ** 0.5,
        randn(d_hidden, d_cond) / d_cond**0.5,
        torch.full((d_hidden,), -2.0, device=device),
    )
    return tuple(t.detach().requires_grad_(True) for t in tensors)


def fwd_bwd(fn: Callable[..., torch.Tensor], tensors: TensorTuple) -> TensorTuple:
    y = fn(*tensors)
    return torch.autograd.grad(y, tensors, torch.ones_like(y))


def graph_bench(make_fn: Callable[[], TensorTuple], fn: Callable[..., torch.Tensor], *, iters: int, warmups: int) -> float:
    tensors = make_fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fwd_bwd(fn, tensors)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fwd_bwd(fn, tensors)
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


def compile_bench(
    compiled_fn: Callable[..., torch.Tensor],
    make_fn: Callable[[], TensorTuple],
    *,
    iters: int,
    warmups: int,
) -> float:
    tensors = make_fn()

    def call() -> TensorTuple:
        y = compiled_fn(*tensors)
        return torch.autograd.grad(y, tensors, torch.ones_like(y))

    for _ in range(warmups):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


def run(args: argparse.Namespace) -> list[dict[str, str | float | int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    set_forward_mode("auto")
    compiled_reference = torch.compile(reference, mode="reduce-overhead")
    rows: list[dict[str, str | float | int]] = []

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    print("\n=== TRAINING fwd+bwd: ours(auto) vs torch.compile (CUDA graph) ===")
    print(
        f"{'stream':>6} {'M':>6} {'d':>4} | {'cos_min':>8} | "
        f"{'ours_us':>8} {'compile_us':>10} {'eager_us':>8} | "
        f"{'vs_compile':>10} {'vs_eager':>8}"
    )

    for stream_name, d_hidden, m_values in STREAMS:
        for m in m_values:
            make_fn = lambda m=m, d_hidden=d_hidden: make_tensors(m, d_hidden)

            correctness_tensors = make_fn()
            y_ref = reference(*correctness_tensors)
            grad_ref = torch.autograd.grad(
                y_ref,
                correctness_tensors,
                torch.ones_like(y_ref),
            )
            grad_ours = fwd_bwd(cond_transition_train, correctness_tensors)
            y_ours = cond_transition_train(*correctness_tensors)
            cos_min = min(
                [cosine(y_ours, y_ref)]
                + [cosine(actual, expected) for actual, expected in zip(grad_ours, grad_ref)]
            )
            del correctness_tensors, y_ref, grad_ref, grad_ours, y_ours
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            ours_us = graph_bench(make_fn, cond_transition_train, iters=args.iters, warmups=args.warmups)
            eager_us = graph_bench(make_fn, reference, iters=args.iters, warmups=args.warmups)
            compile_us = compile_bench(
                compiled_reference,
                make_fn,
                iters=args.iters,
                warmups=args.compile_warmups,
            )
            vs_compile = compile_us / ours_us
            vs_eager = eager_us / ours_us
            print(
                f"{stream_name:>6} {m:6d} {d_hidden:4d} | {cos_min:8.5f} | "
                f"{ours_us:8.1f} {compile_us:10.1f} {eager_us:8.1f} | "
                f"{vs_compile:9.2f}x {vs_eager:7.2f}x"
            )
            rows.append(
                {
                    "stream": stream_name,
                    "m": m,
                    "d_hidden": d_hidden,
                    "d_cond": 384,
                    "n": 2,
                    "mode": "training",
                    "dtype": "float32",
                    "precision": "tf32",
                    "cudagraph": "manual",
                    "forward_mode": "auto",
                    "cos_min": cos_min,
                    "ours_us": ours_us,
                    "compile_us": compile_us,
                    "eager_us": eager_us,
                    "vs_compile": vs_compile,
                    "vs_eager": vs_eager,
                }
            )
    return rows


def write_csv(rows: list[dict[str, str | float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--compile-warmups", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/modules/conditioned_transition/artifacts/"
            "conditioned_transition_reproduce_ct_tvc_training.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    write_csv(rows, args.output)
    print("DONE")


if __name__ == "__main__":
    main()
