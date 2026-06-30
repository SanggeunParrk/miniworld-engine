"""Reproduce the historical fast Transition module log.

This restores the runner that produced:
``benchmarks/modules/transition/artifacts/repro_71fee1c_tree/_trans_tmp/bench_module.out``.

It is intentionally a diagnostic, not the canonical CSV benchmark runner. The historical
regime used ``triton.testing.do_bench`` directly on the real ``Transition`` module with
pair-shaped bf16 input ``(1, L, L, d)``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import triton.testing as tt

from miniworld_kernels import kernels
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.transition import Transition

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMPLS = (
    ("PyTorch", "pytorch", "module:pytorch"),
    ("Triton (prev)", "triton", "triton_prev"),
    ("Triton b2b (ours)", "miniworld", "module:triton"),
    ("cute (ours)", "miniworld-alt", "module:cute"),
)
LSEQ = (384, 512, 640, 768, 896, 1024)


def make_x(seq_len: int, d_hidden: int, *, grad: bool) -> torch.Tensor:
    x = torch.randn(1, seq_len, seq_len, d_hidden, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_(grad)
    return x


def make_callable(kind: str, d_hidden: int):
    if kind == "module:pytorch":
        module = Transition(
            d_hidden=d_hidden,
            n=4,
            implementation=ImplementationType.PYTORCH,
        ).cuda()
        return lambda x: module(x)
    if kind == "module:triton":
        module = Transition(
            d_hidden=d_hidden,
            n=4,
            implementation=ImplementationType.TRITON,
        ).cuda()
        return lambda x: module(x)
    if kind == "module:cute":
        module = Transition(
            d_hidden=d_hidden,
            n=4,
            implementation=ImplementationType.CUTE,
        ).cuda()
        return lambda x: module(x)
    if kind == "triton_prev":
        module = Transition(
            d_hidden=d_hidden,
            n=4,
            implementation=ImplementationType.PYTORCH,
        ).cuda()
        wa = module.expand_a.weight
        wb = module.expand_b.weight
        ws = module.squeeze.weight
        n = module.n
        return lambda x: kernels.triton_transition(module.ln_in(x), wa, wb, ws, n)
    raise ValueError(kind)


def time_ms(fn) -> float:
    try:
        return tt.do_bench(fn, warmup=25, rep=100, return_mode="median")
    except Exception as exc:
        print(f"   [skip] {type(exc).__name__}: {str(exc)[:140]}")
        return float("nan")


def bench_l_sweep(d_hidden: int, mode: str) -> dict[str, dict[int, float]]:
    results: dict[str, dict[int, float]] = {}
    for label, ident, kind in IMPLS:
        call = make_callable(kind, d_hidden)
        row: dict[int, float] = {}
        for seq_len in LSEQ:
            grad = mode == "full"
            x = make_x(seq_len, d_hidden, grad=grad)
            dy = torch.randn_like(x) if grad else None

            def fwd(x=x):
                return call(x)

            def full(x=x, dy=dy):
                if x.grad is not None:
                    x.grad = None
                call(x).backward(dy)

            row[seq_len] = time_ms(full if grad else fwd)
            del x, dy
            torch.cuda.empty_cache()
        results[ident] = row
        values = " ".join(f"{row[seq_len]:8.4f}" for seq_len in LSEQ)
        print(f" d={d_hidden} {mode:7} {label:20} {values}")
    return results


def write_l_sweep_csv(path: Path, results: dict[str, dict[int, float]]) -> None:
    idents = [ident for _, ident, _ in IMPLS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq_len"] + [f"{label}|{ident}" for label, ident, _ in IMPLS])
        for seq_len in LSEQ:
            writer.writerow([seq_len] + [f"{results[ident][seq_len]:.4f}" for ident in idents])
    print(f"  wrote {path}")


def bench_d_crossover(seq_len: int) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    print(f"\n=== d-crossover (forward, L={seq_len}, M={seq_len * seq_len}) — why the dispatch exists ===")
    print(f"{'d':>5} " + " ".join(f"{label:>20}" for label, _, _ in IMPLS))
    for d_hidden in (128, 256, 512):
        cells: dict[str, float] = {}
        for _label, ident, kind in IMPLS:
            call = make_callable(kind, d_hidden)
            x = make_x(seq_len, d_hidden, grad=False)
            cells[ident] = time_ms(lambda x=x: call(x))
            del x
            torch.cuda.empty_cache()
        print(f"{d_hidden:>5} " + " ".join(f"{cells[ident]:>20.4f}" for _, ident, _ in IMPLS))
        for label, ident, kind in IMPLS:
            rows.append(
                {
                    "seq_len": seq_len,
                    "d_hidden": d_hidden,
                    "implementation": label,
                    "identity": ident,
                    "kind": kind,
                    "latency_ms": cells[ident],
                }
            )
    return rows


def write_d_crossover_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/modules/transition/artifacts/transition_module_repro"),
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")

    print("\n=== Transition module, d=128, n=4, bf16 (median ms) ===")
    forward = bench_l_sweep(128, "forward")
    write_l_sweep_csv(args.output_dir / "transition_module_forward.csv", forward)

    full = bench_l_sweep(128, "full")
    write_l_sweep_csv(args.output_dir / "transition_module_fwd_bwd.csv", full)

    d_crossover = bench_d_crossover(512)
    write_d_crossover_csv(args.output_dir / "transition_module_d_crossover_forward.csv", d_crossover)
    print("DONE")


if __name__ == "__main__":
    main()
