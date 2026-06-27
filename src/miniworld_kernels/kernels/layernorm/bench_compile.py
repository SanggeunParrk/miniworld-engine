"""LayerNorm compile benchmark for our method variants.

Compares eager vs torch.compile(reduce-overhead) for:
- Triton atomic
- Triton partial
- Auto dispatch

Uses the standard team-gm parseable output so scripts/plot_bench.py can render
the canonical markdown + plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]

import torch
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.layernorm.compile_native import (
    layernorm_atomic_compile,
    layernorm_dispatch_compile,
    layernorm_partial_compile,
)
from miniworld_kernels.kernels.layernorm.reference import layernorm_pytorch

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
EPS = 1e-5
DEFAULT_L_LIST = [384, 512, 768, 1024]
DEFAULT_D_LIST = [128, 256, 384, 512, 768]


def do_bench(fn, *, grad_to_none: list[torch.Tensor] | None = None) -> float:
    return triton.testing.do_bench(
        fn,
        warmup=25,
        rep=100,
        quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grad_to_none or [],
    )[0]


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    abs_err = (a32 - b32).abs().max().item()
    rel_fro = ((a32 - b32).norm() / (b32.norm() + 1e-12)).item()
    cos = F.cosine_similarity(a32, b32, dim=0).item()
    return abs_err, rel_fro, cos


def compare_line(
    tag: str,
    y: torch.Tensor,
    y_ref: torch.Tensor,
    dx: torch.Tensor,
    dx_ref: torch.Tensor,
    dw: torch.Tensor,
    dw_ref: torch.Tensor,
    db: torch.Tensor,
    db_ref: torch.Tensor,
) -> str:
    ya, yr, yc = metrics(y, y_ref)
    dxa, dxr, dxc = metrics(dx, dx_ref)
    dwa, dwr, dwc = metrics(dw, dw_ref)
    dba, dbr, dbc = metrics(db, db_ref)
    return (
        f"{tag} "
        f"fwd(abs={ya:.3e},rel={yr:.3e},cos={yc:.6f}) "
        f"dx(abs={dxa:.3e},rel={dxr:.3e},cos={dxc:.6f}) "
        f"dw(abs={dwa:.3e},rel={dwr:.3e},cos={dwc:.6f}) "
        f"db(abs={dba:.3e},rel={dbr:.3e},cos={dbc:.6f})"
    )


def warm_compile(fn, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, dy: torch.Tensor):
    cfn = torch.compile(fn, mode="reduce-overhead")
    y = cfn(x, w, b, EPS)
    y.backward(dy)
    torch.cuda.synchronize()
    x.grad = None
    w.grad = None
    b.grad = None
    return cfn


def run_impl(
    tag: str,
    fn,
    x0: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    dy: torch.Tensor,
    y_ref: torch.Tensor,
    dx_ref: torch.Tensor,
    dw_ref: torch.Tensor,
    db_ref: torch.Tensor,
    *,
    compile_mode: bool,
) -> None:
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)

    bench_fn = fn
    if compile_mode:
        bench_fn = warm_compile(fn, x, w, b, dy)

    y = bench_fn(x, w, b, EPS)
    y.backward(dy)
    torch.cuda.synchronize()
    print(
        compare_line(
            tag,
            y.detach(),
            y_ref,
            x.grad.detach(),
            dx_ref,
            w.grad.detach(),
            dw_ref,
            b.grad.detach(),
            db_ref,
        ),
        flush=True,
    )

    x.grad = None
    w.grad = None
    b.grad = None
    t_fwd = do_bench(lambda: bench_fn(x, w, b, EPS))
    t_full = do_bench(lambda: bench_fn(x, w, b, EPS).backward(dy), grad_to_none=[x, w, b])
    print(f"{tag} fwd={t_fwd:.4f} ms fwd+bwd={t_full:.4f} ms", flush=True)


def parse_int_list(text: str) -> list[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LayerNorm compile benchmark")
    p.add_argument("--l-values", default=",".join(str(v) for v in DEFAULT_L_LIST))
    p.add_argument("--d-values", default=",".join(str(v) for v in DEFAULT_D_LIST))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    l_list = parse_int_list(args.l_values)
    d_list = parse_int_list(args.d_values)

    print(f"host={torch.cuda.get_device_name(0)} torch={torch.__version__} dtype={DTYPE}", flush=True)
    print(
        "implementations=pytorch,triton_atomic,triton_atomic_compile,"
        "triton_partial,triton_partial_compile,"
        "layernorm_dispatch,layernorm_dispatch_compile  "
        f"sweep=M=L^2 for L in {l_list}, D in {d_list}  eps={EPS}",
        flush=True,
    )

    methods = [
        ("triton_atomic", layernorm_atomic_compile, False),
        ("triton_atomic_compile", layernorm_atomic_compile, True),
        ("triton_partial", layernorm_partial_compile, False),
        ("triton_partial_compile", layernorm_partial_compile, True),
        ("layernorm_dispatch", layernorm_dispatch_compile, False),
        ("layernorm_dispatch_compile", layernorm_dispatch_compile, True),
    ]

    for d_model in d_list:
        for l in l_list:
            m = l * l
            print(f"\n=== M={m}  d_in={d_model}  d_out={d_model} ===", flush=True)
            print(f"# L={l} D={d_model}", flush=True)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.cuda.empty_cache()
            try:
                x0 = torch.randn(m, d_model, device=DEVICE, dtype=DTYPE)
                w0 = torch.randn(d_model, device=DEVICE, dtype=DTYPE)
                b0 = torch.randn(d_model, device=DEVICE, dtype=DTYPE)
                dy = torch.randn(m, d_model, device=DEVICE, dtype=DTYPE)

                xp = x0.detach().clone().requires_grad_(True)
                wp = w0.detach().clone().requires_grad_(True)
                bp = b0.detach().clone().requires_grad_(True)
                yp = layernorm_pytorch(xp, wp, bp, EPS)
                yp.backward(dy)
                torch.cuda.synchronize()
                y_ref = yp.detach().clone()
                dx_ref = xp.grad.detach().clone()
                dw_ref = wp.grad.detach().clone()
                db_ref = bp.grad.detach().clone()

                print(compare_line("pytorch", y_ref, y_ref, dx_ref, dx_ref, dw_ref, dw_ref, db_ref, db_ref), flush=True)
                t_pt_fwd = do_bench(lambda: layernorm_pytorch(xp, wp, bp, EPS))
                t_pt_full = do_bench(lambda: layernorm_pytorch(xp, wp, bp, EPS).backward(dy), grad_to_none=[xp, wp, bp])
                print(f"pytorch fwd={t_pt_fwd:.4f} ms fwd+bwd={t_pt_full:.4f} ms", flush=True)

                for tag, fn, compile_mode in methods:
                    run_impl(tag, fn, x0, w0, b0, dy, y_ref, dx_ref, dw_ref, db_ref, compile_mode=compile_mode)

                torch._dynamo.reset()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(f"# skipped: CUDA OOM for M={m}, D={d_model}: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            finally:
                torch.cuda.empty_cache()

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

