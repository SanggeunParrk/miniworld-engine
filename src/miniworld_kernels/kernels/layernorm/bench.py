"""Standalone LayerNorm benchmark.

Benchmarks the current baselines:
- PyTorch `F.layer_norm`
- legacy vendored Triton `triton_layernorm`
- cuequivariance `layer_norm_transpose(..., layout="nd->nd")`

Once `layernorm_kernel()` is implemented, the same script will benchmark it too.
Run this on a compute node only.
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
from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose

from miniworld_kernels.kernels.layernorm.interface import layernorm_kernel
from miniworld_kernels.kernels.layernorm.reference import layernorm_pytorch
from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
EPS = 1e-5
DEFAULT_L_LIST = [384, 512, 768, 1024]
DEFAULT_D_LIST = [128, 256, 384, 512, 768]


def do_bench(fn, *, grad_to_none: list[torch.Tensor] | None = None) -> float:
    """Median milliseconds via the repo's Triton benchmarking convention."""
    return triton.testing.do_bench(
        fn,
        warmup=25,
        rep=100,
        quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grad_to_none or [],
    )[0]


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Return max abs error, relative Frobenius error, and cosine similarity."""
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    abs_err = (a32 - b32).abs().max().item()
    rel_fro = ((a32 - b32).norm() / (b32.norm() + 1e-12)).item()
    cos = F.cosine_similarity(a32, b32, dim=0).item()
    return abs_err, rel_fro, cos


def print_cmp(
    tag: str,
    y: torch.Tensor,
    y_ref: torch.Tensor,
    dx: torch.Tensor,
    dx_ref: torch.Tensor,
    dw: torch.Tensor,
    dw_ref: torch.Tensor,
    db: torch.Tensor,
    db_ref: torch.Tensor,
) -> None:
    ya, yr, yc = metrics(y, y_ref)
    dxa, dxr, dxc = metrics(dx, dx_ref)
    dwa, dwr, dwc = metrics(dw, dw_ref)
    dba, dbr, dbc = metrics(db, db_ref)
    print(
        f"  {tag:<14} fwd(abs={ya:.3e}, rel={yr:.3e}, cos={yc:.6f})  "
        f"dx(abs={dxa:.3e}, rel={dxr:.3e}, cos={dxc:.6f})  "
        f"dw(abs={dwa:.3e}, rel={dwr:.3e}, cos={dwc:.6f})  "
        f"db(abs={dba:.3e}, rel={dbr:.3e}, cos={dbc:.6f})",
        flush=True,
    )


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
) -> None:
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)

    y = fn(x, w, b)
    y.backward(dy)
    torch.cuda.synchronize()
    print_cmp(
        tag,
        y.detach(),
        y_ref,
        x.grad.detach(),
        dx_ref,
        w.grad.detach(),
        dw_ref,
        b.grad.detach(),
        db_ref,
    )

    t_fwd = do_bench(lambda: fn(x, w, b))
    t_full = do_bench(lambda: fn(x, w, b).backward(dy), grad_to_none=[x, w, b])
    print(f"{tag} fwd={t_fwd:.4f} ms fwd+bwd={t_full:.4f} ms", flush=True)


def parse_int_list(text: str) -> list[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone LayerNorm M/D sweep benchmark")
    p.add_argument(
        "--l-values",
        default=",".join(str(v) for v in DEFAULT_L_LIST),
        help="comma-separated L values; M is benchmarked as L^2",
    )
    p.add_argument(
        "--d-values",
        default=",".join(str(v) for v in DEFAULT_D_LIST),
        help="comma-separated hidden sizes D to benchmark",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    l_list = parse_int_list(args.l_values)
    d_list = parse_int_list(args.d_values)

    print(f"host={torch.cuda.get_device_name(0)} torch={torch.__version__} dtype={DTYPE}", flush=True)
    print(
        "implementations=pytorch,triton,cuequivariance,layernorm_kernel  "
        f"sweep=M=L^2 for L in {l_list}, D in {d_list}  eps={EPS}",
        flush=True,
    )

    for d_model in d_list:
        for l in l_list:
            m = l * l
            print(f"\n=== M={m}  d_in={d_model}  d_out={d_model} ===", flush=True)
            print(f"# L={l} D={d_model}", flush=True)

            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.cuda.empty_cache()
            x0 = w0 = b0 = dy = None
            xp = wp = bp = yp = None
            y_ref = dx_ref = dw_ref = db_ref = None

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

                print(
                    compare_line(
                        "pytorch",
                        y_ref,
                        y_ref,
                        dx_ref,
                        dx_ref,
                        dw_ref,
                        dw_ref,
                        db_ref,
                        db_ref,
                    ),
                    flush=True,
                )
                t_pt_fwd = do_bench(lambda: layernorm_pytorch(xp, wp, bp, EPS))
                t_pt_full = do_bench(
                    lambda: layernorm_pytorch(xp, wp, bp, EPS).backward(dy),
                    grad_to_none=[xp, wp, bp],
                )
                print(f"pytorch fwd={t_pt_fwd:.4f} ms fwd+bwd={t_pt_full:.4f} ms", flush=True)

                run_impl(
                    "triton",
                    lambda x, w, b: triton_layernorm(x, w, b, EPS),
                    x0,
                    w0,
                    b0,
                    dy,
                    y_ref,
                    dx_ref,
                    dw_ref,
                    db_ref,
                )
                run_impl(
                    "cuequivariance",
                    lambda x, w, b: layer_norm_transpose(x, w, b, eps=EPS, layout="nd->nd"),
                    x0,
                    w0,
                    b0,
                    dy,
                    y_ref,
                    dx_ref,
                    dw_ref,
                    db_ref,
                )

                try:
                    run_impl(
                        "layernorm_kernel",
                        lambda x, w, b: layernorm_kernel(x, w, b, EPS),
                        x0,
                        w0,
                        b0,
                        dy,
                        y_ref,
                        dx_ref,
                        dw_ref,
                        db_ref,
                    )
                except NotImplementedError as exc:
                    print(f"layernorm_kernel [skipped: {exc}]", flush=True)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(f"# skipped: CUDA OOM for M={m}, D={d_model}: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            finally:
                del x0, w0, b0, dy, xp, wp, bp, yp, y_ref, dx_ref, dw_ref, db_ref
                torch.cuda.empty_cache()

    print("\nDONE", flush=True)


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
    """Single-line correctness summary for the raw bench log."""
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


if __name__ == "__main__":
    main()
