"""Standalone LayerNorm benchmark.

Benchmarks:
- PyTorch `F.layer_norm`
- legacy vendored Triton `triton_layernorm`
- cuequivariance `layer_norm_transpose(..., layout="nd->nd")`
- `layernorm_kernel()` with its internal auto-dispatch
- our-methods-only mode: `triton_atomic` / `triton_partial` / `layernorm_dispatch`

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

from miniworld_kernels.kernels.layernorm.cute.quack_adapter import (
    QUACK_AVAILABLE,
    quack_layernorm_fwd,
    quack_rmsnorm,
)
from miniworld_kernels.kernels.layernorm.interface import layernorm_kernel
from miniworld_kernels.kernels.layernorm.reference import layernorm_pytorch
from miniworld_kernels.kernels.layernorm.triton.lowreg import triton_layernorm_lowreg
from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm.triton.partial import triton_layernorm_partial
from miniworld_kernels.kernels.layernorm.triton.persistent import triton_layernorm_persistent

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
EPS = 1e-5
DEFAULT_L_LIST = [384, 512, 768, 1024]
DEFAULT_D_LIST = [128, 256, 384, 512, 768]

# H100 80GB HBM3 peak ~3.35 TB/s; report achieved bandwidth as a fraction of it so
# the forward roofline (how close LN already is to the memory wall) is explicit.
HBM_PEAK_TBS = 3.35


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


def run_fwd_only(
    tag: str,
    fn,
    x0: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    y_ref: torch.Tensor,
    m: int,
    n: int,
) -> None:
    """Time a forward-only kernel and report achieved HBM bandwidth (% of peak).

    Forward LayerNorm moves read(X) + write(Y) = 2*M*N*elem_size bytes; the % of
    HBM peak it sustains is the roofline number that says whether there is any room
    left in forward at all.
    """
    x = x0.detach().clone()
    w = w0.detach().clone()
    b = b0.detach().clone()
    y = fn(x, w, b)
    torch.cuda.synchronize()
    ya, yr, yc = metrics(y.detach(), y_ref)
    print(
        f"  {tag:<16} fwd(abs={ya:.3e}, rel={yr:.3e}, cos={yc:.6f})",
        flush=True,
    )
    t_fwd = do_bench(lambda: fn(x, w, b))
    gb = 2 * m * n * x0.element_size() / 1e9
    bw = gb / (t_fwd / 1e3)  # GB/s
    pct = 100.0 * bw / (HBM_PEAK_TBS * 1e3)
    print(f"{tag} fwd={t_fwd:.4f} ms  bw={bw / 1e3:.2f} TB/s ({pct:.0f}% of {HBM_PEAK_TBS} TB/s peak)", flush=True)


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
    p.add_argument(
        "--suite",
        choices=("baselines", "methods", "all", "fwd_tune", "cute_bwd"),
        default="baselines",
        help="which implementation family to benchmark",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    l_list = parse_int_list(args.l_values)
    d_list = parse_int_list(args.d_values)
    suite = args.suite

    impl_names: list[str]
    if suite == "fwd_tune":
        impl_names = ["pytorch", "triton", "lowreg"] + (["cute"] if QUACK_AVAILABLE else [])
    elif suite == "cute_bwd":
        impl_names = ["pytorch", "triton", "triton_persistent"] + (
            ["cute (quack RMSNorm proxy)"] if QUACK_AVAILABLE else []
        )
    elif suite == "methods":
        impl_names = ["pytorch", "triton_atomic", "triton_partial", "layernorm_dispatch"]
    elif suite == "all":
        impl_names = [
            "pytorch",
            "triton",
            "cuequivariance",
            "triton_atomic",
            "triton_partial",
            "layernorm_dispatch",
        ]
    else:
        impl_names = ["pytorch", "triton", "cuequivariance", "layernorm_kernel"]

    print(f"host={torch.cuda.get_device_name(0)} torch={torch.__version__} dtype={DTYPE}", flush=True)
    print(
        f"implementations={','.join(impl_names)}  "
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

                if suite == "fwd_tune":
                    # Forward-only roofline probe: pytorch vs shipped fused vs low-reg.
                    t_pt_fwd = do_bench(lambda: layernorm_pytorch(xp, wp, bp, EPS))
                    gb = 2 * m * d_model * x0.element_size() / 1e9
                    bw = gb / (t_pt_fwd / 1e3)
                    print(
                        f"pytorch fwd={t_pt_fwd:.4f} ms  bw={bw / 1e3:.2f} TB/s "
                        f"({100.0 * bw / (HBM_PEAK_TBS * 1e3):.0f}% of {HBM_PEAK_TBS} TB/s peak)",
                        flush=True,
                    )
                    run_fwd_only("triton", lambda x, w, b: triton_layernorm(x, w, b, EPS), x0, w0, b0, y_ref, m, d_model)
                    run_fwd_only("lowreg", lambda x, w, b: triton_layernorm_lowreg(x, w, b, EPS), x0, w0, b0, y_ref, m, d_model)
                    if QUACK_AVAILABLE:
                        run_fwd_only("cute", lambda x, w, b: quack_layernorm_fwd(x, w, b, EPS), x0, w0, b0, y_ref, m, d_model)
                    continue

                if suite == "cute_bwd":
                    # cute-vs-triton backward speed. quack ships no LayerNorm bwd, so
                    # we bench quack RMSNorm (cute fwd+bwd, persistent sm_count partials)
                    # as a proxy and our triton LayerNorm fwd+bwd alongside. RMSNorm does
                    # slightly less work than LN (no mean, no db) — noted in the report.
                    t_pt = do_bench(
                        lambda: layernorm_pytorch(xp, wp, bp, EPS).backward(dy), grad_to_none=[xp, wp, bp]
                    )
                    print(f"pytorch fwd={do_bench(lambda: layernorm_pytorch(xp, wp, bp, EPS)):.4f} ms fwd+bwd={t_pt:.4f} ms", flush=True)
                    run_impl(
                        "triton", lambda x, w, b: triton_layernorm(x, w, b, EPS),
                        x0, w0, b0, dy, y_ref, dx_ref, dw_ref, db_ref,
                    )
                    run_impl(
                        "triton_persistent", lambda x, w, b: triton_layernorm_persistent(x, w, b, EPS),
                        x0, w0, b0, dy, y_ref, dx_ref, dw_ref, db_ref,
                    )
                    if QUACK_AVAILABLE:
                        # RMSNorm: weight only (no bias); compare its fwd+bwd cost.
                        xr = x0.detach().clone().requires_grad_(True)
                        wr = w0.detach().clone().requires_grad_(True)
                        yr = quack_rmsnorm(xr, wr, EPS)
                        yr.backward(dy)
                        torch.cuda.synchronize()
                        t_fwd = do_bench(lambda: quack_rmsnorm(xr, wr, EPS))
                        t_full = do_bench(lambda: quack_rmsnorm(xr, wr, EPS).backward(dy), grad_to_none=[xr, wr])
                        print(f"cute fwd={t_fwd:.4f} ms fwd+bwd={t_full:.4f} ms  [quack RMSNorm proxy]", flush=True)
                    continue

                if suite in ("baselines", "methods", "all"):
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

                if suite in ("baselines", "all"):
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

                if suite in ("methods", "all"):
                    run_impl(
                        "triton_atomic",
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
                        "triton_partial",
                        lambda x, w, b: triton_layernorm_partial(x, w, b, EPS),
                        x0,
                        w0,
                        b0,
                        dy,
                        y_ref,
                        dx_ref,
                        dw_ref,
                        db_ref,
                    )

                if suite in ("baselines", "all"):
                    dispatch_tag = "layernorm_kernel"
                else:
                    dispatch_tag = "layernorm_dispatch"

                try:
                    run_impl(
                        dispatch_tag,
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
                    print(f"{dispatch_tag} [skipped: {exc}]", flush=True)
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
