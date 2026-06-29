"""Module-level benchmark for the Transition nn.Module (LN + SwiGLU MLP).

Mirrors scripts/bench.py:bench_transition (real Transition module, pair input (1,L,L,d) bf16).
Compares the user-facing paths end-to-end:
  PyTorch          = eager reference                         (palette: pytorch, grey)
  Triton (prev)    = LEGACY path: LN + kernels.triton_transition (h round-trips HBM)  (triton, blue)
  Triton b2b (ours)= shipped ImplementationType.TRITON dispatch (b2b @ d<=128 / cute @ d>=256)  (miniworld, red)
  cute (ours)      = forced ImplementationType.CUTE (quack SM90 WGMMA)                 (miniworld-alt, orange)

Emits plot_sweep-format CSVs (L, "Label|palette-identity" = median ms) for d=128 forward and
fwd+bwd, plus a d-crossover table. Identities are EXACT viz.PALETTE keys so plot_sweep gives
each series a distinct colour (it does PALETTE.get(ident) directly — non-key idents collide on
the hashed fallback). do_bench median, bf16, TF32 on. Run on a compute node (srun).
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import triton.testing as tt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from miniworld_kernels import kernels
from miniworld_kernels.modules.transition import Transition
from miniworld_kernels.modules.exceptions import ImplementationType

DEV = "cuda"
BENCH_DIR = Path("src/miniworld_kernels/modules/transition/benchmark")

# (display label, palette identity = EXACT viz.PALETTE key, kind)
IMPLS = [
    ("PyTorch", "pytorch", "module:pytorch"),
    ("Triton (prev)", "triton", "triton_prev"),
    ("Triton b2b (ours)", "miniworld", "module:triton"),
    ("cute (ours)", "miniworld-alt", "module:cute"),
]
LSEQ = [384, 512, 640, 768, 896, 1024]


def make_x(L, d, grad):
    x = torch.randn(1, L, L, d, device=DEV, dtype=torch.bfloat16)
    x.requires_grad_(grad)
    return x


def make_callable(kind, d):
    """Return a fn(x)->y closure for the column `kind` at hidden dim d."""
    if kind == "module:pytorch":
        m = Transition(d_hidden=d, n=4, implementation=ImplementationType.PYTORCH).to(DEV)
        return lambda x: m(x)
    if kind == "module:triton":
        m = Transition(d_hidden=d, n=4, implementation=ImplementationType.TRITON).to(DEV)
        return lambda x: m(x)
    if kind == "module:cute":
        m = Transition(d_hidden=d, n=4, implementation=ImplementationType.CUTE).to(DEV)
        return lambda x: m(x)
    if kind == "triton_prev":
        # legacy path: module LN (pytorch) then the OLD triton_transition kernel (h -> HBM).
        m = Transition(d_hidden=d, n=4, implementation=ImplementationType.PYTORCH).to(DEV)
        wa, wb, ws, n = m.expand_a.weight, m.expand_b.weight, m.squeeze.weight, m.n
        return lambda x: kernels.triton_transition(m.ln_in(x), wa, wb, ws, n)
    raise ValueError(kind)


def time_ms(fn):
    try:
        return tt.do_bench(fn, warmup=25, rep=100, return_mode="median")
    except Exception as e:
        print(f"   [skip] {type(e).__name__}: {str(e)[:100]}")
        return float("nan")


def bench(d, mode):
    res = {}
    for label, ident, kind in IMPLS:
        call = make_callable(kind, d)
        row = {}
        for L in LSEQ:
            grad = mode == "full"
            x = make_x(L, d, grad)
            dy = torch.randn_like(x) if grad else None
            def fwd():
                return call(x)
            def full():
                if x.grad is not None:
                    x.grad = None
                call(x).backward(dy)
            row[L] = time_ms(full if grad else fwd)
            del x, dy
            torch.cuda.empty_cache()
        res[ident] = row
        print(f" d={d} {mode:7} {label:20} " + " ".join(f"{row[L]:8.4f}" for L in LSEQ))
    return res


def write_csv(path, res):
    idents = [i for _, i, _ in IMPLS]
    header = ["seq_len"] + [f"{lab}|{i}" for lab, i, _ in IMPLS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for L in LSEQ:
            w.writerow([L] + [f"{res[i][L]:.4f}" for i in idents])
    print(f"  wrote {path}")


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Transition module, d=128, n=4, bf16 (median ms) ===")
    write_csv(BENCH_DIR / "transition_module_forward.csv", bench(128, "forward"))
    write_csv(BENCH_DIR / "transition_module_fwd_bwd.csv", bench(128, "full"))

    print("\n=== d-crossover (forward, L=512, M=262144) — why the dispatch exists ===")
    print(f"{'d':>5} " + " ".join(f"{lab:>20}" for lab, _, _ in IMPLS))
    for d in (128, 256, 512):
        cells = {}
        for label, ident, kind in IMPLS:
            call = make_callable(kind, d)
            x = make_x(512, d, False)
            cells[ident] = time_ms(lambda: call(x))
            del x; torch.cuda.empty_cache()
        print(f"{d:>5} " + " ".join(f"{cells[i]:>20.4f}" for _, i, _ in IMPLS))
    print("DONE")


if __name__ == "__main__":
    main()
