"""gate forward: triton (cuBLAS gemm + triton sigmoid·mul) vs quack (gemm_act sigmoid + mul).
Correctness (cos) + forward ms. event-timed eager. COMPUTE NODE only, fresh QUACK_CACHE_DIR."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src_root = _here
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import triton

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_quack, gate_elem_quack_fused, gate_elem_triton,
)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def _bench(fn):
    for _ in range(10):
        fn()
    return triton.testing.do_bench(fn, warmup=20, rep=80, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"gate fwd triton vs quack on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16
    torch.manual_seed(0)
    print(f"{'d':>4} {'L':>5} | {'triton ms':>10} {'fused ms':>10} {'speedup':>8} | cos")
    print("-" * 56)
    for D in (128, 256, 512):
        for L in (256, 512, 1024):
            M = L * L
            x_n = torch.randn(M, D, device="cuda", dtype=dt) * 0.5
            proj = torch.randn(M, D, device="cuda", dtype=dt)
            Wg = torch.randn(D, D, device="cuda", dtype=dt) * D**-0.5
            yt = gate_elem_triton(x_n, proj, Wg)
            yf = gate_elem_quack_fused(x_n, proj, Wg)               # one fused kernel: σ(x@Wg)⊙proj
            c = cos(yt, yf)
            tt = _bench(lambda: gate_elem_triton(x_n, proj, Wg))
            tf = _bench(lambda: gate_elem_quack_fused(x_n, proj, Wg))
            print(f"{D:>4} {L:>5} | {tt:>10.4f} {tf:>10.4f} {tt/tf:>7.2f}x | {c:.5f}", flush=True)
            del x_n, proj, Wg, yt, yf
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
