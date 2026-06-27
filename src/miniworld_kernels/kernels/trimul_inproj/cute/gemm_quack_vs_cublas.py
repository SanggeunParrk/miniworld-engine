"""Can quack's cute GEMM replace cuBLAS for the trimul backward GEMMs? Bench both on the
front_bwd shapes (dxn, dWs) and the gate-bwd shapes (dx_gate, dWg). bf16. COMPUTE NODE only.

If quack gemm ≳ cuBLAS here, the bwd GEMMs (the "rest" beyond dW) can move to cute and we can
fuse the dx_gate add via gemm_act(act=None, C=...). If quack is much slower, cuBLAS must stay.
"""

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


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def _bench(fn):
    for _ in range(10):
        fn()
    return triton.testing.do_bench(fn, warmup=20, rep=80, return_mode="median")


def main():
    assert torch.cuda.is_available()
    from quack.gemm_interface import gemm as quack_gemm
    print(f"quack gemm vs cuBLAS (bf16) on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16
    torch.manual_seed(0)
    print(f"{'op':>6} {'d':>4} {'L':>5} | {'M':>9} {'K':>6} {'N':>5} | "
          f"{'cuBLAS ms':>10} {'quack ms':>10} {'q/cub':>7} | cos")
    print("-" * 86)
    for D in (128, 256, 512):
        H = 2 * D  # bidir per-side hidden
        for L in (512, 1024):
            M = L * L
            # dxn : (M, 4H) @ (4H, D) -> (M, D)
            A1 = torch.randn(M, 4 * H, device="cuda", dtype=dt) * (4 * H) ** -0.5
            B1 = torch.randn(4 * H, D, device="cuda", dtype=dt) * (4 * H) ** -0.5
            r_cub = A1 @ B1
            r_q = quack_gemm(A1, B1)
            tc = _bench(lambda: A1 @ B1)
            tq = _bench(lambda: quack_gemm(A1, B1))
            print(f"{'dxn':>6} {D:>4} {L:>5} | {M:>9} {4*H:>6} {D:>5} | "
                  f"{tc:>10.4f} {tq:>10.4f} {tq/tc:>6.2f}x | {cos(r_cub, r_q):.4f}", flush=True)
            del A1, B1, r_cub, r_q
            # dWs : (4H, M) @ (M, D) -> (4H, D)   (huge-K reduction)
            A2 = torch.randn(4 * H, M, device="cuda", dtype=dt) * M ** -0.5
            B2 = torch.randn(M, D, device="cuda", dtype=dt) * M ** -0.5
            r_cub = A2 @ B2
            r_q = quack_gemm(A2, B2)
            tc = _bench(lambda: A2 @ B2)
            tq = _bench(lambda: quack_gemm(A2, B2))
            print(f"{'dWs':>6} {D:>4} {L:>5} | {4*H:>9} {M:>6} {D:>5} | "
                  f"{tc:>10.4f} {tq:>10.4f} {tq/tc:>6.2f}x | {cos(r_cub, r_q):.4f}", flush=True)
            del A2, B2, r_cub, r_q
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
