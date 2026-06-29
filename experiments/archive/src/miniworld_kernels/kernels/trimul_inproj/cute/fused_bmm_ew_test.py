"""Does fused bmm+gated-EW beat cuBLAS bmm + torch EW? correctness + speed."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import triton

from miniworld_kernels.kernels.trimul_inproj.triton.fused_bmm_ew import fused_bmm_gated


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    print(f"fused bmm+gated-EW on {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(0)
    D = 128
    for L in (512, 1024):
        dt = torch.bfloat16
        dtri = torch.randn(D, L, L, device="cuda", dtype=dt) * 0.1
        rhs = torch.randn(D, L, L, device="cuda", dtype=dt) * 0.1
        gLlog = torch.randn(D, L, L, device="cuda", dtype=dt)
        pL = torch.randn(D, L, L, device="cuda", dtype=dt)

        d_p, d_glog = fused_bmm_gated(dtri, rhs, gLlog, pL)
        # ref: cuBLAS bmm then torch EW
        d_left = torch.bmm(dtri.float(), rhs.float())  # (D,L,L)
        g = torch.sigmoid(gLlog.float())
        rp = d_left * g
        rg = d_left * pL.float() * g * (1 - g)
        if L == 512:
            print(f"  cos d_p={cos(d_p, rp):.5f} d_glog={cos(d_glog, rg):.5f}", flush=True)

        def fused():
            fused_bmm_gated(dtri, rhs, gLlog, pL)

        def baseline():
            dl = torch.bmm(dtri, rhs)
            gg = torch.sigmoid(gLlog)
            return dl * gg, dl * pL * gg * (1 - gg)
        cbaseline = torch.compile(baseline)
        for _ in range(5):
            cbaseline()
        tf = triton.testing.do_bench(fused, warmup=10, rep=50, return_mode="median")
        tb = triton.testing.do_bench(baseline, warmup=10, rep=50, return_mode="median")
        tcb = triton.testing.do_bench(cbaseline, warmup=10, rep=50, return_mode="median")
        tbmm = triton.testing.do_bench(lambda: torch.bmm(dtri, rhs), warmup=10, rep=50, return_mode="median")
        print(f"  L={L:>4}: fused {tf:.3f} | cuBLAS+EW(eager) {tb:.3f} | "
              f"cuBLAS+EW(compiled) {tcb:.3f} | bmm-only {tbmm:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
