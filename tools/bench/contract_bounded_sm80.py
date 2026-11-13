"""A100: the bounded TriMul contraction (``trimul_inproj/cuda/sm80/contract_bounded.cu``) against cuBLAS (torch.bmm, the path's contraction
today), unpadded and with zero tails (L // 8, L // 4). Prints one line per (h, L, mode) as it finishes."""

import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

from miniworld_engine.kernels._nvcc import cutlass_include, ensure_cuda_home

ensure_cuda_home()
SRC = os.path.join(os.path.dirname(__file__), "../../src/miniworld_engine/kernels/trimul_inproj/cuda/sm80/contract_bounded.cu")
t0 = time.time()
ext = load(
    "contract_bounded_sm80_bench",
    [os.path.abspath(SRC)],
    extra_include_paths=[str(cutlass_include())],
    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
)
print("build", round(time.time() - t0), "s", flush=True)
DEADLINE = time.time() + float(os.environ.get("BUDGET", "400"))
NCFG = int(os.environ.get("NCFG", "6"))


def tm(fn, it=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / it)
    return best


SHAPES = [tuple(map(int, x.split("x"))) for x in os.environ.get("SHAPES", "128x768,128x1024,128x1536,256x1024,128x2048,256x2048").split(",")]
for h, L in SHAPES:
    a = torch.randn(h, L, L, device="cuda").bfloat16()
    b = torch.randn(h, L, L, device="cuda").bfloat16()
    for mode in (0, 1):
        if time.time() > DEADLINE:
            print("deadline", flush=True)
            sys.exit(0)
        ref = torch.empty_like(a)
        A, B = (a, b.transpose(1, 2)) if mode == 0 else (a.transpose(1, 2), b)
        cub = tm(lambda: torch.bmm(A, B, out=ref))
        res = []
        d = torch.empty_like(a)
        for c in range(NCFG):
            try:
                ext.contract(a, b, d, mode, c, None, 1)
                same = torch.equal(d, ref)
                res.append((tm(lambda: ext.contract(a, b, d, mode, c, None, 1)), c, same))
            except Exception as ex:  # noqa: BLE001
                print(f"  c{c} ERR {str(ex)[:80]}", flush=True)
        res.sort()
        top = " | ".join(f"c{c} {t:.3f} ({cub / t:.3f}x){'' if s else ' DIFF'}" for t, c, s in res[:4])
        print(f"h{h} L{L} m{mode} cublas {cub:.3f}ms {2 * h * L**3 / 1e9 / cub:.0f}TF || {top}", flush=True)
        t, c, _ = res[0]
        for p in (8, 4):
            n = L - L // p
            ap, bp = a.clone(), b.clone()
            ap[:, n:], ap[:, :, n:], bp[:, n:], bp[:, :, n:] = 0, 0, 0, 0
            Ap, Bp = (ap, bp.transpose(1, 2)) if mode == 0 else (ap.transpose(1, 2), bp)
            torch.bmm(Ap, Bp, out=ref)
            full = torch.empty_like(d)
            ext.contract(ap, bp, full, mode, c, None, 1)
            bd = torch.tensor([[n, n, n]], dtype=torch.int32, device="cuda")
            d.fill_(7)
            ext.contract(ap, bp, d, mode, c, bd, 1)
            tb = tm(lambda: ext.contract(ap, bp, d, mode, c, bd, 1))
            tf = tm(lambda: torch.bmm(Ap, Bp, out=ref))
            print(
                f"   pad L//{p} n{n}: cublas {tf:.3f} bounded {tb:.3f} ({tf / tb:.2f}x) bitwise-vs-own-full {torch.equal(d, full)} "
                f"max|d-cublas| {(d.float() - ref.float()).abs().max().item():.3g}",
                flush=True,
            )
            del ap, bp, full
        del ref, d
    del a, b
    torch.cuda.empty_cache()
