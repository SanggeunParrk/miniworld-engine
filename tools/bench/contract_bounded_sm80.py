"""A100: the bounded TriMul contraction (``trimul_inproj/cuda/sm80/contract_bounded.cu``) against cuBLAS (torch.bmm, the path's contraction
today), unpadded and with zero tails (L // 8, L // 4). A quick pass ranks (config, swizzle); the best three are then timed interleaved with
cuBLAS (ROUNDS rounds, median ratio). One line per (h, L, mode) as it finishes."""

import os
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load

from miniworld_engine.kernels._nvcc import cutlass_include, ensure_cuda_home

ensure_cuda_home()
SRC = os.path.join(os.path.dirname(__file__), "../../src/miniworld_engine/kernels/trimul_inproj/cuda/sm80/contract_bounded.cu")
t0 = time.time()
ext = load(
    "contract_bounded_sm80_bench3",
    [os.path.abspath(SRC)],
    extra_include_paths=[str(cutlass_include())],
    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
)
print("build", round(time.time() - t0), "s", flush=True)
if os.environ.get("BUILD_ONLY"):
    sys.exit(0)
DEADLINE = time.time() + float(os.environ.get("BUDGET", "400"))
CFGS = [int(x) for x in os.environ.get("CFGS", "0,1,2,3,4,5,6,7,8").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "5"))


def graph(fn, it):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def once(g, it):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def tm(fn, it=5):
    g = graph(fn, it)
    return min(once(g, it) for _ in range(2))


def interleaved(fa, fb, it=10):
    ga, gb = graph(fa, it), graph(fb, it)
    ra, rb = [], []
    for _ in range(ROUNDS):
        ra.append(once(ga, it))
        rb.append(once(gb, it))
    return statistics.median(ra), statistics.median(rb), statistics.median(x / y for x, y in zip(ra, rb, strict=True))


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
        torch.bmm(A, B, out=ref)
        res = []
        d = torch.empty_like(a)
        for c in CFGS:
            for lg in (0, 1, 2, 3):
                ext.contract(a, b, d, mode, c, None, 1, lg)
                res.append((tm(lambda c=c, lg=lg: ext.contract(a, b, d, mode, c, None, 1, lg)), c, lg, torch.equal(d, ref)))
        res.sort()
        out = []
        for _, c, lg, same in res[:3]:
            tc, tb, ratio = interleaved(lambda: torch.bmm(A, B, out=ref), lambda c=c, lg=lg: ext.contract(a, b, d, mode, c, None, 1, lg))
            out.append((ratio, c, lg, tc, tb, same))
        out.sort(reverse=True)
        ratio, c, lg, tc, tb, same = out[0]
        rest = " ".join(f"c{x[1]}g{x[2]} {x[0]:.3f}" for x in out[1:])
        print(
            f"h{h} L{L} m{mode} cublas {tc:.3f}ms best c{c}g{lg} {tb:.3f}ms speedup(median) {ratio:.3f}{'' if same else ' DIFF'} | {rest}",
            flush=True,
        )
        for p in (8, 4):
            n = L - L // p
            ap, bp = a.clone(), b.clone()
            ap[:, n:], ap[:, :, n:], bp[:, n:], bp[:, :, n:] = 0, 0, 0, 0
            Ap, Bp = (ap, bp.transpose(1, 2)) if mode == 0 else (ap.transpose(1, 2), bp)
            torch.bmm(Ap, Bp, out=ref)
            full = torch.empty_like(d)
            ext.contract(ap, bp, full, mode, c, None, 1, lg)
            bd = torch.tensor([[n, n, n]], dtype=torch.int32, device="cuda")
            d.fill_(7)
            ext.contract(ap, bp, d, mode, c, bd, 1, lg)
            tf, tbd, r = interleaved(lambda: torch.bmm(Ap, Bp, out=ref), lambda: ext.contract(ap, bp, d, mode, c, bd, 1, lg))
            print(
                f"   pad L//{p} n{n}: cublas {tf:.3f} bounded {tbd:.3f} ({r:.2f}x) bitwise-vs-own-full {torch.equal(d, full)} "
                f"bitwise-vs-cublas {torch.equal(d, ref)}",
                flush=True,
            )
            del ap, bp, full
        del ref, d
    del a, b
    torch.cuda.empty_cache()
