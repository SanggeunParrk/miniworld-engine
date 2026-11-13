"""H100: the bounded TriMul contraction (``trimul_inproj/cuda/_bounded.py``, ``h100_sources/bounded``) against the path's cuBLAS contraction
(``_h100_bmm.contract``: the measured cuBLASLt configurations where listed), unpadded and with zero tails (L // 8, L // 4). A quick pass
ranks (config, swizzle, raster); the best three are timed interleaved with cuBLAS (ROUNDS rounds, median ratio). One line per cell."""

import os
import statistics
import sys
import time

import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _bounded
from miniworld_engine.kernels.trimul_inproj.cuda._h100_bmm import contract as cublas

t0 = time.time()
ext = _bounded._ext_h100()
print("build", round(time.time() - t0), "s", flush=True)
if os.environ.get("BUILD_ONLY"):
    sys.exit(0)
DEADLINE = time.time() + float(os.environ.get("BUDGET", "400"))
CFGS = [int(x) for x in os.environ.get("CFGS", ",".join(map(str, range(_bounded.H100_CONFIGS)))).split(",")]
SWZ = [int(x) for x in os.environ.get("SWZ", "1,2,4").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "7"))
# raster 1 / 2 (along m / n), + 4: CUTLASS's tile order instead of live units first
RASTERS = [int(x) for x in os.environ.get("RASTERS", "1,2,5,6").split(",")]


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


SHAPES = [tuple(map(int, x.split("x"))) for x in os.environ.get("SHAPES", "128x768,128x1024").split(",")]
MODES = [int(x) for x in os.environ.get("MODES", "0,1").split(",")]
for h, L in SHAPES:
    a = torch.randn(h, L, L, device="cuda").bfloat16()
    b = torch.randn(h, L, L, device="cuda").bfloat16()
    for mode in MODES:
        if time.time() > DEADLINE:
            print("deadline", flush=True)
            sys.exit(0)
        ref = torch.empty_like(a)
        A, B = (a, b.transpose(1, 2)) if mode == 0 else (a.transpose(1, 2), b)
        cublas(A, B, ref)
        res = []
        d = torch.empty_like(a)
        for c in CFGS:
            for s in SWZ:
                for r in RASTERS:
                    ext.contract(a, b, d, mode, c, None, 1, s, r)
                    res.append((tm(lambda c=c, s=s, r=r: ext.contract(a, b, d, mode, c, None, 1, s, r)), c, s, r, torch.equal(d, ref)))
        res.sort()
        out = []
        for _, c, s, r, same in res[:3]:
            tc, tb, ratio = interleaved(lambda: cublas(A, B, ref), lambda c=c, s=s, r=r: ext.contract(a, b, d, mode, c, None, 1, s, r))
            out.append((ratio, c, s, r, tc, tb, same))
        out.sort(reverse=True)
        ratio, c, s, r, tc, tb, same = out[0]
        rest = " ".join(f"c{x[1]}s{x[2]}r{x[3]} {x[0]:.3f}" for x in out[1:])
        print(
            f"h{h} L{L} m{mode} cublas {tc:.3f}ms best c{c}s{s}r{r} {tb:.3f}ms speedup(median) {ratio:.3f}{'' if same else ' DIFF'} | {rest}",
            flush=True,
        )
        for p in (8, 4):
            n = L - L // p
            ap, bp = a.clone(), b.clone()
            ap[:, n:], ap[:, :, n:], bp[:, n:], bp[:, :, n:] = 0, 0, 0, 0
            Ap, Bp = (ap, bp.transpose(1, 2)) if mode == 0 else (ap.transpose(1, 2), bp)
            cublas(Ap, Bp, ref)
            full = torch.empty_like(d)
            ext.contract(ap, bp, full, mode, c, None, 1, s, r)
            bd = torch.tensor([[n, n, n]], dtype=torch.int32, device="cuda")
            d.fill_(7)
            ext.contract(ap, bp, d, mode, c, bd, 1, s, r)
            tf, tbd, rr = interleaved(lambda: cublas(Ap, Bp, ref), lambda: ext.contract(ap, bp, d, mode, c, bd, 1, s, r))
            print(
                f"   pad L//{p} n{n}: cublas {tf:.3f} bounded {tbd:.3f} ({rr:.2f}x) bitwise-vs-own-full {torch.equal(d, full)} "
                f"bitwise-vs-cublas {torch.equal(d, ref)}",
                flush=True,
            )
            del ap, bp, full
        del ref, d
    del a, b
    torch.cuda.empty_cache()
