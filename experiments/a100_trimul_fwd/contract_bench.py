"""Custom contraction vs cuBLAS (torch.bmm) on the TriMul plane shapes: accuracy (vs fp32 bmm) and CUDA-graph time.
   python contract_bench.py [--extra -DFOO=1]"""
import argparse, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--grid", type=int, default=0)
a_ = ap.parse_args()
ext = TA.build(extra=a_.extra)
def t(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(int(max(20, 200000 / 1000))): g.replay()
    r = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n): g.replay()
        e.record(); e.synchronize(); r.append(s.elapsed_time(e) / n * 1000)
    return statistics.median(r)
for L in (384, 768):
    for CH, h, name in ((256, 128, "bidir"), (128, 128, "out"), (128, 0, "in")):
        a = torch.randn(CH, L, L, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
        x = torch.empty_like(a); xr = torch.empty_like(a)
        def cub():
            if h: torch.bmm(a[:h], b[:h].transpose(1, 2), out=xr[:h])
            if h < CH: torch.bmm(a[h:].transpose(1, 2), b[h:], out=xr[h:])
        ext.contract(a, b, x, h, a_.grid); cub(); torch.cuda.synchronize()
        ref = torch.empty(CH, L, L, device="cuda")
        if h: ref[:h] = torch.bmm(a[:h].float(), b[:h].float().transpose(1, 2))
        if h < CH: ref[h:] = torch.bmm(a[h:].float().transpose(1, 2), b[h:].float())
        e_c = float((x.float() - ref).norm() / ref.norm()); e_b = float((xr.float() - ref).norm() / ref.norm())
        tc = t(lambda: ext.contract(a, b, x, h, a_.grid)); tb = t(cub)
        fl = 2 * CH * L ** 3
        print(f"L{L} {name:5s}: custom {tc:8.1f} us ({fl / tc / 1e6:6.1f} TF)  cuBLAS {tb:8.1f} us  ratio {tb / tc:5.3f}  rel custom {e_c:.2e} cublas {e_b:.2e}", flush=True)
        del a, b, x, xr, ref
