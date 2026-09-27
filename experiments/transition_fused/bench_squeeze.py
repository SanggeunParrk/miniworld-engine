"""squeeze_gemm variants (out = h Ws^T + x) against cuBLAS addmm.   python bench_squeeze.py --width 512 --length 384 --variants sq_bn256:256,sq_bn128_s5:128"""
import argparse, statistics, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=512); p.add_argument("--length", type=int, default=384)
p.add_argument("--variants", default="sq_bn256:256"); a = p.parse_args()
D, H, M = a.width, 4 * a.width, a.length ** 2
PEAK = 989e12
torch.manual_seed(3)
h = (torch.randn(M, H, device="cuda") * 0.3).to(torch.bfloat16)
ws = (torch.randn(D, H, device="cuda") * H ** -0.5).to(torch.bfloat16)
x = torch.randn(M, D, device="cuda").to(torch.bfloat16)
ref = torch.addmm(x.float(), h.float(), ws.float().t())
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
def t(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
flop = 2 * M * H * D
wst = ws.t().contiguous()
us = t(lambda: torch.addmm(x, h, wst)); print(f"D{D} L{a.length} {'cuBLAS addmm':16s} {us:8.1f} us {100 * flop / PEAK / (us * 1e-6):5.1f} %")
for v in a.variants.split(","):
    name, bn = v.split(":"); bn = int(bn)
    if D % bn: continue
    out = torch.empty_like(x)
    k = drv.Kernel(str(HERE / f"build/{name}.cubin"), "squeeze_gemm", 231424)
    maps = (tm(h, [H, M], H * 2, [64, 64]), tm(ws, [H, D], H * 2, [64, min(bn, 256)]), tm(x, [D, M], D * 2, [64, 64]), tm(out, [D, M], D * 2, [64, 64]))
    run = lambda: k((132, 1, 1), (384, 1, 1), *maps, int(M), int(H), int(D))
    run(); torch.cuda.synchronize()
    err = float((out.float() - ref).norm() / ref.norm())
    us = t(run); print(f"D{D} L{a.length} {name:16s} {us:8.1f} us {100 * flop / PEAK / (us * 1e-6):5.1f} %   rel vs fp32 {err:.2e}")
