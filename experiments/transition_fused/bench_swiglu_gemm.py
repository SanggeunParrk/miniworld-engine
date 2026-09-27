"""swiglu_gemm (h = silu(xn Wa^T) * xn Wb^T in one wgmma GEMM) against cuBLAS and the engine's Triton expand, per width.

  python bench_swiglu_gemm.py --width 512 --length 384
"""
import argparse, statistics, sys
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=512); p.add_argument("--length", type=int, default=384)
p.add_argument("--cubin", default=str(HERE / "build/swiglu_gemm.cubin")); p.add_argument("--ncta", type=int, default=132)
p.add_argument("--engine", action="store_true"); p.add_argument("--cluster", type=int, default=0); a = p.parse_args()
D, H, M = a.width, 4 * a.width, a.length ** 2
PEAK = 989e12
torch.manual_seed(1)
xn = (torch.randn(M, D, device="cuda") * 0.8).to(torch.bfloat16)
wa = (torch.randn(H, D, device="cuda") * D ** -0.5).to(torch.bfloat16)
wb = (torch.randn(H, D, device="cuda") * D ** -0.5).to(torch.bfloat16)
w1p = torch.stack([wa.view(H // 128, 128, D), wb.view(H // 128, 128, D)], 1).reshape(2 * H, D).contiguous()
h = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
k = drv.Kernel(a.cubin, "swiglu_gemm", 229376 + 128, cluster=(a.cluster or None))
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
maps = (tm(xn, [D, M], D * 2, [64, 64]), tm(w1p, [D, 2 * H], D * 2, [64, 128 if a.cluster else 256]), tm(h, [H, M], H * 2, [64, 64]))
run = lambda: k((a.ncta, 1, 1), (384, 1, 1), *maps, int(M), int(D), int(H))
run(); torch.cuda.synchronize()
rows = torch.randperm(M, device="cuda")[:4096]
ref = (F.silu(xn[rows].float() @ wa.float().t()) * (xn[rows].float() @ wb.float().t()))
d = h[rows].float() - ref
print(f"{Path(a.cubin).name}: regs {k.regs} lmem {k.lmem} | D {D} H {H} M {M}: rel_rms vs fp32 {float(d.norm() / ref.norm()):.3e}  finite {bool(torch.isfinite(h).all())}")
bf = (F.silu(xn[rows] @ wa.t()) * (xn[rows] @ wb.t())).float()
print(f"  (cuBLAS bf16 unfused vs fp32: {float((bf - ref).norm() / ref.norm()):.3e})")
h2 = h.clone(); run(); torch.cuda.synchronize(); print("  bit-reproducible:", bool(torch.equal(h, h2)))
def t(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
flop = 2 * M * D * 2 * H
w1 = torch.cat([wa, wb], 0)
for name, fn in (("swiglu_gemm", run), ("cuBLAS GEMM only [M,8D]", lambda: xn @ w1.t()),
                 ("cuBLAS GEMM + silu*b", lambda: (lambda ab: F.silu(ab[:, :H]) * ab[:, H:])(xn @ w1.t()))):
    us = t(fn); print(f"  {name:26s} {us:8.1f} us  {100 * flop / PEAK / (us * 1e-6):5.1f} % of tensor peak")
if a.engine:
    from miniworld_engine.autotune.shape_key import both_key
    from miniworld_engine.kernels.transition.triton.main import _expand_swiglu
    key = both_key(M)
    us = t(lambda: _expand_swiglu(xn, wa, wb, key)); print(f"  {'engine Triton _expand_swiglu':26s} {us:8.1f} us  {100 * flop / PEAK / (us * 1e-6):5.1f} % of tensor peak")
