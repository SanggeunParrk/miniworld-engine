"""dxn_lnbwd (d_xn GEMM + LayerNorm backward + dy) against the engine's cuBLAS d_xn + _transition_ln_bwd + add, and fp32.
   python bench_dxln.py --width 256 --length 384 --cubin build/dl_d256c2.cubin --tbk 64 --cols 2"""
import argparse, statistics, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=384)
p.add_argument("--cubin", required=True); p.add_argument("--tbk", type=int, default=64); p.add_argument("--cols", type=int, default=1)
p.add_argument("--no-engine", action="store_true"); a = p.parse_args()
from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd
D, H, M = a.width, 4 * a.width, a.length ** 2; bf = torch.bfloat16
torch.manual_seed(3)
x = torch.randn(M, D, device="cuda").to(bf); dy = torch.randn(M, D, device="cuda").to(bf)
dab = (torch.randn(M, 2 * H, device="cuda") * 0.3).to(bf); w_ab = (torch.randn(2 * H, D, device="cuda") * (2 * H) ** -0.5).to(bf)
gamma = (1 + 0.2 * torch.randn(D, device="cuda")).to(bf)
xf = x.float(); mu = xf.mean(-1); rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5); c1 = (mu * rstd).contiguous()
wabT = w_ab.t().contiguous(); g32 = gamma.float().contiguous()
NCTA = 132
out = torch.empty(M, D, device="cuda", dtype=bf); pdg = torch.empty(NCTA, D, device="cuda"); pdb = torch.empty(NCTA, D, device="cuda")
k = drv.Kernel(a.cubin, "dxn_lnbwd", 231424)
T = a.tbk; NW = D // a.cols; BR = NW if a.cols == 2 else D; ROWB = 128 // a.cols
mA = drv.TensorMap(dab, dims=[2 * H, M], stride_bytes=2 * H * 2, box=[T, 64], swizzle=2 * T)
mB = drv.TensorMap(wabT, dims=[2 * H, D], stride_bytes=2 * H * 2, box=[T, BR], swizzle=2 * T)
mX = drv.TensorMap(x, dims=[D, M], stride_bytes=D * 2, box=[64, 64])
mDX = drv.TensorMap(out, dims=[D, M], stride_bytes=D * 2, box=[64, 64])
run = lambda: k((NCTA, 1, 1), (384, 1, 1), mA, mB, mX, mDX, x, dy, out, g32, rstd, c1, pdg, pdb, int(M))
run(); torch.cuda.synchronize()
dg, db = pdg.sum(0), pdb.sum(0)
# fp32 reference on a row subset for dx, all rows for dgamma / dbeta
dxn = dab.float() @ w_ab.float()
xh = xf * rstd[:, None] - c1[:, None]; w = g32 * dxn
ref_dx = (w - xh * (xh * w).mean(-1, keepdim=True) - w.mean(-1, keepdim=True)) * rstd[:, None] + dy.float()
ref_dg, ref_db = (dxn * xh).sum(0), dxn.sum(0)
del dxn, w
rel = lambda u, r: float((u.float() - r).norm() / r.norm())
def eng():
    d_xn = dab @ w_ab
    ddx, ddg, ddb = _transition_ln_bwd(d_xn, x, rstd, c1, gamma)
    return ddx.add_(dy), ddg, ddb
edx, edg, edb = eng()
print(f"D{D} L{a.length} cols{a.cols} regs {k.regs}: dx {rel(out, ref_dx):.2e} (engine {rel(edx, ref_dx):.2e})  dgamma {rel(dg, ref_dg):.2e} ({rel(edg, ref_dg):.2e})  dbeta {rel(db, ref_db):.2e} ({rel(edb, ref_db):.2e})")
o2 = out.clone(); p2 = pdg.clone(); run(); torch.cuda.synchronize(); print("  bit-reproducible:", bool(torch.equal(o2, out) and torch.equal(p2, pdg)))
del ref_dx
def t(fn, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
fl = 2 * M * 2 * H * D
us = t(lambda: (run(), pdg.sum(0), pdb.sum(0)))
print(f"  dxn_lnbwd {us:8.1f} us  {100 * fl / 989e12 / (us * 1e-6):5.1f} % of tensor peak")
if not a.no_engine:
    ue = t(eng); print(f"  engine    {ue:8.1f} us  -> x{ue / us:.2f}")
