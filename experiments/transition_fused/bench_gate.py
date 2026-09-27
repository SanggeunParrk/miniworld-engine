"""gate_gemm (ab = xn [Wa;Wb]^T, dh = bf16(dy Ws), h / dA / dB) against the engine's backward gate stage (cuBLAS dh + its
gate kernel, as the module runs it) and an fp32 reference.   python bench_gate.py --width 256 --length 384"""
import argparse, statistics, sys
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=384)
p.add_argument("--cubin", default=str(HERE / "build/gate_gemm.cubin")); p.add_argument("--tbk", type=int, default=64); p.add_argument("--cluster", type=int, default=0); p.add_argument("--hb", type=int, default=64); p.add_argument("--no-engine", action="store_true"); a = p.parse_args()
D, H, M = a.width, 4 * a.width, a.length ** 2
PEAK = 989e12
torch.manual_seed(5); bf = torch.bfloat16
xn = (torch.randn(M, D, device="cuda") * 0.8).to(bf); dy = torch.randn(M, D, device="cuda").to(bf)
wa = (torch.randn(H, D, device="cuda") * D ** -0.5).to(bf); wb = (torch.randn(H, D, device="cuda") * D ** -0.5).to(bf)
ws = (torch.randn(D, H, device="cuda") * H ** -0.5).to(bf)
w1p = torch.stack([wa.view(H // a.hb, a.hb, D), wb.view(H // a.hb, a.hb, D)], 1).reshape(2 * H, D).contiguous()
wst = ws.t().contiguous()
h = torch.empty(M, H, device="cuda", dtype=bf); dab = torch.empty(M, 2 * H, device="cuda", dtype=bf)
k = drv.Kernel(a.cubin, "gate_gemm", 231424, cluster=(a.cluster or None))
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
T, SWZ = a.tbk, 2 * a.tbk
tk = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box, swizzle=SWZ)
maps = (tk(xn, [D, M], D * 2, [T, 64]), tk(dy, [D, M], D * 2, [T, 64]), tk(w1p, [D, 2 * H], D * 2, [T, 128]),
        tk(wst, [D, H], D * 2, [T, a.hb // (2 if a.cluster else 1)]), tm(h, [H, M], H * 2, [64, 64]), tm(dab, [2 * H, M], 2 * H * 2, [64, 64]))
run = lambda: k((132, 1, 1), (384, 1, 1), *maps, int(M), int(D), int(H), h, dab)
run(); torch.cuda.synchronize()
rows = torch.randperm(M, device="cuda")[:4096]
A = xn[rows].float() @ wa.float().t(); B = xn[rows].float() @ wb.float().t(); G = dy[rows].float() @ ws.float()
S = torch.sigmoid(A); Lx = A * S
ref = {"h": Lx * B, "dA": G * B * (S + Lx * (1 - S)), "dB": G * Lx}
got = {"h": h[rows].float(), "dA": dab[rows, :H].float(), "dB": dab[rows, H:].float()}
print(f"D{D} L{a.length}: regs {k.regs} | " + "  ".join(f"{n} rel {float((got[n] - ref[n]).norm() / ref[n].norm()):.2e}" for n in ref))
h2, d2 = h.clone(), dab.clone(); run(); torch.cuda.synchronize(); print("  bit-reproducible:", bool(torch.equal(h, h2) and torch.equal(dab, d2)))
def t(fn, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True); s_.record()
        for _ in range(reps): fn()
        e_.record(); torch.cuda.synchronize(); o.append(s_.elapsed_time(e_) * 1e3 / reps)
    return statistics.median(o)
fl = 2 * M * D * 3 * H
us = t(run); print(f"  gate_gemm                        {us:8.1f} us  {100 * fl / PEAK / (us * 1e-6):5.1f} % of tensor peak")
if a.no_engine: sys.exit(0)
try:
    from miniworld_engine.kernels.transition.triton import fused as TF
    from miniworld_engine.autotune.shape_key import both_key
    key = both_key(M)
    rstd = torch.ones(M, device="cuda"); c1 = torch.zeros(M, device="cuda"); g1 = torch.ones(D, device="cuda"); b0 = torch.zeros(D, device="cuda")
    def eng():
        dh = dy @ ws
        return TF._transition_expand_gatebwd_savedxn_stacked(xn, wa, wb, dh, shape_key=key)
    ue = t(eng); print(f"  engine (cuBLAS dh + Triton gate)  {ue:8.1f} us  {100 * fl / PEAK / (ue * 1e-6):5.1f} % of tensor peak  -> x{ue / us:.2f}")
except Exception as e:  # noqa: BLE001
    print("  engine gate unavailable:", repr(e)[:160])
