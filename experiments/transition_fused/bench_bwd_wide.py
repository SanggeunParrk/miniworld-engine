"""Full Transition backward for the wide widths (D >= 256) from our pieces, against the engine module and fp32.
   gate_gemm2 (h, [dA|dB]) -> cuBLAS dWs = dy^T h, dWab = dAB^T xn, d_xn = dAB [Wa;Wb] -> engine LN backward (+ dy residual).
   python bench_bwd_wide.py --width 256 --length 384"""
import argparse, copy, statistics, sys
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=384)
p.add_argument("--cubin", default=str(HERE / "build/gg2_k32s4h_w0.cubin")); p.add_argument("--tbk", type=int, default=32); p.add_argument("--hb", type=int, default=128)
p.add_argument("--dxln", default="", help="dxn_lnbwd cubin (COLS=2, TBK=64); empty = cuBLAS d_xn + engine LN backward"); a = p.parse_args()
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd
D, L = a.width, a.length; H, M = 4 * D, L * L; bf = torch.bfloat16
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
        else: prm.copy_(1 + 0.2 * torch.randn_like(prm)) if prm is m.ln_in.weight else prm.normal_(std=0.2)
x = torch.randn(1, L, L, D, device="cuda", dtype=bf, requires_grad=True); dy = torch.randn_like(x)
gamma, beta = m.ln_in.weight, m.ln_in.bias
wa, wb, ws = m.expand_a.weight, m.expand_b.weight, m.squeeze.weight
# ---------------------------------------------------------------- fp32 reference and the engine's own grads
ref = copy.deepcopy(m).float(); xr = x.detach().float().requires_grad_(True)
ref(xr).backward(dy.float())
R = {"dx": xr.grad, "dgamma": ref.ln_in.weight.grad, "dbeta": ref.ln_in.bias.grad, "dWa": ref.expand_a.weight.grad, "dWb": ref.expand_b.weight.grad, "dWs": ref.squeeze.weight.grad}
del ref, xr
y = m(x); y.backward(dy)
E = {"dx": x.grad.clone(), "dgamma": gamma.grad.clone(), "dbeta": beta.grad.clone(), "dWa": wa.grad.clone(), "dWb": wb.grad.clone(), "dWs": ws.grad.clone()}
# ---------------------------------------------------------------- our backward
x2 = x.detach().reshape(M, D); go = dy.reshape(M, D)
with torch.no_grad():
    xf = x2.float(); mu = xf.mean(-1); var = xf.var(-1, unbiased=False); rstd = torch.rsqrt(var + 1e-5); c1 = mu * rstd
    xn = ((xf * rstd[:, None] - c1[:, None]) * gamma.float() + beta.float()).to(bf).contiguous()
hb = a.hb
w1p = torch.stack([wa.detach().view(H // hb, hb, D), wb.detach().view(H // hb, hb, D)], 1).reshape(2 * H, D).contiguous()
wst = ws.detach().t().contiguous(); w_ab = torch.cat((wa.detach(), wb.detach()), 0)
h = torch.empty(M, H, device="cuda", dtype=bf); dab = torch.empty(M, 2 * H, device="cuda", dtype=bf)
k = drv.Kernel(a.cubin, "gate_gemm", 231424)
T = a.tbk
tk = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box, swizzle=2 * T)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
maps = (tk(xn, [D, M], D * 2, [T, 64]), tk(go, [D, M], D * 2, [T, 64]), tk(w1p, [D, 2 * H], D * 2, [T, 128]),
        tk(wst, [D, H], D * 2, [T, hb]), tm(h, [H, M], H * 2, [64, 64]), tm(dab, [2 * H, M], 2 * H * 2, [64, 64]))
gate = lambda: k((132, 1, 1), (384, 1, 1), *maps, int(M), int(D), int(H))
if a.dxln:
    kd = drv.Kernel(a.dxln, "dxn_lnbwd", 231424)
    wabT = w_ab.t().contiguous(); g32 = gamma.detach().float().contiguous()
    mA2 = drv.TensorMap(dab, dims=[2 * H, M], stride_bytes=2 * H * 2, box=[64, 64])
    mB2 = drv.TensorMap(wabT, dims=[2 * H, D], stride_bytes=2 * H * 2, box=[64, D // 2])
    dxo = torch.empty(M, D, device="cuda", dtype=bf); pdg = torch.empty(132, D, device="cuda"); pdb = torch.empty(132, D, device="cuda")
    def dxln():
        kd((132, 1, 1), (384, 1, 1), mA2, mB2, x2, go, dxo, g32, rstd, c1, pdg, pdb, int(M))
        return dxo, pdg.sum(0), pdb.sum(0)
else:
    def dxln():
        dx, dg, db = _transition_ln_bwd(dab @ w_ab, x2, rstd, c1, gamma.detach())
        return dx.add_(go), dg, db
def ours():
    gate()
    dWs = go.t() @ h
    dWab = dab.t() @ xn
    dx, dg, db = dxln()
    return dx, dg, db, dWab[:H], dWab[H:], dWs
O = dict(zip(["dx", "dgamma", "dbeta", "dWa", "dWb", "dWs"], ours()))
rel = lambda u, r: float((u.float().reshape(-1) - r.reshape(-1)).norm() / r.norm())
print(f"D{D} L{L}: grad rel error vs fp32 (ours | engine)")
for n in R: print(f"  {n:7s} {rel(O[n], R[n]):.2e} | {rel(E[n], R[n]):.2e}")
def t(fn, reps=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
def eng_bwd():
    y = m(x); y.backward(dy)
eng_fwd = lambda: m(x)
ef, efb = t(eng_fwd), t(eng_bwd)
parts = {"gate": gate, "dWs": lambda: go.t() @ h, "dWab": lambda: dab.t() @ xn, "d_xn+ln_bwd": dxln}
pt = {n: t(f) for n, f in parts.items()}
ob = t(ours)
print(f"  engine module: fwd {ef:.1f}  fwd+bwd {efb:.1f}  -> bwd {efb - ef:.1f} us")
print(f"  ours bwd {ob:.1f} us  (" + "  ".join(f"{n} {v:.1f}" for n, v in pt.items()) + f")  -> bwd x{(efb - ef) / ob:.2f}")
