"""Correctness and timing of the fused sm_100a backward (and the training step) at one length."""
import argparse, json
import torch
from common import D, make_inputs, contract_fwd, rel, graph_time
from fwd_op import FusedFwd
from bwd_op import FusedTrain

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--repl", type=int, default=9)
p.add_argument("--no-time", action="store_true")
p.add_argument("--v2", action="store_true", help="tbwd2 (exchange, no DW recompute)")
p.add_argument("--x", action="store_true", help="tbwdx (bf16 exchange: DW publishes [dA | dB], no DX recompute)")
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
x, wa, wb, ws, gamma, beta = make_inputs(a.length)
dy = (torch.randn(x.shape, generator=torch.Generator().manual_seed(7)) * 0.1).to("cuda", torch.bfloat16)
from fwd_op import FusedFwd2
f = FusedFwd2(); f.set_weights(wa, wb, ws)
tr = FusedTrain(f, repl=a.repl, v2=a.v2, x=a.x)
print(f"bwd regs {tr.b.k.regs} lmem {tr.b.k.lmem}")
step = tr.bind(x, gamma, beta, dy)
step(); torch.cuda.synchronize()
g = step.grads
# ---- the contract, emulated (same rounding points)
_, xn, rs, c1 = contract_fwd(x, wa, wb, ws, gamma, beta)
xnf = xn.float(); dyf = dy.float()
dh = (dyf @ ws.float()).bfloat16().float()
A = xnf @ wa.float().t(); Bv = xnf @ wb.float().t()
s = torch.sigmoid(A); l = A * s
h = (l * Bv).bfloat16().float()
dA = ((dh * Bv) * (s + l * (1 - s))).bfloat16().float()
dB = (dh * l).bfloat16().float()
c = dict(dws=(dyf.t() @ h).bfloat16(), dwa=(dA.t() @ xnf).bfloat16(), dwb=(dB.t() @ xnf).bfloat16())
dxn = (dA @ wa.float() + dB @ wb.float()).bfloat16().float()
mean = (c1 / rs)[:, None]
xhat = (x.float() - mean) * rs[:, None]
w = gamma * dxn
ca = (xhat * w).mean(-1, keepdim=True); cb = w.mean(-1, keepdim=True)
c["dx"] = (((w - xhat * ca - cb) * rs[:, None]).bfloat16().float() + dyf).bfloat16()
c["dgamma"] = (dxn * xhat).sum(0); c["dbeta"] = dxn.sum(0)
# ---- fp32 autograd reference of the module
xr = x.float().requires_grad_(True)
prm = [t.float().requires_grad_(True) for t in (wa, wb, ws, gamma, beta)]
xnr = torch.nn.functional.layer_norm(xr, (D,), prm[3], prm[4], 1e-5)
y = xr + (torch.nn.functional.silu(xnr @ prm[0].t()) * (xnr @ prm[1].t())) @ prm[2].t()
y.backward(dyf)
ref = dict(dx=xr.grad, dwa=prm[0].grad, dwb=prm[1].grad, dws=prm[2].grad, dgamma=prm[3].grad, dbeta=prm[4].grad)
res = {"L": a.length, "repl": a.repl}
for k in ref:
    res[k] = {"vs_contract": rel(g[k], c[k]), "vs_fp32": rel(g[k], ref[k]), "contract_vs_fp32": rel(c[k], ref[k])}
res["finite"] = all(bool(torch.isfinite(v.float()).all()) for v in g.values())
g0 = {k: v.clone() for k, v in g.items()}; step(); torch.cuda.synchronize()
res["bit_repro"] = all(torch.equal(g0[k], g[k]) for k in g)
if not a.no_time:
    res["us_train_step"] = graph_time(step)
    run_b = step.keep[1]
    res["us_bwd"] = graph_time(run_b)
print(json.dumps(res, indent=1))
