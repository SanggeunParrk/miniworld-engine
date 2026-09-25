"""Accuracy (vs the fp32 autograd module) and timing of the e4m3 backward tbwd8, fed by the bf16 forward's saves (xn quantized in torch
for this test). Compare: the bf16 backward tbwd (v9)."""
import argparse, json, torch
from common import D, make_inputs, rel, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
from bwd8_op import FusedBwd8, Quant8, scales_for

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--repl", type=int, default=9)
p.add_argument("--cubin", default="build/tbwd8.cubin"); p.add_argument("--no-time", action="store_true")
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
x, wa, wb, ws, gamma, beta = make_inputs(a.length)
dy = (torch.randn(x.shape, generator=torch.Generator().manual_seed(7)) * 0.1).to("cuda", torch.bfloat16)
f = FusedFwd2(); f.set_weights(wa, wb, ws)
run_f, out, xn, rstd, c1 = f.bind(x, gamma, beta, save=True)
run_f(); torch.cuda.synchronize()
sc = scales_for(x, wa, wb, ws, gamma, beta, dy)
xq = (xn.float() / sc[0]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
qrun, wab_q, wst_q, ws_q = Quant8(a.cubin).bind(wa, wb, ws, sc)
qrun(); torch.cuda.synchronize()
b8 = FusedBwd8(a.cubin, a.repl)
print(f"bwd8 regs {b8.k.regs} lmem {b8.k.lmem}")
run_b, g = b8.bind(dy, xq, x, rstd, c1, gamma, wab_q, wst_q, sc, wa, wb, ws)
run_b(); torch.cuda.synchronize()
# fp32 autograd reference
xr = x.float().requires_grad_(True)
prm = [t.float().requires_grad_(True) for t in (wa, wb, ws, gamma, beta)]
xnr = torch.nn.functional.layer_norm(xr, (D,), prm[3], prm[4], 1e-5)
y = xr + (torch.nn.functional.silu(xnr @ prm[0].t()) * (xnr @ prm[1].t())) @ prm[2].t()
y.backward(dy.float())
ref = dict(dx=xr.grad, dwa=prm[0].grad, dwb=prm[1].grad, dws=prm[2].grad, dgamma=prm[3].grad, dbeta=prm[4].grad)
# the bf16 backward (v9) for comparison
st16 = FusedTrain(f).bind(x, gamma, beta, dy); st16(); torch.cuda.synchronize()
res = {"L": a.length, "sc": [round(v, 8) for v in sc.tolist()]}
for k in ref:
    res[k] = {"e4m3_vs_fp32": rel(g[k], ref[k]), "bf16_v9_vs_fp32": rel(st16.grads[k], ref[k])}
res["finite"] = all(bool(torch.isfinite(v.float()).all()) for v in g.values())
g0 = {k: v.clone() for k, v in g.items()}; run_b(); torch.cuda.synchronize()
res["bit_repro"] = all(torch.equal(g0[k], g[k]) for k in g)
if not a.no_time:
    res["us_bwd8"] = graph_time(run_b)
    res["us_bwd16"] = graph_time(st16.keep[1])
print(json.dumps(res, indent=1))
