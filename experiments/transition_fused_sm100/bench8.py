"""Relaxed-precision (e4m3) Transition: accuracy of the forward output and all gradients vs the fp32 module, timing of inference
(forward, weights pre-quantized) and of the training step (quantize weights + forward + backward); bf16 kernels alongside."""
import argparse, json, torch
from common import D, make_inputs, rel, graph_time, fp32_fwd
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
from fwd8_op import Train8

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--repl", type=int, default=14)
p.add_argument("--fcubin", default="build/tfwd8.cubin"); p.add_argument("--bcubin", default="build/tbwd8x.cubin")
p.add_argument("--no-time", action="store_true")
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
x, wa, wb, ws, gamma, beta = make_inputs(a.length)
dy = (torch.randn(x.shape, generator=torch.Generator().manual_seed(7)) * 0.1).to("cuda", torch.bfloat16)
st = Train8(a.repl, a.fcubin, a.bcubin).bind(x, wa, wb, ws, gamma, beta, dy)
st(); st.infer(); torch.cuda.synchronize()
xr = x.float().requires_grad_(True)
prm = [t.float().requires_grad_(True) for t in (wa, wb, ws, gamma, beta)]
xnr = torch.nn.functional.layer_norm(xr, (D,), prm[3], prm[4], 1e-5)
y = xr + (torch.nn.functional.silu(xnr @ prm[0].t()) * (xnr @ prm[1].t())) @ prm[2].t()
y.backward(dy.float())
ref = dict(dx=xr.grad, dwa=prm[0].grad, dwb=prm[1].grad, dws=prm[2].grad, dgamma=prm[3].grad, dbeta=prm[4].grad)
f16 = FusedFwd2(); f16.set_weights(wa, wb, ws)
run16, out16, *_ = f16.bind(x, gamma, beta, save=False); run16()
st16 = FusedTrain(f16).bind(x, gamma, beta, dy); st16(); torch.cuda.synchronize()
res = {"L": a.length, "repl": a.repl}
yref = y.detach()
res["y"] = {"e4m3_vs_fp32": rel(st.out, yref), "bf16_vs_fp32": rel(out16, yref),
            "residual_branch_e4m3": rel(st.out.float() - x.float(), yref - x.float()), "residual_branch_bf16": rel(out16.float() - x.float(), yref - x.float())}
xn_ref = torch.nn.functional.layer_norm(x.float(), (D,), gamma, beta, 1e-5)
res["xq_vs_fp32_xn"] = rel(st.xq.view(torch.float8_e4m3fn).float() * st.sc[0], xn_ref)
for k in ref:
    res[k] = {"e4m3_vs_fp32": rel(st.grads[k], ref[k]), "bf16_vs_fp32": rel(st16.grads[k], ref[k])}
res["finite"] = all(bool(torch.isfinite(v.float()).all()) for v in st.grads.values()) and bool(torch.isfinite(st.out.float()).all())
g0 = {k: v.clone() for k, v in st.grads.items()}; st(); torch.cuda.synchronize()
res["bit_repro"] = all(torch.equal(g0[k], st.grads[k]) for k in g0)
if not a.no_time:
    res["us_infer_e4m3"] = graph_time(st.infer); res["us_infer_bf16"] = graph_time(run16)
    res["us_step_e4m3"] = graph_time(st); res["us_step_bf16"] = graph_time(st16)
    res["us_quant"] = graph_time(st.keep[0]); res["us_fwd_train_e4m3"] = graph_time(st.keep[1]); res["us_bwd_e4m3"] = graph_time(st.keep[2])
print(json.dumps(res, indent=1))
