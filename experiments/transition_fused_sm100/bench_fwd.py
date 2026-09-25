"""Correctness and graph timing of the fused sm_100a forward at one length."""
import argparse, json
import torch
from common import make_inputs, contract_fwd, fp32_fwd, rel, graph_time
from fwd_op import FusedFwd, FusedFwd2, FusedFwd3

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--cubin", default=None)
p.add_argument("--no-time", action="store_true")
p.add_argument("--v2", action="store_true", help="the 2-CTA forward (tfwd2)")
p.add_argument("--v3", action="store_true", help="tfwd3: three [a|b] buffers")
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
x, wa, wb, ws, gamma, beta = make_inputs(a.length)
f = FusedFwd3() if a.v3 else FusedFwd2() if a.v2 else (FusedFwd(a.cubin) if a.cubin else FusedFwd())
print(f"regs {f.k.regs} lmem {f.k.lmem}")
f.set_weights(wa, wb, ws)
run, out, xn, rstd, c1 = f.bind(x, gamma, beta, save=True)
run(); torch.cuda.synchronize()
ref_out, ref_xn, ref_rs, ref_c1 = contract_fwd(x, wa, wb, ws, gamma, beta)
r32 = fp32_fwd(x, wa, wb, ws, gamma, beta)
res = {
    "L": a.length,
    "out_vs_contract": rel(out, ref_out), "out_vs_fp32": rel(out, r32), "contract_vs_fp32": rel(ref_out, r32),
    "out_mismatch_frac": float((out != ref_out).float().mean()),
    "xn_vs_contract": rel(xn, ref_xn), "rstd": rel(rstd, ref_rs), "c1": rel(c1, ref_c1),
    "finite": bool(torch.isfinite(out.float()).all()),
}
out0 = out.clone(); run(); torch.cuda.synchronize(); res["bit_repro"] = bool(torch.equal(out0, out))
if not a.no_time:
    res["us_train"] = graph_time(run)
    run_i, *_ = f.bind(x, gamma, beta, save=False)
    res["us_infer"] = graph_time(run_i)
print(json.dumps(res, indent=1))
