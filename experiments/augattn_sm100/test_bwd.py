"""Backward accuracy (vs fp64 autograd) and timing of the sm_100a backward kernels."""
import argparse, math, torch
from common import H, D, make, rel, graph_time
from ops import Fwd2, Dqb, Dkv

p = argparse.ArgumentParser(); p.add_argument("--lengths", type=int, nargs="+", default=[384, 768]); p.add_argument("--A", type=int, default=48)
p.add_argument("--fwd", default="build/attn_fwd2.cubin"); p.add_argument("--dqb", default="build/attn_dqb.cubin")
p.add_argument("--dkv", default="build/attn_dkv.cubin"); p.add_argument("--notime", action="store_true")
a = p.parse_args()


def ref_grads(q, k, v, bias, do):
    qd, kd, vd, bd = (t.double().requires_grad_() for t in (q, k, v, bias))
    qh, kh, vh = (t[:, 0].transpose(1, 2) for t in (qd, kd, vd))
    s = qh @ kh.transpose(-1, -2) / math.sqrt(D) + bd[None]
    o = (torch.softmax(s, -1) @ vh).transpose(1, 2)[:, None]
    o.backward(do.double())
    return qd.grad, kd.grad, vd.grad, bd.grad


for L in a.lengths:
    q, k, v, bias = make(a.A, L)
    do = torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16)
    frun, O, LSE = Fwd2(a.fwd).bind(q, k, v, bias)
    frun(); torch.cuda.synchronize()
    Dd = (do.float().reshape(a.A * L, H, D) * O.view(a.A * L, H, D)).sum(-1).view(a.A, L, H).permute(0, 2, 1).contiguous()
    run, DQ, DB = Dqb(a.dqb).bind(q, k, v, do, bias, LSE, Dd, zeroed=True)
    bias_t = bias.transpose(1, 2).contiguous()
    run2, DK, DV = Dkv(a.dkv).bind(q, k, v, do, bias_t, LSE, Dd, dq_zero=DQ)
    run2(); run(); torch.cuda.synchronize()
    gq, gk, gv, gb = ref_grads(q, k, v, bias, do)
    print(f"L{L} A{a.A}: dq rel {rel(DQ.view_as(gq), gq):.2e}  dbias rel {rel(DB, gb):.2e}  dk rel {rel(DK.view_as(gk), gk):.2e}  dv rel {rel(DV.view_as(gv), gv):.2e}"
          f"  finite {bool(all(torch.isfinite(t).all() for t in (DQ, DB, DK, DV)))}", flush=True)
    del gq, gk, gv, gb
    if not a.notime:
        print(f"L{L}: dqb {graph_time(run):7.1f} us  (kernel only {graph_time(run.k_launch):7.1f} us)   dkv {graph_time(run2):7.1f} us", flush=True)
