"""Forward accuracy (vs fp64) and timing of the sm_100a core against the engine's Triton core and SDPA."""
import argparse, math, torch
import torch.nn.functional as F
from common import H, D, make, reference, rel, graph_time
from ops import Fwd, Fwd2

p = argparse.ArgumentParser(); p.add_argument("--lengths", type=int, nargs="+", default=[384, 768]); p.add_argument("--A", type=int, default=48)
p.add_argument("--cubin", default="build/attn_fwd.cubin"); p.add_argument("--qwid", type=int, default=48)
a = p.parse_args()
for L in a.lengths:
    q, k, v, bias = make(a.A, L)
    run, O, LSE = (Fwd2 if "fwd2" in a.cubin else Fwd)(a.cubin).bind(q, k, v, bias, qwid=a.qwid)
    run(); torch.cuda.synchronize()
    ref = reference(q, k, v, bias)
    qh, kh = (t.double()[:, 0].transpose(1, 2) for t in (q, k))
    lse_ref = torch.logsumexp(qh @ kh.transpose(-1, -2) / math.sqrt(D) + bias.double()[None], -1) / math.log(2.0)
    o = O.view(a.A, 1, L, H, D)
    print(f"L{L} A{a.A}: O rel {rel(o, ref):.2e}  LSE max |err| {(LSE.double() - lse_ref).abs().max().item():.2e}  finite {bool(torch.isfinite(O).all())}")
    kern_only = lambda: run.k_launch()
    print(f"L{L}: sm100 fwd {graph_time(run):7.1f} us  (kernel only {graph_time(kern_only):7.1f} us)", flush=True)
