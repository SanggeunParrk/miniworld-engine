"""Time the training core kernels (graph replay, median of rounds) for one or more cubins.
usage: L=768 python t_core.py fwd build/attn_fwd2.cubin [build/x.cubin ...]   (kinds: fwd dqb dkv)"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import torch
from common import make, H, D
from ops import Fwd2, Dqb, Dkv
L = int(os.environ.get("L", "768")); A = int(os.environ.get("A", "48"))
kind, cubins = sys.argv[1], sys.argv[2:]
q, k, v, bias = make(A, L)
do = torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16)
fr, O, LSE = Fwd2("build/attn_fwd2.cubin").bind(q, k, v, bias); fr()
Dd = torch.randn(A, H, L, device="cuda") * 0.1


def gt(fn, reps=10, rounds=5):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    ts = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3 / reps)
    return sorted(ts)[len(ts) // 2]


ref = None
for cb in cubins:
    if kind == "fwd":
        run, X, Y = Fwd2(cb).bind(q, k, v, bias)
    elif kind == "dqb":
        run, X, Y = Dqb(cb).bind(q, k, v, do, bias, LSE, Dd, zeroed=False)
    else:
        run, X, Y = Dkv(cb).bind(q, k, v, do, bias.transpose(1, 2).contiguous(), LSE, Dd)
    if kind == "dqb":
        X.zero_()
    run(); torch.cuda.synchronize()
    out = (X.clone(), Y.clone())
    if ref is None:
        ref = out
    err = max(((a - b).norm() / b.norm().clamp_min(1e-30)).item() for a, b in zip(out, ref))
    print(f"{kind} L{L} A{A} {cb:40s} {gt(run):8.1f} us   (outputs vs {cubins[0].split('/')[-1]}: rel {err:.1e})", flush=True)
