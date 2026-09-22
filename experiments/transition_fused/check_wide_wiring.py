"""End-to-end check of the wired wide-width path through modules.Transition (default settings): output + six gradients vs
fp32 and vs the same module with MINIWORLD_TRANSITION_WIDE_SM90A=0, the no_grad path, then CUDA-graph timing of both.
   python check_wide_wiring.py --width 256 --length 384"""
import argparse, copy, os, statistics
import torch
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=384); a = p.parse_args()
from miniworld_engine.modules import Transition
from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as W
D, L = a.width, a.length; bf = torch.bfloat16
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()      # the default resolves to PyTorch
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
        elif prm is m.ln_in.weight: prm.copy_(1 + 0.2 * torch.randn_like(prm))
        else: prm.normal_(std=0.2)
P = list(m.parameters())
x = torch.randn(1, L, L, D, device="cuda", dtype=bf, requires_grad=True); dy = torch.randn_like(x)
print(f"D{D} L{L}: supported={W.supported(x, m.expand_a.weight, m.squeeze.weight)} available={W.available(x, m.expand_a.weight, m.squeeze.weight)}")
ref = copy.deepcopy(m).float(); xr = x.detach().float().requires_grad_(True); yr = ref(xr); yr.backward(dy.float())
R = [yr.detach(), xr.grad] + [q.grad for q in ref.parameters()]
names = ["y", "dx"] + [n for n, _ in ref.named_parameters()]
del ref, xr, yr
def run(wide):
    os.environ["MINIWORLD_TRANSITION_WIDE_SM90A"] = "1" if wide else "0"
    for q in [x] + P: q.grad = None
    y = m(x); y.backward(dy)
    return [y.detach()] + [x.grad.clone()] + [q.grad.clone() for q in P]
rel = lambda u, r: float((u.float().reshape(-1) - r.reshape(-1)).norm() / r.norm())
Gw, Ge = run(True), run(False)
worse = []
for n, gw, ge, r in zip(names, Gw, Ge, R):
    ew, ee = rel(gw, r), rel(ge, r)
    print(f"  {n:22s} wired {ew:.2e} | engine {ee:.2e}  dtype {gw.dtype}")
    if ew > 1.1 * ee: worse.append(n)
os.environ["MINIWORLD_TRANSITION_WIDE_SM90A"] = "1"
with torch.no_grad():
    yi = m(x)
print(f"  no_grad forward vs grad forward: max |diff| {float((yi - Gw[0]).abs().max()):.3e}  (vs fp32 {rel(yi, R[0]):.2e})")
Gw2 = run(True); rep = all(torch.equal(u, v) for u, v in zip(Gw, Gw2))
print(f"  bit-reproducible: {rep}   worse than engine by >10 %: {worse or 'none'}")
def graphed(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g.replay
def t(fn, reps=5):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(7):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
res = {}
for wide in (False, True):
    os.environ["MINIWORLD_TRANSITION_WIDE_SM90A"] = "1" if wide else "0"
    def step():
        for q in [x] + P: q.grad = None
        m(x).backward(dy)
    def inf():
        with torch.no_grad(): return m(x)
    res[wide] = (t(graphed(inf)), t(graphed(lambda: m(x))), t(graphed(step)))
(ei, ef, efb), (wi, wf, wfb) = res[False], res[True]
print(f"  timing (CUDA graph, us)  engine -> wired: inference fwd {ei:.1f} -> {wi:.1f} (x{ei / wi:.2f}) | train fwd {ef:.1f} -> {wf:.1f} (x{ef / wf:.2f})"
      f" | fwd+bwd {efb:.1f} -> {wfb:.1f} (x{efb / wfb:.2f}) | bwd {efb - ef:.1f} -> {wfb - wf:.1f} (x{(efb - ef) / (wfb - wf):.2f})")
