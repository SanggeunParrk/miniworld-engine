"""Idle time between kernels inside a CUDA-graph replay of one module call (steady clocks): per replay, the span from the first
kernel start to the last kernel end vs the summed kernel time, and the largest gaps with the kernels around them.
    python gaps.py opm train"""
import os, sys, time, pathlib, torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import make_inputs, make_module, module_fn
op, mode = sys.argv[1], sys.argv[2]
L, S = 384, 1024
mod = make_module(op, "ours"); msa, pair, mask = make_inputs(op, L, S)
if mode == "infer":
    mod.eval(); torch.set_grad_enabled(False); f = module_fn(op, mod, msa, pair, mask)
else:
    mod.train(); msa.requires_grad_(True); pair.requires_grad_(True); params = [msa, pair, *mod.parameters()]
    g0 = module_fn(op, mod, msa, pair, mask)
    gout = torch.randn(msa.shape if op == "pwa" else pair.shape, device="cuda", dtype=torch.bfloat16)
    f = lambda: torch.autograd.grad(g0(), params, gout)
for _ in range(3): f()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    f(); f()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    f()
t0 = time.time()
while time.time() - t0 < 2.0: g.replay()
torch.cuda.synchronize()
R = 10
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(R): g.replay()
    torch.cuda.synchronize()
ev = sorted([e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
per = len(ev) // R
print(f"{len(ev)} kernels, {per} per replay")
spans, sums, gaps = [], [], {}
for r in range(R):
    ks = ev[r * per:(r + 1) * per]
    spans.append(ks[-1].time_range.end - ks[0].time_range.start); sums.append(sum(k.time_range.end - k.time_range.start for k in ks))
    for a, b in zip(ks, ks[1:]):
        key = (a.name[:50], b.name[:50]); gaps.setdefault(key, []).append(b.time_range.start - a.time_range.end)
print(f"span {sum(spans)/R:.1f} us   kernels {sum(sums)/R:.1f} us   idle {sum(spans)/R - sum(sums)/R:.1f} us")
for (a, b), v in sorted(gaps.items(), key=lambda kv: -sum(kv[1]))[:12]:
    print(f"  {sum(v)/len(v):6.1f} us  x{len(v)//R}  {a}  ->  {b}")
