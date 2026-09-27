"""Timeline of ONE CUDA-graph replay of the wired module step: kernels in order with the idle gap before each."""
import argparse, os, torch
from torch.profiler import profile, ProfilerActivity
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=768)
p.add_argument("--wide", type=int, default=1); a = p.parse_args()
os.environ["MINIWORLD_TRANSITION_WIDE_SM90A"] = str(a.wide)
from miniworld_engine.modules import Transition
D, L = a.width, a.length
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
P = list(m.parameters())
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
def step():
    for q in [x] + P: q.grad = None
    m(x).backward(dy)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): step()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): step()
for _ in range(3): g.replay()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    g.replay(); torch.cuda.synchronize()
ev = sorted([e for e in pr.events() if e.device_type.name == "CUDA"], key=lambda e: e.time_range.start)
t0 = ev[0].time_range.start; prev = t0; busy = 0
for e in ev:
    gap = e.time_range.start - prev
    dur = e.time_range.end - e.time_range.start; busy += dur
    if dur > 5 or gap > 5: print(f"  +{(e.time_range.start - t0):8.1f}  gap {gap:7.1f}  dur {dur:8.1f}  {e.name[:70]}")
    prev = max(prev, e.time_range.end)
print(f"D{D} L{L} wide={a.wide}: span {prev - t0:.1f} us, kernels {busy:.1f} us")
