"""Per-kernel breakdown of the wired module step (fwd + bwd) for one width.  python prof_wired.py --width 256 --length 768"""
import argparse, collections, torch
from torch.profiler import profile, ProfilerActivity
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=768); a = p.parse_args()
from miniworld_engine.modules import Transition
D, L = a.width, a.length
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
for _ in range(3): m(x).backward(dy)
torch.cuda.synchronize()
N = 5
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(N):
        m(x).backward(dy)
    torch.cuda.synchronize()
agg = collections.Counter()
for e in pr.events():
    if e.device_type.name == "CUDA":
        agg[e.name[:80]] += e.device_time_total
tot = sum(agg.values()) / N
print(f"D{D} L{L}: wired module fwd+bwd kernels {tot:.0f} us per step")
for k, v in agg.most_common(16): print(f"  {v / N:9.1f} us  {k}")
