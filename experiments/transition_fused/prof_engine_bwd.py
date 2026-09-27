"""Kernel-by-kernel breakdown of the engine module's Transition backward (training step) per width."""
import argparse, collections, torch
from torch.profiler import profile, ProfilerActivity
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=384); a = p.parse_args()
from miniworld_engine import settings
from miniworld_engine.modules import Transition
D, L = a.width, a.length
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
for _ in range(3):
    y = m(x); y.backward(dy)
torch.cuda.synchronize()
N = 5
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(N):
        y = m(x)
        torch.cuda.synchronize()
        y.backward(dy)
    torch.cuda.synchronize()
agg = collections.Counter()
for e in pr.events():
    if e.device_type.name == "CUDA":
        agg[e.name[:70]] += (e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
tot = sum(agg.values()) / N
print(f"D{D} L{L}: engine module fwd+bwd kernels {tot:.0f} us per step")
for n, v in agg.most_common(14):
    print(f"  {v / N:8.1f} us  {n}")
