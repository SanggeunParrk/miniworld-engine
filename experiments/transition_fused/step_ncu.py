"""One wired fwd+bwd step inside an NVTX range 'measure' (after warm-up), for NCU fixed-clock kernel-time sums."""
import argparse, torch
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=384); p.add_argument("--length", type=int, default=384); p.add_argument("--count", action="store_true"); a = p.parse_args()
from miniworld_engine import settings
from miniworld_engine.modules import Transition
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
D, L = a.width, a.length
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
P = list(m.parameters())
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
def step():
    for q in [x] + P: q.grad = None
    y = m(x)
    torch.cuda.nvtx.range_push("bwd"); y.backward(dy); torch.cuda.nvtx.range_pop()
for _ in range(3): step()
torch.cuda.synchronize()
if a.count:                                               # kernels per step, so NCU can skip exactly the warm-up steps
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        step(); torch.cuda.synchronize()
    print("KERNELS", sum(1 for e in pr.events() if e.device_type.name == "CUDA" and "Memcpy" not in e.name and "Memset" not in e.name))
else:
    step(); torch.cuda.synchronize(); print("ok")
