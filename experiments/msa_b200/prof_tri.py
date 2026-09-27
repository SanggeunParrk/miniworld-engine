"""per-kernel CUDA time of one TriangleAttention call (ours / pytorch), infer and train.  python prof_tri.py ours train [L]"""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench_tri import make
from torch.profiler import profile, ProfilerActivity
impl, mode = sys.argv[1], sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 384
m = make(impl)
pair = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16); gout = torch.randn_like(pair)
if mode == "train":
    m.train(); xb = pair.clone().requires_grad_(True); params = [xb, *m.parameters()]
    f = lambda: torch.autograd.grad(m(xb), params, gout)
else:
    m.eval(); torch.set_grad_enabled(False); f = lambda: m(pair)
for _ in range(5): f()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(10): f()
    torch.cuda.synchronize()
tot = 0
rows = []
for e in p.key_averages():
    if e.device_time_total > 0:
        rows.append((e.device_time_total / 10, e.count // 10, e.key)); tot += e.device_time_total / 10
for t, c, k in sorted(rows, reverse=True): print(f"{t:8.1f} us  x{c:<3d} {k[:110]}")
print(f"{tot:8.1f} us total")
