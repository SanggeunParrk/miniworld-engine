"""Per-kernel CUDA time of the H100 fused block (Triton path) on B200."""
import sys, torch
from torch.profiler import profile, ProfilerActivity
from common import make, hoist_mod, HW
import h100_fused as SF
mode, L = sys.argv[1], int(sys.argv[2])
A = 5 if mode == "inference" else 48
q, cb, cos, sin, su, w = make(A, 8 * L)
train = mode == "training"
if train:
    q.requires_grad_(); cb.requires_grad_(); [w[k].requires_grad_() for k in w]
ws = [w[k] for k in ("wqkv", "wg", "wo", "wu", "wd")]
dy = torch.randn_like(q)
def step():
    if train:
        SF.swa_block(q, hoist_mod(cb, w["wmod"]), cos, sin, su, *ws, 1, HW).backward(dy)
    else:
        with torch.no_grad():
            SF.swa_block(q, hoist_mod(cb, w["wmod"]), cos, sin, su, *ws, 1, HW)
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(3): step()
    torch.cuda.synchronize()
rows = sorted(((e.key, e.device_time_total / 3, e.count // 3) for e in pr.key_averages() if e.device_time_total > 0), key=lambda r: -r[1])
tot = sum(r[1] for r in rows)
print(f"{mode} L{L}: total kernel time {tot:.0f} us")
for k, t, n in rows[:20]:
    print(f"  {t:9.1f} us  {100 * t / tot:5.1f}%  x{n:<3d} {k[:100]}")
