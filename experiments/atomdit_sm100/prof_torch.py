"""Per-kernel CUDA time of the atom DiT block (torch.profiler), one implementation and mode."""
import sys, torch
from torch.profiler import profile, ProfilerActivity
from common import make_block, make_inputs
impl, mode, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
blk = make_block("pytorch" if impl in ("eager", "compile") else "miniworld")
fwd = torch.compile(blk) if impl == "compile" else blk
A = 5 if mode == "inference" else 48
s, c, z = make_inputs(A, L, grad=mode == "training")
dy = torch.randn_like(s)
def step():
    if mode == "inference":
        with torch.no_grad():
            fwd(s, c, z)
    else:
        fwd(s, c, z).backward(dy)
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(3): step()
    torch.cuda.synchronize()
rows = sorted(((e.key, e.device_time_total / 3, e.count // 3) for e in pr.key_averages() if e.device_time_total > 0), key=lambda r: -r[1])
tot = sum(r[1] for r in rows)
print(f"{impl} {mode} L{L}: total kernel time {tot:.0f} us")
for k, t, n in rows[:18]:
    print(f"  {t:9.1f} us  {100 * t / tot:5.1f}%  x{n:<3d} {k[:110]}")
