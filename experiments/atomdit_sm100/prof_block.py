"""Per-kernel CUDA time of the sm_100a atom DiT block (training, A = 48)."""
import sys, torch
from torch.profiler import profile, ProfilerActivity
from common import make_block, make_inputs
from atom_block import AtomBlock
L = int(sys.argv[1]) if len(sys.argv) > 1 else 384
mode = sys.argv[2] if len(sys.argv) > 2 else "training"
ours = AtomBlock(make_block("pytorch"))
A = 48 if mode == "training" else 5
s, c, z = make_inputs(A, L, grad=mode == "training")
dy = torch.randn_like(s)
def step():
    if mode == "training":
        ours(s, c, z).backward(dy)
    else:
        with torch.no_grad():
            ours(s, c, z)
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(3): step()
    torch.cuda.synchronize()
rows = sorted(((e.key, e.device_time_total / 3, e.count // 3) for e in pr.key_averages() if e.device_time_total > 0), key=lambda r: -r[1])
tot = sum(r[1] for r in rows)
print(f"{mode} L{L}: total kernel time {tot:.0f} us")
for k, t, n in rows[:25]:
    print(f"  {t:9.1f} us  {100 * t / tot:5.1f}%  x{n:<3d} {k[:100]}")
