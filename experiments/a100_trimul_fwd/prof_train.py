"""Per-kernel GPU time of one training step (fwd + bwd).  python prof_train.py --variant bidir --length 768"""
import argparse, collections, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA, trimul_train as TT  # noqa: E401
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="bidir"); ap.add_argument("--length", type=int, default=768)
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args(); ext = TA.build(extra=[x for e in a.extra for x in e.split()])
L = a.length
mod = (BidirectionalTriangleMultiplication if a.variant == "bidir" else TriangleMultiplication)(128, implementation=I.PYTORCH, p_drop=.25)
mod = mod.cuda().bfloat16().train()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.rand(1, L, device="cuda") > .1
ds = ((torch.rand(1, 1, L, 128, device="cuda") > .25).to(torch.bfloat16) / .75).to(torch.bfloat16)
dy = torch.randn_like(z).detach()
params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
step = lambda: torch.autograd.grad(TT.forward_train(ext, mod, z, mask, ds), [z] + params, dy)  # noqa: E731
for _ in range(5): step()
torch.cuda.synchronize()
N = 5
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(N): step()
    torch.cuda.synchronize()
agg = collections.defaultdict(lambda: [0.0, 0])
for e in prof.events():
    if e.device_type.name == "CUDA":
        k = e.name[:70]; agg[k][0] += e.device_time; agg[k][1] += 1
tot = sum(v[0] for v in agg.values()) / N
print(f"{a.variant} L{L}: total GPU {tot/1000:.3f} ms")
for k, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    if t / N < 3: continue
    print(f"  {t/N:8.1f} us  x{c//N:<3d} {100*t/N/tot:5.1f}%  {k}")
