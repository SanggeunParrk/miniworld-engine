"""Per-kernel CUDA time of FusedTokenDiT.step (eager replays after warmup)."""
import argparse, collections, sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, str(Path(__file__).resolve().parent))
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType
from tdit import FusedTokenDiT
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--blocks", type=int, default=4)
a = p.parse_args(); L, S, NB, dev, bf = a.length, 5, a.blocks, "cuda", torch.bfloat16
torch.manual_seed(0)
blocks = torch.nn.ModuleList(DiTBlock(768, 384, 128, 16, n=2, implementation=ImplementationType.PYTORCH) for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in blocks.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[1] ** -0.5)
blocks = blocks.to(bf).eval()
f = FusedTokenDiT(blocks)
s = torch.randn(S, 1, L, 768, device=dev, dtype=bf); c = torch.randn(1, 1, L, 384, device=dev, dtype=bf).expand(S, 1, L, 384).contiguous()
z = torch.randn(1, L, L, 128, device=dev, dtype=bf)
with torch.no_grad():
    bias = f.hoist(z)
    for _ in range(5): f.step(s, c, bias)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10): f.step(s, c, bias)
        torch.cuda.synchronize()
per = collections.defaultdict(lambda: [0.0, 0])
for e in prof.key_averages():
    t = getattr(e, "self_device_time_total", 0)
    if t > 0: per[e.key][0] += t / 10 / NB; per[e.key][1] += e.count // 10
tot = sum(v[0] for v in per.values())
print(f"L={L} S={S}: {tot:.1f} us per block (kernel time), launches per step {sum(v[1] for v in per.values())}")
for k, (t, c) in sorted(per.items(), key=lambda x: -x[1][0])[:14]: print(f"  {t:7.1f} us/block  x{c:<3d} {k[:90]}")
