"""Bounded waits in the dW role: report which ring slot never filled."""
import os, torch
os.environ.setdefault('MW_ONLY_LAUNCH', '0'); os.environ.setdefault('MW_NO_REDUCE', '1')
from wide_plan import WidePlan
from bench_front_wide import setup
with torch.no_grad():
    a = setup(384, 256)
    p = WidePlan(a, 256, allow_spills=True)
    for i in range(3):
        p(); torch.cuda.synchronize()
        hit = p.counts.nonzero().flatten().tolist()
        print('iter', i, 'stalls', [(h // 4, h % 4, int(p.counts[h])) for h in hit[:10]] or 'none', flush=True)
