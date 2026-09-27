"""Run the dx kernel with bounded waits and report which barrier site stalled."""
import os, torch
from wide_plan import WidePlan
from bench_front_wide import setup
os.environ.setdefault('MW_ONLY_LAUNCH', '1')
with torch.no_grad():
    a = setup(384, 256)
    p = WidePlan(a, 256, allow_spills=True)
    for i in range(400):
        p()
        if i % 50 == 0:
            torch.cuda.synchronize()
            hit = p.counts.nonzero().flatten().tolist()
            if hit:
                print('STALL at replay', i, [(h // 4, ['h', 'weights', 'gate', 'xres'][h % 4],
                                              int(p.counts[h])) for h in hit[:8]], flush=True)
                raise SystemExit
            print('ok', i, flush=True)
    torch.cuda.synchronize(); print('no stall in 400', flush=True)
