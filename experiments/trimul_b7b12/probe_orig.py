"""The qualified C = 128 kernel (front_prefetch_lnpair_storepipe), alone in a graph, replayed back to back."""
import torch
from warp_plan import WarpPlan
from front_core import setup, capture
with torch.no_grad():
    a = setup(384)
    p = WarpPlan(a, count=264, splits=13, source='front_prefetch_lnpair_storepipe')
    p(); torch.cuda.synchronize(); print('eager ok', flush=True)
    g = capture(p); print('captured', flush=True)
    for i in range(600):
        g.replay()
        if i % 100 == 0:
            torch.cuda.synchronize(); print('replay', i, flush=True)
    torch.cuda.synchronize(); print('ORIGINAL OK', flush=True)
