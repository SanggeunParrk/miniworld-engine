"""Baseline only: capture and replay the Triton path with no kernel of ours in the process."""
import sys, torch
from bench_front_wide import setup, baseline
from front_core import capture
w = int(sys.argv[1]) if len(sys.argv) > 1 else 128
with torch.no_grad():
    a = setup(384, w)
    baseline(a); torch.cuda.synchronize(); print('eager ok', flush=True)
    g = capture(lambda: baseline(a)); print('captured', flush=True)
    for i in range(300):
        g.replay()
        if i % 100 == 0:
            torch.cuda.synchronize(); print('replay', i, flush=True)
    torch.cuda.synchronize(); print('BASELINE OK', w, flush=True)
