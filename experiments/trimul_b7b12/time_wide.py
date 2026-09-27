"""Time one launch configuration per process: a single captured graph, replayed.

Capturing several graphs from the same plan in one process is what the earlier sweep did, and it is unreliable here; one
graph per process is not.
"""
import argparse, json, os, statistics, torch
from wide_plan import WidePlan, R
from bench_front_wide import setup

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=384)
ap.add_argument('--width', type=int, default=256)
ap.add_argument('--mode', choices=('both', 'dW', 'dx'), default='both')
args = ap.parse_args()
if args.mode != 'both':
    os.environ['MW_ONLY_LAUNCH'] = '0' if args.mode == 'dW' else '1'
    os.environ['MW_NO_REDUCE'] = '1'

with torch.no_grad():
    a = setup(args.length, args.width)
    p = WidePlan(a, args.width)
    for _ in range(3):
        p()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            p()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        p()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        b.record(); g.replay(); e.record(); ts.append((b, e))
    torch.cuda.synchronize()
    us = sorted(x.elapsed_time(y) * 1000 for x, y in ts)
    print('TIME', args.width, args.mode, round(statistics.median(us), 1),
          dict(dw_threads=getattr(p, 'dw_threads', 0), cspan=getattr(p, 'cspan', 0), dwctas=p.dwctas,
               splits=p.splits, dxcount=p.dxcount), flush=True)
