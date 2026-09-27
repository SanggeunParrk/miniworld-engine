"""Per-kernel timing for the split wide path, over the dW occupancy/ring-depth choices.

Each launch is timed on its own (captured alone), so the two roles are separated without a ROLE_ONLY rebuild.
"""
import argparse, json, os, statistics, torch
from wide_plan import WidePlan, R
from front_core import capture, paired
from bench_front_wide import setup

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=384)
ap.add_argument('--width', type=int, default=256)
ap.add_argument('--configs', default='c1b2a0,c1b2a1')        # dW channel span, CTAs per SM, GLU-ahead
args = ap.parse_args()

with torch.no_grad():
    a = setup(args.length, args.width)
    rows = {}
    for spec in args.configs.split(','):
        cspan, blocks, ahead = (int(v) for v in spec.lstrip('c').replace('b', ' ').replace('a', ' ').split())
        os.environ['MW_DW_CSPAN'], os.environ['MW_DW_BLOCKS'] = str(cspan), str(blocks)
        os.environ['MW_DW_GLU_AHEAD'] = str(ahead)
        try:
            p = WidePlan(a, args.width, split=True, allow_spills=True)
        except ValueError as exc:
            print('SKIP', spec, exc, flush=True)
            continue
        graphs = {}
        for name, (k, grid, shared, threads) in zip(('dW', 'dx'), p.launches):
            graphs[name] = capture(lambda k=k, grid=grid, shared=shared, threads=threads:
                                   k.launch((grid, 1, 1), (threads, 1, 1), [p.p], shared))
        graphs['both'] = capture(p)
        t = paired(graphs)
        rows[spec] = {k: round(v['median_us'], 1) for k, v in t.items()}
        rows[spec]['dwctas'] = p.dwctas
        rows[spec]['dw_threads'] = p.dw_threads
        rows[spec]['splits'] = p.splits
        print('SWEEP', args.width, spec, rows[spec], flush=True)
    (R / f'records/wide-sweep-L{args.length}-C{args.width}.json').write_text(json.dumps(rows, indent=2))
