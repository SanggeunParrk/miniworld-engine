"""Where the wide B7-B12 time goes: the same kernel with one role switched off (outputs are then partial, timing is not)."""
import argparse, json, statistics, torch
from wide_plan import WidePlan, R
from front_core import capture, paired
from bench_front_wide import setup

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=384)
ap.add_argument('--width', type=int, default=256)
args = ap.parse_args()
with torch.no_grad():
    a = setup(args.length, args.width)
    graphs, cfg = {}, None
    for name, role in (('both', 0), ('dW_only', 1), ('dx_only', 2)):
        p = WidePlan(a, args.width, role=role)
        cfg = dict(c=p.c, count=p.count, dwctas=p.dwctas, dxcount=p.dxcount, splits=p.splits, groups=p.g['groups'])
        graphs[name] = capture(p)
    times = paired(graphs)
    out = {k: round(v['median_us'], 1) for k, v in times.items()}
    print('CONFIG', cfg, flush=True)
    print('ROLES', args.length, args.width, out, flush=True)
    (R / f'records/wide-roles-L{args.length}-C{args.width}.json').write_text(json.dumps(dict(config=cfg, times=out), indent=2))
