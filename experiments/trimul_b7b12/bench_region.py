"""Benchmark the qualified B7-B12 replacement against the existing reference."""
from ring_plan import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, choices=(384, 768), required=True)
args = ap.parse_args()
with torch.no_grad():
    a = setup(args.length)
    cls = WarpPlan if args.length == 384 else RingPlan
    p = cls(a)
    reference = baseline(a)
    actual = p()
    torch.cuda.synchronize()
    es = errors(actual, reference)
    assert all(e['finite'] and e['relative_l2'] <= LIMITS[k] for k,e in es.items()), es
    graphs = {'baseline': capture(lambda: baseline(a)), 'cuda': capture(p)}
    blocks = [paired(graphs) for _ in range(3)]
    times = {}
    for name in graphs:
        samples = sorted(t for block in blocks for t in block[name]['samples_us'])
        times[name] = dict(median_us=statistics.median(samples), samples_us=samples)
    record = dict(L=args.length, scope='B7-B12', source=p.source, dropout=.25,
                  errors=es, times=times, blocks=blocks)
    (R/f'new-region-L{args.length}.json').write_text(json.dumps(record, indent=2))
    print({name:v['median_us'] for name,v in times.items()})
