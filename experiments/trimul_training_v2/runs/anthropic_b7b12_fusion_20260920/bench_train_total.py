"""Direct forward+backward comparison with only B7-B12 replaced.

Uses the qualified shape-specific sources, identical fused-LN forward,
identical B1-B6, fixed 25% dropout and saved tensors freshly produced
inside each captured training call. No addition of isolated timings.
"""
from ring_plan import *
from integrate_front import backward_full
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, choices=(384, 768), required=True)
ap.add_argument('--source', help='Explicit candidate source; requires a distinct record label')
ap.add_argument('--label', default='qualified')
args = ap.parse_args()
if args.source and args.label == 'qualified':
    ap.error('--source requires --label to preserve qualified checkpoint records')
n = args.length
cfg = (3, 64, 2, 2, 1)
names = ('dx', 'dWL', 'dWLg', 'dWR', 'dWRg', 'dWgate', 'dWproj',
         'dgamma_in', 'dbeta_in', 'dgamma_out', 'dbeta_out')
limits = (2e-5, *([5e-4] * 6), *([5e-6] * 4))

def validate(out, ref):
    result = {}
    for name, x, y, limit in zip(names, out, ref, limits):
        error = rel(x, y)
        finite = bool(torch.isfinite(x).all())
        assert finite and error <= limit, (name, error, limit)
        result[name] = dict(relative_l2=error, limit=limit, finite=finite)
    return result

def capture_outputs(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = fn()
    return graph, outputs

with torch.no_grad():
    a = setup(n)
    if n == 384:
        source = args.source or 'front_prefetch_lnpair_storepipe'
        p = WarpPlan(a, count=264, splits=13, source=source)
    else:
        source = args.source or 'front_ring96_cache3'
        p = RingPlan(a, count=264, splits=20, source=source)

    def train_reference():
        y, saves = C.forward(a['d'], True, cfg, (1, 1))
        return y, C.backward(a['d'], saves, a['dy'])

    def train_candidate():
        y, saves = C.forward(a['d'], True, cfg, (1, 1))
        a['s'] = saves
        ctx, a['mu'], a['rs'] = saves
        (a['xn'], a['wl'], a['wlg'], a['wr'], a['wrg'], a['wg'],
         _, _, a['pre'], *_) = ctx.saved_tensors
        return y, backward_full(a, p)

    ref = train_reference()
    out = train_candidate()
    torch.cuda.synchronize()
    assert torch.equal(out[0], ref[0])
    initial_errors = validate(out[1], ref[1])
    print('INITIAL_PASS', n, flush=True)

    graphs = {}
    outputs = {}
    for name, fn in (('baseline', train_reference), ('b7b12_cuda', train_candidate)):
        graphs[name], outputs[name] = capture_outputs(fn)
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(outputs['baseline'][0], outputs['b7b12_cuda'][0])
    replay_errors = validate(outputs['b7b12_cuda'][1], outputs['baseline'][1])
    assert bool((p.counts[:2] == 0).all())
    print('GRAPH_PASS', n, flush=True)

    blocks = [paired(graphs) for _ in range(3)]
    times = {}
    for name in graphs:
        samples = sorted(t for block in blocks for t in block[name]['samples_us'])
        times[name] = dict(median_us=statistics.median(samples),
                           p90_us=samples[int(.9 * (len(samples)-1))],
                           samples_us=samples)
    speedup = times['baseline']['median_us'] / times['b7b12_cuda']['median_us']
    record = dict(L=n, C=128, packed_hidden=256, dtype='bfloat16', dropout=.25,
                  source=source, forward_config=cfg,
                  scope='direct forward + full backward; identical fused-input-LN forward and B1-B6; only B7-B12 replaced',
                  excludes=['optimizer', 'RNG', 'CPU/autograd dispatch', 'weight packing', 'compilation'],
                  baseline_note='existing Triton/cuBLAS backward; input-dual cache-miss fallback uses 3 configurations, not exhaustive retuning',
                  device=torch.cuda.get_device_name(), initial_errors=initial_errors,
                  replay_errors=replay_errors, times=times, blocks=blocks, speedup=speedup)
    (R / f'train-total-{args.label}-L{n}.json').write_text(json.dumps(record, indent=2))
    print('TRAIN_TOTAL', n, {k:v['median_us'] for k,v in times.items()}, 'speedup', speedup, flush=True)
