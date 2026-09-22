"""200 paired CUDA-event samples after 20 warmups, plus diagnostic time splits.

Graph contains exactly one call of each variant; baseline is core.baseline.
No compilation/allocation/descriptor packing in the measured graph. Both
normal and instrumented variants are measured to expose instrumentation cost.
"""
import argparse
import statistics
from dual_experiment import *
from check_experiment import check, change_inputs


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        fn()
    return g


def paired_events(graphs, warmup=20, iterations=200):
    for _ in range(warmup):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    events = {k: [] for k in graphs}
    keys = list(graphs)
    for iteration in range(iterations):
        order = keys if iteration % 2 == 0 else list(reversed(keys))
        for k in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[k].replay()
            end.record()
            events[k].append((start, end))
    torch.cuda.synchronize()
    result = {}
    for k, pairs in events.items():
        times = sorted(start.elapsed_time(end) * 1000 for start, end in pairs)
        result[k] = dict(median_us=statistics.median(times), p90_us=times[int(.9 * (len(times)-1))],
                         min_us=times[0], max_us=times[-1], samples_us=times)
    return result


def read_phases(plan, iterations=20):
    records = []
    for _ in range(iterations):
        plan()
        torch.cuda.synchronize()
        t = plan.timestamps.flatten()[:plan.grid * 6].reshape(plan.grid, 6).cpu().tolist()
        assert all(all(row[i+1] >= row[i] for i in range(5)) for row in t)
        # Report per-CTA durations AND critical completion frontiers. Median
        # CTA durations do not add to end-to-end kernel latency across CTAs.
        duration = [[(row[i+1]-row[i])/1000 for i in range(5)] for row in t]
        endpoints = [min(row[0] for row in t)] + [max(row[i] for row in t) for i in range(1,6)]
        critical = [(endpoints[i+1]-endpoints[i])/1000 for i in range(5)]
        latest = max(range(len(t)), key=lambda i: t[i][5])
        records.append(dict(cta_ns=t, critical_frontier_us=critical,
                            latest_cta=latest, latest_cta_phase_us=duration[latest],
                            cta_median_us=[statistics.median(x[i] for x in duration) for i in range(5)],
                            cta_max_us=[max(x[i] for x in duration) for i in range(5)]))
    names = ('body', 'partial_dump_and_publish', 'grid_wait', 'reduce_and_publish', 'reset')
    return dict(phase_names=names, timebase='PTX globaltimer ns',
                notes='Diagnostic-only. Completion-frontier deltas add; independent CTA medians do not.',
                median_critical_frontier_us={k: statistics.median(r['critical_frontier_us'][i] for r in records) for i,k in enumerate(names)},
                median_cta_us={k: statistics.median(r['cta_median_us'][i] for r in records) for i,k in enumerate(names)},
                records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', default=['dual', 'dual_dspref', 'dual_dspref_timing'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[384, 768])
    parser.add_argument('--count', type=int, default=132)
    parser.add_argument('--parts', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--output', default='diagnostic-results.json')
    args = parser.parse_args()
    rows = []
    with torch.no_grad():
        for n in args.lengths:
            d, dy, saved = data(n)
            change_inputs(d, dy, .25, 20260920 + n)
            ref = baseline(d, dy, saved)
            plans = {f'{source}/part{part}': Experiment(d,dy,saved,args.count,part,source)
                     for source in args.sources for part in args.parts}
            errors = {key: check(p(),ref) for key,p in plans.items()}
            print('CHECK', n, json.dumps(errors), flush=True)
            funcs = dict(baseline=lambda: baseline(d,dy,saved), **plans)
            if args.full:
                from integrate import backward_cuda
                sys.path.insert(0,str(R.parent/'anthropic_ln_equal_saves_20260919'))
                import core_saved as C
                funcs['full/baseline'] = lambda: C.backward(d,saved,dy)
                full_ref = C.backward(d,saved,dy)
                for key,p in plans.items():
                    out = backward_cuda(d,saved,dy,p)
                    full_errors = [rel(x,y) for x,y in zip(out,full_ref)]
                    assert max(full_errors) <= 5e-4, full_errors
                    errors['full/'+key] = full_errors
                    funcs['full/'+key] = lambda p=p: backward_cuda(d,saved,dy,p)
            graphs = {k: capture(f) for k,f in funcs.items()}
            times = paired_events(graphs)
            phases = {k:read_phases(p) for k,p in plans.items() if k.split('/')[0].endswith('_timing')}
            row = dict(L=n,dropout=.25,count=args.count,warmup=20,iterations=200,
                       errors=errors,times=times,phases=phases)
            rows.append(row)
            print('RESULT', n, json.dumps({k:{m:v for m,v in z.items() if m!='samples_us'} for k,z in times.items()}), flush=True)
            print('PHASES', n, json.dumps({k:v['median_critical_frontier_us'] for k,v in phases.items()}), flush=True)
            (R/args.output).write_text(json.dumps(rows,indent=2))
