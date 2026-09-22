"""Exhaust the existing saved-training K3 schedule space; no algorithm changes."""
from compare_cueq_training import *
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=int, choices=(384, 768), required=True)
    args = ap.parse_args()
    n = args.length
    cfgs = list(T.candidates(128, 256))
    print('BUILD_CANDIDATES', len(cfgs), flush=True)
    failures = []
    built = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        pending = {ex.submit(T.build, 128, 256, c): c for c in cfgs}
        for fut in as_completed(pending):
            c = pending[fut]
            try:
                fut.result()
                built.append(c)
            except Exception as err:
                failures.append(dict(config=c, error=str(err)))
    print('BUILT', len(built), 'FAILED', len(failures), flush=True)
    with torch.no_grad():
        d = C.setup(n)
        _, saved = C.forward(d, True, (3, 64, 2, 2, 1), (1, 1))
        vals = saved[0].saved_tensors
        xn, tri = vals[0], vals[11]
        def run(cfg):
            return T.output_training(tri, xn.reshape(n, n, 128), d['wp'],
                d['leaves'][5], d['go'], d['bo'], d['x'].reshape(n*n,128),
                d['ds'], 1e-5, list(cfg))
        ref = run(T.default_config(256))
        records, times = {}, {}
        for cfg in sorted(built):
            output = run(cfg)
            torch.cuda.synchronize()
            errors = [dict(relative_l2=rel(a,b), exact=bool(torch.equal(a,b)),
                           finite=bool(torch.isfinite(a).all())) for a,b in zip(output,ref)]
            # BF16 saves must match exactly; FP32 LN statistics may reorder.
            valid = all(e['finite'] and (e['exact'] if a.dtype == torch.bfloat16
                          else e['relative_l2'] <= 1e-6) for a,e in zip(output,errors))
            key = ','.join(map(str,cfg))
            records[key] = dict(config=cfg, valid=valid, errors=errors)
            if valid:
                graph, graph_outputs = capture_outputs(lambda cfg=cfg: run(cfg))
                times[key] = paired({key:graph}, iterations=100)[key]
                del graph, graph_outputs
            else:
                print('REJECT', cfg, errors, flush=True)
        ranked = sorted(times, key=lambda k:times[k]['median_us'])
        # Independent longer confirmation of shortlist and starting schedule.
        default = ','.join(map(str,T.default_config(256)))
        finalists = {}
        for key in dict.fromkeys(ranked[:5]+[default]):
            cfg = records[key]['config']
            finalists[key], _ = capture_outputs(lambda cfg=cfg:run(cfg))
        confirmed = pool([paired(finalists, iterations=200) for _ in range(3)])
        winner = min(confirmed, key=lambda k:confirmed[k]['median_us'])
        report = dict(L=n, dropout=.25, saves=True, candidate_count=len(cfgs),
            build_failures=failures, candidates=records, screening=times,
            confirmed=confirmed, winner=records[winner]['config'],
            note='Existing candidate space only; all BF16 outputs/saves bit-exact vs default, FP32 stats relative L2 <=1e-6')
        (R/('training-k3-audit-L%d.json'%n)).write_text(json.dumps(report,indent=2))
        print('WINNER',report['winner'], 'CONFIRMED', {k:v['median_us'] for k,v in confirmed.items()},flush=True)


if __name__ == '__main__':
    main()
