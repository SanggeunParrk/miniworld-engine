from front_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=384);ap.add_argument('--sources',nargs='+',default=['front_selected']);ap.add_argument('--splits',type=int,default=15);ap.add_argument('--output',required=True);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);plans={k:Plan(a,splits=args.splits,source=k) for k in args.sources};checks={k:errors(p(),ref) for k,p in plans.items()}
 assert all(v['finite'] and v['relative_l2']<5e-4 for e in checks.values() for v in e.values())
 gs={'baseline':capture(lambda:baseline(a)),**{k:capture(p) for k,p in plans.items()}};blocks=[paired(gs) for _ in range(3)];times={}
 for k in gs:
  ts=sorted(v for b in blocks for v in b[k]['samples_us']);times[k]=dict(median_us=statistics.median(ts),p90_us=ts[int(.9*(len(ts)-1))],samples_us=ts)
 (R/args.output).write_text(json.dumps(dict(L=args.length,dropout=.25,splits=args.splits,errors=checks,times=times,blocks=blocks),indent=2))
 print('COMPARE',{k:v['median_us'] for k,v in times.items()},flush=True)
