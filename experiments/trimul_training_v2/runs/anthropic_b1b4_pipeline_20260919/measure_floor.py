"""Diagnostic timing without correctness checks (traffic-floor replicas produce meaningless outputs)."""
import argparse,statistics
from dual_experiment import *
from check_experiment import change_inputs
from measure_experiment import capture,paired_events
a=argparse.ArgumentParser();a.add_argument('--sources',nargs='+',default=['dual_ln_prefetch','dual_traffic_floor']);a.add_argument('--lengths',type=int,nargs='+',default=[384,768]);a.add_argument('--output',default='claude-floor.json');args=a.parse_args()
rows=[]
with torch.no_grad():
 for n in args.lengths:
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n)
  plans={k:Experiment(d,dy,s,132,2,k) for k in args.sources}
  for p in plans.values():p();torch.cuda.synchronize()
  graphs={k:capture(p) for k,p in plans.items()}
  blocks=[paired_events(graphs) for _ in range(3)]
  times={}
  for k in graphs:
   smp=sorted(t for b in blocks for t in b[k]['samples_us']);times[k]=dict(median_us=statistics.median(smp),p90_us=smp[int(.9*(len(smp)-1))],min_us=smp[0])
  print('FLOOR',n,json.dumps({k:{m:round(v,2) for m,v in z.items()} for k,z in times.items()}),flush=True)
  rows.append(dict(L=n,times=times));(R/args.output).write_text(json.dumps(rows,indent=2))
