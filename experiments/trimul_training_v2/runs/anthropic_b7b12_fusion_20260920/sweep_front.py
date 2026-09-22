from front_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=384);ap.add_argument('--source',default='front_mn_nostack');ap.add_argument('--splits',type=int,nargs='+',default=[6,8,10,12,14,16,18,20]);ap.add_argument('--output',required=True);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);plans={};checks={}
 for sp in args.splits:
  k=f'splits{sp}';p=Plan(a,splits=sp,source=args.source);e=errors(p(),ref);torch.cuda.synchronize()
  assert all(v['finite'] and v['relative_l2']<5e-4 for v in e.values()),e
  plans[k]=p;checks[k]=e
 gs={'baseline':capture(lambda:baseline(a)),**{k:capture(p) for k,p in plans.items()}}
 times=paired(gs);record=dict(L=args.length,source=args.source,errors=checks,times=times)
 (R/args.output).write_text(json.dumps(record,indent=2));print('SWEEP',{k:v['median_us'] for k,v in times.items()},flush=True)
