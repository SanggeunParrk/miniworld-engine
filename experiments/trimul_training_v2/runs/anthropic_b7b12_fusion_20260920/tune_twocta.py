from warp_plan import *
from check_front import LIMITS
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--source',default='front_twocta_gamma');ap.add_argument('--splits',type=int,nargs='+',default=[8,10,12,14,15,16,18,20,22]);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);plans={};rec={}
 for sp in args.splits:
  try:
   p=WarpPlan(a,count=264,splits=sp,source=args.source);p();torch.cuda.synchronize();e=errors(p.outputs,ref)
   ok=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in e.items());rec[str(sp)]=dict(errors=e,status='pass' if ok else 'numerical_rejection')
   if ok:plans[str(sp)]=p
  except RuntimeError as ex:rec[str(sp)]=dict(status='build_or_launch_rejection',reason=str(ex))
 ts=paired({'baseline':capture(lambda:baseline(a)),**{k:capture(p) for k,p in plans.items()}})
 (R/f'{args.source}-tune-L{args.length}.json').write_text(json.dumps(dict(L=args.length,records=rec,times=ts),indent=2));print('TUNE',{k:v['median_us'] for k,v in ts.items()},flush=True)
