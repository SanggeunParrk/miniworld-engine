from warp_plan import *
from check_front import LIMITS
import argparse,faulthandler
faulthandler.dump_traceback_later(60,repeat=True)
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--count',type=int,default=264);ap.add_argument('--splits',type=int,default=8);ap.add_argument('--sources',nargs='+',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);plans={};rec={}
 for src in args.sources:
  try:
   p=WarpPlan(a,count=args.count,splits=args.splits,source=src);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
   if any(v['relative_l2']>LIMITS[k] or not v['finite'] for k,v in es.items()):rec[src]=dict(status='numerical_rejection',errors=es);continue
   plans[src]=p;rec[src]=dict(status='initial_pass',errors=es)
  except RuntimeError as e:rec[src]=dict(status='rejected_before_launch',reason=str(e))
 gs={'baseline':capture(lambda:baseline(a)),**{src:capture(p) for src,p in plans.items()}};ts=paired(gs)
 (R/args.output).write_text(json.dumps(dict(L=args.length,splits=args.splits,records=rec,times=ts),indent=2));print('WARP_VARIANTS',{k:v['median_us'] for k,v in ts.items()},'STATUS',{k:v['status'] for k,v in rec.items()},flush=True)
faulthandler.cancel_dump_traceback_later()
