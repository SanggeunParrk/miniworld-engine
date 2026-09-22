from warp_plan import *
from check_front import LIMITS
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);plans={};rec={}
 for count in [216,224,232,240,248,256,264]:
  mid=round(count*13/264)
  for split in [mid-1,mid,mid+1]:
   key=f'c{count}s{split}'
   try:
    p=WarpPlan(a,count=count,splits=split,source='front_prefetch_lnpair');p();torch.cuda.synchronize();es=errors(p.outputs,ref)
    ok=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items());rec[key]=dict(status='pass' if ok else 'incorrect',errors=es)
    if ok:plans[key]=p
   except RuntimeError as e:rec[key]=dict(status='rejected',reason=str(e))
 gs={k:capture(p) for k,p in plans.items()};ts=paired(gs)
 (R/f'lnpair-grid-L{args.length}.json').write_text(json.dumps(dict(records=rec,times=ts),indent=2));print('GRID',sorted([(k,v['median_us']) for k,v in ts.items()],key=lambda x:x[1]),flush=True)
