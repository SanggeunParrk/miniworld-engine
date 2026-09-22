from warp_plan import *
from check_front import LIMITS
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=768);ap.add_argument('--name',default='candidates');ap.add_argument('sources',nargs='+');args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);rec={};gs={}
 for src in args.sources:
  try:
   p=WarpPlan(a,count=264,splits=13,source=src);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
   if not all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in es.items()):rec[src]=dict(status='incorrect',errors=es);print('INCORRECT',src,es,flush=True);continue
   gs[src]=capture(p);rec[src]=dict(status='pass',errors=es,plan=p)
  except RuntimeError as e:rec[src]=dict(status='rejected',reason=str(e));print('REJECTED',src,str(e),flush=True)
 ts=paired(gs)
 for k,v in rec.items():v.pop('plan',None);v['time']=ts.get(k)
 (R/f'{args.name}-L{args.length}.json').write_text(json.dumps(rec,indent=2));print('RESULT',{k:(v.get('time') or {}).get('median_us',v['status']) for k,v in rec.items()},flush=True)
