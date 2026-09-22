from ring_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--source',default='front_ring256_pipe');args=ap.parse_args()
with torch.no_grad():
 a=setup(768);ref=baseline(a);plans={};records={}
 for sp in [10,12,13,14,16,18,20]:
  try:
   p=RingPlan(a,count=264,splits=sp,source=args.source);p();torch.cuda.synchronize();es=errors(p.outputs,ref);ok=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items());records[str(sp)]=dict(errors=es,status='pass' if ok else 'incorrect')
   if ok:plans[str(sp)]=p
  except RuntimeError as e:records[str(sp)]=dict(status='rejected',reason=str(e))
 ts=paired({k:capture(p) for k,p in plans.items()});(R/f'{args.source}-tune.json').write_text(json.dumps(dict(records=records,times=ts),indent=2));print('RING_TUNE',{k:v['median_us'] for k,v in ts.items()},flush=True)
