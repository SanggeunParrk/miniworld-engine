from warp_plan import *
from check_front import LIMITS
with torch.no_grad():
 a=setup(384);ref=baseline(a);plans={};records={}
 for sp in [11,12,13,14,15,16]:
  try:
   p=WarpPlan(a,count=264,splits=sp,source='front_prefetch_lnpair_storepipe');p();torch.cuda.synchronize();es=errors(p.outputs,ref);ok=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items());records[str(sp)]=dict(errors=es,status='pass' if ok else 'incorrect')
   if ok:plans[str(sp)]=p
  except RuntimeError as e:records[str(sp)]=dict(status='rejected',reason=str(e))
 ts=paired({k:capture(p) for k,p in plans.items()});(R/'direct-store-fine-L384.json').write_text(json.dumps(dict(records=records,times=ts),indent=2));print('DIRECT_TUNE',{k:v['median_us'] for k,v in ts.items()},flush=True)
