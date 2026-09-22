from warp_plan import *
from check_front import LIMITS
with torch.no_grad():
 a=setup(768);ref=baseline(a);gs={};rec={}
 for src in ['front_kindprefetch', 'front_kindmap_a0_n8', 'front_kindmap_a8_n8', 'front_kindmap_a16_n8', 'front_kindmap_a8_n16', 'front_kindmap_a16_n16']:
  try:
   p=WarpPlan(a,count=264,splits=13,source=src);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
   if not all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in es.items()):rec[src]=dict(status='incorrect',errors=es);continue
   gs[src]=capture(p);rec[src]=dict(status='pass',errors=es,plan=p)
  except RuntimeError as e:rec[src]=dict(status='rejected',reason=str(e))
 ts=paired(gs)
 for k,v in rec.items():v.pop('plan',None);v['time']=ts.get(k)
 (R/'kind-prefetch-rolemap-sweep-L768.json').write_text(json.dumps(rec,indent=2));print('RESULT',{k:(v.get('time') or {}).get('median_us',v['status']) for k,v in rec.items()},flush=True)
