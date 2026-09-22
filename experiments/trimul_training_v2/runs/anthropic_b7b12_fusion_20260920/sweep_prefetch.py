from warp_plan import *
from check_front import LIMITS
with torch.no_grad():
 a=setup(768);ref=baseline(a);gs={};rec={}
 for src,split in [('front_kindprefetch', 11), ('front_kindprefetch', 12), ('front_kindprefetch', 13), ('front_kindprefetch', 14), ('front_kindprefetch', 15), ('front_kindprefetch', 16)]:
  key=src+'_s'+str(split)
  try:
   p=WarpPlan(a,count=264,splits=split,source=src);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
   if not all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in es.items()):rec[key]=dict(status='incorrect',errors=es);continue
   gs[key]=capture(p);rec[key]=dict(status='pass',errors=es,plan=p)
  except RuntimeError as e:rec[key]=dict(status='rejected',reason=str(e))
 ts=paired(gs)
 for k,v in rec.items():v.pop('plan',None);v['time']=ts.get(k)
 (R/'kindprefetch-split-sweep-L768.json').write_text(json.dumps(rec,indent=2));print('RESULT',{k:(v.get('time') or {}).get('median_us',v['status']) for k,v in rec.items()},flush=True)
