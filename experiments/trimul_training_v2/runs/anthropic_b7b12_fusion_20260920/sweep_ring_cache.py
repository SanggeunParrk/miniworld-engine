from ring_plan import *
with torch.no_grad():
 a=setup(768);ref=baseline(a);plans={};records={}
 for src in ['front_ring96_pipe_earlyfree','front_ring96_cache1','front_ring96_cache2','front_ring96_cache3','front_ring96_cache4']:
  try:
   p=RingPlan(a,count=264,splits=20,source=src);p();torch.cuda.synchronize();es=errors(p.outputs,ref);ok=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items());records[src]=dict(errors=es,status='pass' if ok else 'incorrect')
   if ok:plans[src]=p
  except RuntimeError as e:records[src]=dict(status='rejected',reason=str(e))
 ts=paired({k:capture(p) for k,p in plans.items()});(R/'ring-cache-sweep.json').write_text(json.dumps(dict(records=records,times=ts),indent=2));print('CACHE',{k:v['median_us'] for k,v in ts.items()},flush=True)
