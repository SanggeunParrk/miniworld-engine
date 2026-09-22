from paircta_plan import *
with torch.no_grad():
 a=setup(768);ref=baseline(a);plans={};records={}
 for unroll in [8,4,2]:
  for sp in [14,16,18,20,22,24]:
   key=f'u{unroll}s{sp}'
   try:
    p=PairCTAPlan(a,splits=sp,source=f'front_ring_paircta_u{unroll}');p();torch.cuda.synchronize();es=errors(p.outputs,ref);ok=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items());records[key]=dict(errors=es,status='pass' if ok else 'incorrect')
    if ok:plans[key]=p
    else:print('INCORRECT',key,es,flush=True)
   except RuntimeError as e:records[key]=dict(status='rejected',reason=str(e));print('REJECTED',key,str(e),flush=True)
 ts=paired({k:capture(p) for k,p in plans.items()});(R/'paircta-tune-L768.json').write_text(json.dumps(dict(records=records,times=ts),indent=2));print('PAIRCTA_TUNE',{k:v['median_us'] for k,v in ts.items()},flush=True)
