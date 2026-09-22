from cluster_plan import *
with torch.no_grad():
 a=setup(768);p=ClusterPlan(a,120,'front_cluster512_phase_trace');p();torch.cuda.synchronize();v=p.counts[2:].reshape(120,8).to(torch.int64).cpu();delta=(v[:,1:4]-v[:,:3])&0xffffffff;res={}
 for role in ['dw','dx']:
  vals=delta[(torch.arange(120)%8<4) if role=='dw' else (torch.arange(120)%8>=4)]
  res[role]=dict(median_cycles=vals.median(0).values.tolist(),samples=vals.tolist())
 (R/'cluster512-phase-trace.json').write_text(json.dumps(res,indent=2));print('PHASES',{k:v['median_cycles'] for k,v in res.items()},flush=True)
