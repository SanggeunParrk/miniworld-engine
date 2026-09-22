"""Instrumented role/barrier/reduction durations, not promotion benchmarks."""
from ring_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=768)
ap.add_argument('--splits',type=int,default=20);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a)
 p=RingPlan(a,count=264,splits=args.splits,source='front_ring96_role_trace')
 plain=RingPlan(a,count=264,splits=args.splits,source='front_ring96_cache3')
 p();torch.cuda.synchronize();es=errors(p.outputs,ref)
 assert all(e['finite'] and e['relative_l2']<=LIMITS[k] for k,e in es.items()),es
 g=capture(p);traces=[]
 for _ in range(20):g.replay()
 for _ in range(10):
  g.replay();torch.cuda.synchronize()
  offset=2+9*p.config['ring_tiles'];dwcount=8*args.splits
  ticks=p.counts[offset:offset+8*264].view(torch.int64).reshape(264,4).cpu()
  ids=p.counts[offset+8*264:].cpu().tolist()
  start=int(ticks[:,0].min());dt=(ticks[:,1:]-ticks[:,:-1]).double()/1000
  item=dict(timestamps_us=((ticks-start).double()/1000).tolist(),sm_ids=ids)
  for name,sel in [('dw',slice(0,dwcount)),('dx',slice(dwcount,None))]:
   for phase,col in [('role',0),('grid_wait',1),('reduce',2)]:
    v=dt[sel,col];item[name+'_'+phase+'_median_us']=float(v.median())
    item[name+'_'+phase+'_max_us']=float(v.max())
  traces.append(item)
 keys=[k for k in traces[0] if k.endswith('_us') and k!='timestamps_us']
 summary={k:statistics.median(t[k] for t in traces) for k in keys}
 ts=paired({'selected':capture(plain),'trace':g})
 out=dict(L=args.length,splits=args.splits,errors=es,traces=traces,summary=summary,times=ts)
 (R/f'ring-role-stages-L{args.length}.json').write_text(json.dumps(out,indent=2))
 print('TRACE',summary,flush=True)
 print('OVERHEAD',{k:v['median_us'] for k,v in ts.items()},flush=True)
