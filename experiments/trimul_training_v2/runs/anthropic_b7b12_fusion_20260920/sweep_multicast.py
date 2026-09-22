from multicast_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--ring',action='store_true');args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);cls=MulticastRingPlan if args.ring else MulticastWarpPlan;plain=RingPlan if args.ring else WarpPlan;sp=20 if args.ring else 13
 base='front_ring96_cache3' if args.ring else 'front_prefetch_lnpair_storepipe'
 plans={};records={}
 for mode in ['plain','weights','xn','both']:
  src=base if mode=='plain' else base+'_mcast_'+mode
  try:
   p=(plain if mode=='plain' else cls)(a,count=264,splits=sp,source=src);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
   ok=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items())
   records[mode]=dict(source=src,errors=es,status='pass' if ok else 'incorrect')
   if ok:plans[mode]=p
  except RuntimeError as e:records[mode]=dict(status='rejected',reason=str(e));print('REJECTED',mode,str(e),flush=True)
 ts=paired({k:capture(p) for k,p in plans.items()})
 (R/f'multicast-sweep-L{args.length}.json').write_text(json.dumps(dict(records=records,times=ts),indent=2));print('MULTICAST',{k:v['median_us'] for k,v in ts.items()},flush=True)
