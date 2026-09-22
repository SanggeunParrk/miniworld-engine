from front_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=768);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);_,dc,_=baseline(a,True);plans={};es={}
 for count in [24,32,40,48,64,96,128]:
  p=Plan(a,count=count,splits=1,debug=1,source='front_producer_bench');p();torch.cuda.synchronize();e=rel(p.debugdc,dc);assert e==0,e;es[str(count)]=e;plans[str(count)]=p
 ts=paired({k:capture(p) for k,p in plans.items()});(R/f'producer-only-L{args.length}.json').write_text(json.dumps(dict(errors=es,times=ts),indent=2));print('PRODUCER',{k:v['median_us'] for k,v in ts.items()},flush=True)
