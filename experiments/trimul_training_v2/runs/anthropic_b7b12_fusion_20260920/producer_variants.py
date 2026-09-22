from front_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=768);ap.add_argument('--source',default='front_producer_bench');args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);_,dc,_=baseline(a,True);plans={};es={}
 for count in [32,64,128]:
  p=Plan(a,count=count,splits=1,debug=1,source=args.source);p();torch.cuda.synchronize();e=rel(p.debugdc,dc);assert e<1e-5,e;es[str(count)]=e;plans[str(count)]=p
 ts=paired({k:capture(p) for k,p in plans.items()});(R/f'{args.source}-L{args.length}.json').write_text(json.dumps(dict(errors=es,times=ts),indent=2));print('PRODUCER',{k:v['median_us'] for k,v in ts.items()},flush=True)
