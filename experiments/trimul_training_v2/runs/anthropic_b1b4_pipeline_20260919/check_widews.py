from core import *
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'));from bench_k3 import graph,paired
with torch.no_grad():
 for n in (64,72,384,768):
  d,dy,s=data(n);ref=baseline(d,dy,s)
  cfgs=[(64,64,1,0,1,w,0,0,2,f,ss) for w,f,ss in ((2,0,4),)];plans={}
  for cfg in cfgs:
   p=Plan(d,dy,s,*cfg);o=p();torch.cuda.synchronize();er=[rel(x,y) for x,y in zip(o,ref)];print('CHECK',n,cfg,er,flush=True);assert max(er)<.01,er;plans[str(cfg)]=p
  if n>=384:
   tt=paired({k:graph(p) for k,p in plans.items()},reps=80,rounds=10);print('RESULT',n,{k:round(v['median_us'],2) for k,v in tt.items()},flush=True);(R/f'widews-{n}.json').write_text(json.dumps(tt,indent=2))
