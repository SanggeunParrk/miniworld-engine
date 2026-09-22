from dual import *
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'));from bench_k3 import graph,paired
with torch.no_grad():
 for n in (64,384,768):
  d,dy,s=data(n);ref=baseline(d,dy,s);plans={}
  for part in (1,2):
   p=Unified(d,dy,s,min(132,n*n//64),part);o=p();torch.cuda.synchronize();er=[rel(x,y) for x,y in zip(o,ref)];print('CHECK',n,part,er,flush=True);assert max(er)<.01,er;plans[str(part)]=p
  tt=paired({k:graph(p) for k,p in plans.items()},reps=80,rounds=10);print('RESULT',n,{k:round(v['median_us'],2) for k,v in tt.items()},flush=True)
