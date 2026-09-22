from core import *
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'));from bench_k3 import graph,paired
with torch.no_grad():
 for n in (64,72,384,768):
  d,dy,s=data(n);ref=baseline(d,dy,s);p=Plan(d,dy,s,64,64,1,0,1,2,0,0,2,1);out=p();torch.cuda.synchronize();er=[rel(x,y) for x,y in zip(out,ref)];print('CHECK',n,er,flush=True);assert max(er)<.01,er
  if n>=384:
   plans={str(f):Plan(d,dy,s,64,64,1,1,1,2,0,0,2,f) for f in (0,1)};gs={k:graph(x) for k,x in plans.items()};gs['full']=graph(p);tt=paired(gs,reps=80,rounds=10);print('RESULT',n,{k:round(v['median_us'],2) for k,v in tt.items()},flush=True);(R/f'dfast-{n}.json').write_text(json.dumps(tt,indent=2))
