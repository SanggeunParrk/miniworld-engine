from core_saved import *
import gc,statistics,concurrent.futures
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'))
from bench_k3 import graph,paired
from miniworld_engine import settings
cs=list(S.front_candidates());print('CONFIGS',len(cs),'per variant',flush=True)
jobs=[('front',f,c) for f in (False,True) for c in cs]+[('ln',False,c) for c in ((0,0),(0,1),(1,0),(1,1))]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for j,_ in enumerate(pool.map(lambda a:build(*a),jobs)):
  if j%12==0:print('BUILD',j+1,len(jobs),flush=True)

def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
def tune(n,name,candidates,fn,ref):
 rows=[]
 for j,c in enumerate(candidates):
  gg,o=graph(lambda:fn(c));err=max(rel(a,b) for a,b in zip(o,ref));assert err<.001,(name,c,err)
  tm=paired({'one':(gg,o)},reps=16,rounds=3)['one']['median_us'];rows.append(dict(config=c,us=tm,error=err));del gg,o
  if j%12==0:print('TUNE',n,name,j+1,len(candidates),flush=True)
 (R/f'tune-{n}-{name}.json').write_text(json.dumps(rows,indent=2));top=sorted(rows,key=lambda r:r['us'])[:4]
 gs={str(r['config']):graph(lambda c=tuple(r['config']):fn(c)) for r in top};tm=paired(gs,reps=80,rounds=12);best=min(top,key=lambda r:tm[str(r['config'])]['median_us'])['config']
 (R/f'tune-{n}-{name}-final.json').write_text(json.dumps(tm,indent=2));print('BEST',n,name,best,flush=True);return tuple(best)
results=[]
with torch.no_grad():
 for n in (384,768):
  d=setup(n);settings.configure(autotune_miss_cap=3);cf=(2,64,6,1,0)
  ref=front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],True,cf)
  lc=tune(n,'ln',[(0,0),(0,1),(1,0),(1,1)],lambda c:ln(d['x'],d['gi'],d['bi'],c),ref[:3])
  configs={}
  for fused in (True,False):configs[fused]=tune(n,'fused' if fused else 'split',cs,lambda c:front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],fused,c,lc),ref)
  del ref;gc.collect();dy=torch.randn_like(d['x']);outs={f:forward(d,f,configs[f],lc) for f in (True,False)}
  err=rel(outs[True][0],outs[False][0]);assert err==0,err
  ga=backward(d,outs[True][1],dy);gb=backward(d,outs[False][1],dy);errs=[rel(a,b) for a,b in zip(ga,gb)];assert max(errs)<.001
  def train(f):
   y,s=forward(d,f,configs[f],lc);return y,backward(d,s,dy)
  gs={}
  for f in (True,False):
   k='fused' if f else 'split'
   gs[k+'-front']=graph(lambda f=f:front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],f,configs[f],lc))
   gs[k+'-fwd']=graph(lambda f=f:forward(d,f,configs[f],lc))
   gs[k+'-bwd']=graph(lambda f=f:backward(d,outs[f][1],dy))
   gs[k+'-train']=graph(lambda f=f:train(f))
  tm=paired(gs,reps=80,rounds=20)
  row=dict(N=n,dropout=.25,rng_timed=False,backward_saves=True,weights_prepacked=True,configs=dict(fused=configs[True],split=configs[False],ln=lc),candidate_count=len(cs),fwd_relative_l2=err,gradient_relative_l2=errs,times=tm,backward_autotune_miss_cap=3)
  results.append(row);(R/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',n,{k:round(v['median_us'],2) for k,v in tm.items()},flush=True)
  del gs,outs,ga,gb,d;gc.collect();torch.cuda.empty_cache()
print('DONE',flush=True)
