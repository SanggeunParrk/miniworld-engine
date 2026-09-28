import gc,json,torch
from pathlib import Path
from check_full import setup,rel
from bench_k3 import graph,paired
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=24)
results=[]
for N in (384,768):
 leaves,call,mask,ds=setup(N,'bidir');dy=torch.randn_like(leaves[0])
 refs={};gs={}
 for backend in ('triton','anthropic_cuda'):
  def train(backend=backend):
   y=call(backend)
   return y,torch.autograd.grad(y,leaves,dy)
  gs[backend+'_fwd']=graph(lambda backend=backend:call(backend))
  gs[backend+'_train']=graph(train)
 yr,rg=gs['triton_train'][1];y,gg=gs['anthropic_cuda_train'][1]
 err=[rel(a,b) for a,b in zip((y,*gg),(yr,*rg))]
 assert max(err)<.015,err
 times=paired(gs,reps=12,rounds=8)
 row=dict(N=N,kind='bidirectional',dropout=.25,C=128,H=256,relative_l2=err,times=times)
 results.append(row);print('RESULT',json.dumps(row),flush=True)
 Path('/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919/full-times.json').write_text(json.dumps(results,indent=2))
 del gs,refs,leaves,call,mask,ds,dy,yr,rg,y,gg
 gc.collect();torch.cuda.empty_cache()
