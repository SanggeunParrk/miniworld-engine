import itertools,json,time
from pathlib import Path
import torch
from validate import inputs
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl,feasibility,_COMPILE_CACHE
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
import triton.testing

def bench(fn):
 for _ in range(3): fn()
 return triton.testing.do_bench_cudagraph(fn,rep=80)

results=[]
for L in [128,384,768]:
 inp=list(inputs(L*L,256,128,128,L,0)); inp[3]=inp[3].contiguous(); inp=tuple(inp)
 ref=output_f567_train(*inp); base=bench(lambda:output_f567_train(*inp))
 rows=[]
 # 24 representative schedules; all axes remain in the full registered domain.
 for bm,bn,bk,stages in itertools.product([64,128],[32,64,128],[32,64],[2]):
  for group in [1,4]:
   c=dict(BLOCK_M1=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=group,num_warps=bm//16,num_stages=stages)
   if feasibility(c): continue
   got=output_f567_impl(*inp,c); torch.cuda.synchronize()
   errors=[((v.float()-r.float()).norm()/r.float().norm()).item() for v,r in zip(got,ref)]
   assert max(errors)<1e-4,(L,c,errors)
   ms=bench(lambda:output_f567_impl(*inp,c))
   row=dict(config=c,ms=ms,rel_l2=errors); rows.append(row); print(json.dumps(dict(L=L,**row)),flush=True)
 best=min(rows,key=lambda r:r['ms'])
 result=dict(L=L,triton_ms=base,cute_ms=best['ms'],speedup=base/best['ms'],config=best['config'],trials=rows)
 results.append(result); print('RESULT',json.dumps(result),flush=True)
 Path(__file__).with_name('benchmark.json').write_text(json.dumps(results,indent=2))
fn=next(iter(_COMPILE_CACHE.values()))
ptx=fn.__ptx__
if callable(ptx):ptx=ptx()
Path(__file__).with_name('f567.ptx').write_text(str(ptx))
print('PTX',str(ptx)[:200],flush=True)
