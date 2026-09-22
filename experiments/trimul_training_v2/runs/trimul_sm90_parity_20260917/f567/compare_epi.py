import json, torch, triton.testing
from validate import inputs
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl as base
from epi3 import output_f567_impl as epi3
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
rows=[]
for L in [128,384,768]:
 inp=list(inputs(L*L,256,128,128,L,0));inp[3]=inp[3].contiguous();inp=tuple(inp)
 for bm,bn,bk,nw in [(64,32,32,4),(128,32,64,8),(128,64,64,8),(64,64,64,4),(64,128,64,4),(128,128,64,8),(128,32,64,4),(128,64,64,4),(64,64,64,8),(64,128,64,8),(128,128,64,4)]:
  c=dict(BLOCK_M1=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=1,num_warps=nw,num_stages=2)
  out1=base(*inp,c);out2=epi3(*inp,c);torch.cuda.synchronize()
  assert all(torch.equal(a,b) for a,b in zip(out1,out2))
  b=triton.testing.do_bench_cudagraph(lambda:base(*inp,c),rep=60)
  e=triton.testing.do_bench_cudagraph(lambda:epi3(*inp,c),rep=60)
  row=dict(L=L,config=c,base_ms=b,epi3_ms=e);rows.append(row);print(json.dumps(row),flush=True)
open(__file__.replace('.py','.json'),'w').write(json.dumps(rows,indent=2))
