import json,itertools,gc,torch,triton
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cute.parity_dual_bwd import input_dual_bwd_sm90_impl,DEFAULT_CONFIG,feasibility
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_dual_bwd,_input_dual_bwd_kernel
out=[]
configs=[dict(BLOCK_M1=64,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=group,num_warps=4,num_stages=st) for bn,bk,group,st in itertools.product((32,64),(64,128),(1,4),(2,3))]
configs += [dict(BLOCK_M1=bm,BLOCK_N=128,BLOCK_K=128,GROUP_M=1,num_warps=bm//16,num_stages=st) for bm,st in itertools.product((64,128),(2,3,4))]
for l in (384,):
 m,kg,kp,n=l*l,256,1024,128
 torch.manual_seed(72)
 g=torch.randn(m,kg,device='cuda',dtype=torch.bfloat16)
 f=torch.randn(kp,m,device='cuda',dtype=torch.bfloat16).t()
 w=torch.randn(n,kg,device='cuda',dtype=torch.bfloat16).t()
 v=torch.randn(kp,n,device='cuda',dtype=torch.bfloat16)
 base=lambda:input_dual_bwd(g,f,w,v,l)
 ref=base();torch.cuda.synchronize()
 trms=triton.testing.do_bench_cudagraph(base,rep=1000)
 row=dict(L=l,triton_ms=trms,triton_config=str(_input_dual_bwd_kernel.best_config),cute=[])
 print('triton',l,trms,flush=True)
 for c in configs:
  if feasibility(c):continue
  fn=lambda:input_dual_bwd_sm90_impl(g,f,w,v,l,c)
  y=fn();torch.cuda.synchronize()
  err=((y-ref).float().norm()/ref.float().norm()).item()
  ms=triton.testing.do_bench_cudagraph(fn,rep=300)
  r=dict(config=c,ms=ms,relative_l2=err)
  print(json.dumps(dict(L=l,**r)),flush=True);row['cute'].append(r)
 out.append(row)
 Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/dual_bwd/narrow-bench.json').write_text(json.dumps(out,indent=2))
 del g,f,w,v,y,ref;gc.collect();torch.cuda.empty_cache()
