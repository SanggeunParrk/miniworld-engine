import json,torch,triton.testing
from pathlib import Path
from validate import inputs
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
rows=[]
for L in [128,384,768]:
 inp=list(inputs(L*L,256,128,128,L,0));inp[3]=inp[3].contiguous();inp=tuple(inp)
 ref=output_f567_train(*inp)
 base=triton.testing.do_bench_cudagraph(lambda:output_f567_train(*inp),rep=150)
 candidates=[]
 for bm,bn,bk,gm,nw,ns in [(128,32,64,1,8,3),(64,64,64,4,4,2),(64,64,64,1,8,2),(64,64,64,4,8,2)]:
  c=dict(BLOCK_M1=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=gm,num_warps=nw,num_stages=ns)
  got=output_f567_impl(*inp,c);torch.cuda.synchronize()
  errs=[((v.float()-r.float()).norm()/r.float().norm()).item() for v,r in zip(got,ref)]
  assert max(errs)<1e-4,(L,c,errs)
  ms=triton.testing.do_bench_cudagraph(lambda:output_f567_impl(*inp,c),rep=150)
  candidates.append(dict(config=c,ms=ms,rel_l2=errs))
 best=min(candidates,key=lambda r:r['ms'])
 row=dict(L=L,triton_ms=base,cute_ms=best['ms'],speedup=base/best['ms'],best=best,candidates=candidates)
 rows.append(row);print(json.dumps(row),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2))
