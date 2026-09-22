import json,statistics
from pathlib import Path
import torch,triton
import small_ring as mod
import projection_prefetch_initial as base
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel
root=Path(__file__).parent;result={}
def cfg(bn=64,bk=64,gm=2):return dict(BLOCK_M1=64,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=gm,num_warps=4,num_stages=2)
for L in (384,768):
 m=L*L;n=128;kp=256;kg=128;kw=dict(device='cuda',dtype=torch.bfloat16);torch.manual_seed(123)
 a=torch.randn(m,kp,**kw)*.2;x=torch.randn(m,kg,**kw)*.2;wp=torch.randn(n,kp,**kw)*.2;wg=torch.randn(kg,n,**kw)*.2;r=torch.randn(m,n,**kw);ds=(torch.rand(L,n,device='cuda')>.2).bfloat16()*1.25
 args=(a,x,wp,wg,r,ds,L);y,p,g=(torch.empty_like(r) for _ in range(3))
 def tri(c):return _output_f567_kernel.fn[(triton.cdiv(m,c['BLOCK_M1'])*triton.cdiv(n,c['BLOCK_N']),)](a,x,wp,wg,p,g,y,r,ds,m,L,kp,kg,n,*wp.stride(),*wg.stride(),shape_key=0,**c)
 tc=dict(cfg(gm=4),BLOCK_M1=128,num_warps=8,num_stages=3);tri(tc);rows=[];result[str(L)]={'rows':rows}
 for c in (cfg(gm=2),cfg(gm=4)):
  try:
   got=mod.output_f567_impl(*args,c);torch.cuda.synchronize();err=[((u.float()-v.float()).norm()/v.float().norm()).item() for u,v in zip(got,(y,p,g))];assert max(err)<1e-4,err
   fns={'candidate':lambda:mod.output_f567_impl(*args,c),'checkpoint':lambda:base.output_f567_impl(*args,c),'triton':lambda:tri(tc)};samples={k:[] for k in fns}
   for rnd in range(3):
    for k in list(fns)[::1 if rnd%2==0 else -1]:samples[k].append(triton.testing.do_bench_cudagraph(fns[k],rep=45))
   row={'config':c,'samples_ms':samples,'medians_ms':{k:statistics.median(v) for k,v in samples.items()},'relative_l2':err}
  except Exception as e:row={'config':c,'error':repr(e)}
  rows.append(row);print(L,json.dumps(row),flush=True);(root/'small_ring.json').write_text(json.dumps(result,indent=2))
 del a,x,wp,wg,r,ds,args,y,p,g;torch.cuda.empty_cache()
