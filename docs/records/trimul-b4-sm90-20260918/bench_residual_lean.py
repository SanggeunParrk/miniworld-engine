import argparse,itertools,json,statistics
from pathlib import Path
import torch,triton
from benchmarks.runners import bench
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _ln_bwd_residual_kernel
from miniworld_engine.autotune.shape_key import both_key
from residual_lean import prepare
p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=768);a=p.parse_args();root=Path(__file__).parent
m=a.length**2;n=128
torch.manual_seed(231)
x=torch.randn(m,n,device='cuda',dtype=torch.bfloat16);dy=torch.randn_like(x);dr=torch.randn_like(x);w=torch.randn(n,device='cuda');mean=x.float().mean(1);rs=(x.float().var(1,unbiased=False)+1e-5).rsqrt();out=torch.empty_like(x);dw=torch.empty(n,device='cuda');db=torch.empty_like(dw)
_ln_bwd_residual_kernel.configs=[triton.Config({'BLOCK_M1':64,'BLOCK_K':128},num_warps=4,num_stages=1)];_ln_bwd_residual_kernel.early_config_prune=None

def tri():
 dw.zero_();db.zero_();_ln_bwd_residual_kernel[lambda c:(triton.cdiv(m,c['BLOCK_M1']),)](out,dy,dw,db,dr,x,w,mean,rs,rs,1,1,*x.stride(),m,n,shape_key=both_key(m,N=n),HAS_ROWSCALE=False)
tri();torch.cuda.synchronize();ref=[v.clone() for v in (out,dw,db)]
tg=torch.cuda.CUDAGraph()
with torch.cuda.graph(tg):tri()
rows=[]
for bm,nw,ns in itertools.product((16,32,64,128),(1,2,4,8),(1,)):
 c=dict(BLOCK_M1=bm,BLOCK_K=n,num_warps=nw,num_stages=ns)
 fn=prepare(x,dy,w,mean,rs,out,dw,db,dr,c)
 def full():dw.zero_();db.zero_();fn()
 full();torch.cuda.synchronize();errs=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip((out,dw,db),ref)];assert max(errs)<.004,errs
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):full()
 out.fill_(float('nan'));cg.replay();torch.cuda.synchronize();assert torch.isfinite(out).all()
 samples={'triton':[],'cute':[]}
 for rnd in range(3):
  for tag,gr in ([('triton',tg),('cute',cg)] if rnd%2==0 else [('cute',cg),('triton',tg)]):samples[tag].append(float(bench.bench_time(gr.replay,warmup=5,rep=30)['median_ms']))
 med={k:statistics.median(v) for k,v in samples.items()};row=dict(config=c,ms=med,speedup=med['triton']/med['cute'],errors=errs);rows.append(row)
 print(json.dumps(row),flush=True);(root/f'residual-lean-L{a.length}.json').write_text(json.dumps(rows,indent=2));del cg
