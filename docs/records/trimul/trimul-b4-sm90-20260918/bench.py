import argparse,itertools,json,statistics,sys
from pathlib import Path
import torch,triton
from benchmarks.runners import bench
from miniworld_engine.kernels.layernorm.cute.tma_backward import prepare,config_rejection
from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent,_persistent_grid
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_bwd_kernel
from miniworld_engine.autotune.shape_key import both_key
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);a=p.parse_args();root=Path(__file__).parent
m=a.length**2;n=256;g=_persistent_grid(torch.device('cuda'))
x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t();dy=torch.randn_like(x);w=torch.randn(n,device='cuda');mean=x.float().mean(1);rs=(x.float().var(1,unbiased=False)+1e-5).rsqrt();out=torch.empty_like(x);pdw=torch.empty(g,n,device='cuda');pdb=torch.empty_like(pdw);dw=torch.empty(n,device='cuda');db=torch.empty_like(dw)
_ln_bwd_persistent.configs=[triton.Config({'BLOCK_M1':64,'BLOCK_K':256},num_warps=8,num_stages=1)];_ln_bwd_persistent.early_config_prune=None

def tri():
 if a.length==384:
  dw.zero_();db.zero_();_ln_bwd_kernel[lambda c:(triton.cdiv(m,c['BLOCK_M1']),)](dy,x,w,mean,rs,out,dw,db,m,n,*dy.stride(),*x.stride(),*out.stride(),N_PAD=256,shape_key=both_key(m,N=n))
 else:
  _ln_bwd_persistent[lambda c:(g,triton.cdiv(n,c['BLOCK_K']))](out,pdw,pdb,dy,x,w,mean,rs,n,*x.stride(),m,n,shape_key=both_key(m,N=n));torch.sum(pdw,0,out=dw);torch.sum(pdb,0,out=db)
tri();torch.cuda.synchronize();ref=[v.clone() for v in (out,dw,db)]
tg=torch.cuda.CUDAGraph()
with torch.cuda.graph(tg):tri()
rows=[]
configs=[dict(BLOCK_M1=bm,BLOCK_K=bk,num_warps=nw,num_stages=ns) for bm,bk,nw,ns in itertools.product((16,32,64),(256,),(4,8,16),(1,2,3))]
for c in configs:
 reason=config_rejection(c,n=n,itemsize=2,m_major=True,smem_limit=torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
 if reason:continue
 fn=prepare(x,dy,w,mean,rs,out,pdw,pdb,c)
 def full():fn();torch.sum(pdw,0,out=dw);torch.sum(pdb,0,out=db)
 full();torch.cuda.synchronize();errs=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip((out,dw,db),ref)];assert max(errs)<.004,errs
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):full()
 out.fill_(float('nan'));cg.replay();torch.cuda.synchronize();assert torch.isfinite(out).all()
 samples={'triton':[],'cute':[]}
 for rnd in range(3):
  for tag,gr in ([('triton',tg),('cute',cg)] if rnd%2==0 else [('cute',cg),('triton',tg)]):samples[tag].append(float(bench.bench_time(gr.replay,warmup=5,rep=30)['median_ms']))
 med={k:statistics.median(v) for k,v in samples.items()};row=dict(config=c,ms=med,speedup=med['triton']/med['cute'],errors=errs);rows.append(row)
 print(json.dumps(row),flush=True);(root/f'bench-L{a.length}.json').write_text(json.dumps(rows,indent=2));del cg
