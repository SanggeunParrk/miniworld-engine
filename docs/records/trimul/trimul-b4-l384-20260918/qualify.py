import argparse,itertools,json,statistics
from pathlib import Path
import torch,triton
from benchmarks.runners import bench
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_bwd_kernel
from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent,_persistent_grid
from miniworld_engine.kernels.layernorm.cute.tma_backward import prepare as persistent
from miniworld_engine.autotune.shape_key import both_key
from atomic_tma import prepare as atomic
p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=384);p.add_argument('--profile',action='store_true');p.add_argument('--atomic-sweep',action='store_true');a=p.parse_args();root=Path(__file__).parent
m=a.length**2;n=256;g=_persistent_grid(torch.device('cuda'))
torch.manual_seed(153)
x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t();dy=torch.randn_like(x);w=torch.randn(n,device='cuda');mu=x.float().mean(1);rs=(x.float().var(1,unbiased=False)+1e-5).rsqrt();out=torch.empty_like(x);dw=torch.empty(n,device='cuda');db=torch.empty_like(dw);pw=torch.empty(g,n,device='cuda');pb=torch.empty_like(pw)

def ak():return _ln_bwd_kernel[lambda c:(triton.cdiv(m,c['BLOCK_M1']),)](dy,x,w,mu,rs,out,dw,db,m,n,*dy.stride(),*x.stride(),*out.stride(),N_PAD=n,shape_key=both_key(m,N=n))
def af():dw.zero_();db.zero_();ak()
_ln_bwd_persistent.configs=[triton.Config({'BLOCK_M1':64,'BLOCK_K':256},num_warps=8,num_stages=1)];_ln_bwd_persistent.early_config_prune=None

def pk():return _ln_bwd_persistent[lambda c:(g,triton.cdiv(n,c['BLOCK_K']))](out,pw,pb,dy,x,w,mu,rs,n,*x.stride(),m,n,shape_key=both_key(m,N=n))
def reduce():torch.sum(pw,0,out=dw);torch.sum(pb,0,out=db)
def pf():pk();reduce()
ck=persistent(x,dy,w,mu,rs,out,pw,pb,dict(BLOCK_M1=32,BLOCK_K=256,num_warps=8,num_stages=2))
def cf():ck();reduce()
af();torch.cuda.synchronize();ref=[v.clone() for v in (out,dw,db)]

from persistent_mapped import prepare as mapped
from reduce_tiled import prepare as reduction
mk=mapped(x,dy,w,mu,rs,out,pw,pb,dict(BLOCK_M1=32,num_warps=4,num_stages=2,row_lanes=4))
rw=reduction(pw,dw);rb=reduction(pb,db)
def mf():mk();rw();rb()
funcs={'atomic':af,'triton_persistent':pf,'tma_persistent':cf,'tma_mapped_reduce':mf}
report={'L':a.length,'atomic_config':str(_ln_bwd_kernel.best_config),'captures':[]}
for capture in range(2):
 graphs={};errors={}
 for tag,f in funcs.items():
  f();torch.cuda.synchronize();errors[tag]=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip((out,dw,db),ref)];assert max(errors[tag])<.004,errors
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):f()
  out.fill_(float('nan'));graph.replay();torch.cuda.synchronize();assert torch.isfinite(out).all()
  graphs[tag]=graph
 samples={tag:[] for tag in graphs}
 for rnd in range(7):
  tags=list(graphs);tags=tags[rnd%len(tags):]+tags[:rnd%len(tags)]
  if rnd%2:tags.reverse()
  for tag in tags:samples[tag].append(float(bench.bench_time(graphs[tag].replay,warmup=10,rep=50)['median_ms']))
 row={'ms':{k:statistics.median(v) for k,v in samples.items()},'samples':samples,'errors':errors}
 report['captures'].append(row);print(json.dumps(row),flush=True)
 (root/f'qualification-L{a.length}.json').write_text(json.dumps(report,indent=2))
