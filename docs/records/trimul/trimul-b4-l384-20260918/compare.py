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
funcs={'triton_atomic_full':af,'triton_atomic_main':ak,'triton_persistent_full':pf,'triton_persistent_main':pk,'tma_persistent_full':cf,'tma_persistent_main':ck,'two_final_reductions':reduce}
report={'length':a.length,'M':m,'N':n,'grid':g,'atomic_config':str(_ln_bwd_kernel.best_config),'components':{},'atomic_candidates':[]}
for label,f in funcs.items():
 f();torch.cuda.synchronize()
 if a.profile:torch.cuda.cudart().cudaProfilerStart();f();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):f()
 ms=float(bench.bench_time(graph.replay,warmup=10,rep=50)['median_ms']);report['components'][label]=ms;print(label,ms,flush=True)
(root/f'components-L{a.length}.json').write_text(json.dumps(report,indent=2))
if a.atomic_sweep:
 tg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(tg):af()
 configs=[dict(BLOCK_M1=bm,num_warps=nw,num_stages=ns) for bm,nw,ns in itertools.product((8,16,32,64,128),(1,2,4,8,16),(1,2))]
 for c in configs:
  row={'config':c}
  try:
   fn=atomic(x,dy,w,mu,rs,out,dw,db,c)
   def full():dw.zero_();db.zero_();fn()
   full();torch.cuda.synchronize();errs=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip((out,dw,db),ref)];assert max(errs)<.004,errs
   cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):full()
   out.fill_(float('nan'));cg.replay();torch.cuda.synchronize();assert torch.isfinite(out).all()
   samples={'triton':[],'cute':[]}
   for rnd in range(3):
    for tag,gr in ([('triton',tg),('cute',cg)] if rnd%2==0 else [('cute',cg),('triton',tg)]):samples[tag].append(float(bench.bench_time(gr.replay,warmup=5,rep=30)['median_ms']))
   med={k:statistics.median(v) for k,v in samples.items()};row.update(ms=med,speedup=med['triton']/med['cute'],errors=errs)
  except ValueError as exc:row['excluded']=str(exc)
  report['atomic_candidates'].append(row);print(json.dumps(row),flush=True);(root/f'atomic-L{a.length}.json').write_text(json.dumps(report,indent=2))
