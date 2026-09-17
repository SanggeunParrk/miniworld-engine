"""Qualification of the experimental covering-tile TMA LN backward.

This is not a production dispatch path. L384 uses atomic Triton in production,
so only L768 compares identical persistent-reduction algorithms.
"""
import argparse,json,statistics
from pathlib import Path
import torch,triton
from norm_tma import prepare
from benchmarks.runners import bench
from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent,_persistent_grid
p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');p.add_argument('--small',action='store_true');a=p.parse_args()
root=Path(__file__).parent;m=1032 if a.small else 768**2;n=256;g=_persistent_grid(torch.device('cuda'))
torch.manual_seed(173);x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t();dy=torch.randn_like(x);w=torch.randn(n,device='cuda');b=torch.randn_like(w)
mean=x.float().mean(1);rs=torch.rsqrt(x.float().var(1,unbiased=False)+1e-5);out=torch.empty_like(x);pw=torch.empty(g,n,device='cuda');pb=torch.empty_like(pw);dw=torch.empty_like(w);db=torch.empty_like(w)
c=dict(BLOCK_M1=32,BLOCK_K=256,num_warps=8,num_stages=2)
fn=prepare(x,dy,w,b,mean,rs,out,pw,pb,c,True)
def tri():
 return _ln_bwd_persistent.fn[(g,1)](out,pw,pb,dy,x,w,mean,rs,n,*x.stride(),m,n,shape_key=0,BLOCK_M1=64,BLOCK_K=256,num_warps=8,num_stages=1)
def full(f):
 f();torch.sum(pw,dim=0,out=dw);torch.sum(pb,dim=0,out=db)
full(tri);torch.cuda.synchronize();ref=[z.clone() for z in (out,dw,db)]
full(fn);torch.cuda.synchronize();errors=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip((out,dw,db),ref)];assert max(errors)<.004,errors
if a.small:
 xr=x.float().detach().requires_grad_();wr=w.detach().requires_grad_();br=b.detach().requires_grad_();y=torch.nn.functional.layer_norm(xr,(n,),wr,br,1e-5)
 want=torch.autograd.grad(y,(xr,wr,br),dy.float());errors_autograd=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip((out,dw,db),want)];assert max(errors_autograd)<.004,errors_autograd
 print('AUTOGRAD',errors_autograd,flush=True)
if a.profile:
 torch.cuda.synchronize();torch.cuda.profiler.start();tri();fn();torch.cuda.synchronize();torch.cuda.profiler.stop();print('PROFILE ORDER: updated Triton, CuTe TMA',flush=True)
else:
 report={'M':m,'N':n,'grid':g,'candidate':c,'errors':errors,'scope':'kernel plus both parameter reductions','independent_captures':[]}
 for build in range(1 if a.small else 3):
  graphs={}
  for tag,f in [('triton',tri),('cute',fn)]:
   full(f);cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):full(f)
   for z in (out,dw,db):z.fill_(float('nan'))
   cg.replay();torch.cuda.synchronize();assert all(torch.isfinite(z).all() for z in (out,dw,db))
   assert all(((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item()<.004 for v,r in zip((out,dw,db),ref));graphs[tag]=cg
  samples={tag:[] for tag in graphs}
  for rnd in range(1 if a.small else 9):
   for tag in list(graphs)[::1 if rnd%2 else -1]:samples[tag].append(float(bench.bench_time(graphs[tag].replay,warmup=1 if a.small else 10,rep=1 if a.small else 50)['median_ms']))
  med={k:statistics.median(v) for k,v in samples.items()};report['independent_captures'].append({'ms':med,'speedup':med['triton']/med['cute'],'samples_ms':samples});print('RESULT',json.dumps(report['independent_captures'][-1]),flush=True)
 (root/('tma-focus-small.json' if a.small else 'tma-focus-L768.json')).write_text(json.dumps(report,indent=2)+'\n')
