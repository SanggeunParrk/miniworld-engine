import argparse,itertools,json,statistics,time
from pathlib import Path
import torch,triton
from benchmarks.runners import bench
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_mat_kernel,_ln_bwd_kernel
from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent,_persistent_grid
from miniworld_engine.autotune.shape_key import both_key
from norm_tma import prepare
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args()
root=Path(__file__).parent
# Compare against the already qualified covering-tile B4 winner, not a fallback heuristic.
_ln_bwd_persistent.configs=[triton.Config({'BLOCK_M1':64,'BLOCK_K':256},num_warps=8,num_stages=1)]
_ln_bwd_persistent.early_config_prune=None
m=a.length*a.length;n=256;g=_persistent_grid(torch.device('cuda'))
torch.manual_seed(312)
x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t();dy=torch.randn_like(x);w=torch.randn(n,device='cuda');b=torch.randn_like(w)
mean=x.float().mean(1);rs=torch.rsqrt(x.float().var(1,unbiased=False)+1e-5)
pdw=torch.empty(g,n,device='cuda');pdb=torch.empty_like(pdw)
report={'L':a.length,'M':m,'N':n,'grid':g,'scope':'F4 and B4 including parameter reductions; same layouts, BF16 input/FP32 parameters','limitation':'Prototype only BK=N=256; other Triton grid feature tiles not implemented. No production dispatch.', 'results':{}}
for bwd in (True,False):
 label='B4' if bwd else 'F4';out=torch.empty_like(x) if bwd else torch.empty(m,n,device='cuda',dtype=x.dtype)
 dw=torch.empty(n,device='cuda');db=torch.empty_like(dw)
 if bwd:
  def tri():
   if a.length==384:
    dw.zero_();db.zero_();return _ln_bwd_kernel[lambda c:(triton.cdiv(m,c['BLOCK_M1']),)](dy,x,w,mean,rs,out,dw,db,m,n,*dy.stride(),*x.stride(),*out.stride(),N_PAD=256,shape_key=both_key(m,N=n))
   _ln_bwd_persistent[lambda c:(g,triton.cdiv(n,c['BLOCK_K']))](out,pdw,pdb,dy,x,w,mean,rs,n,*x.stride(),m,n,shape_key=both_key(m,N=n));torch.sum(pdw,dim=0,out=dw);torch.sum(pdb,dim=0,out=db)
 else:
  def tri():return _ln_mat_kernel[lambda c:(triton.cdiv(m,c['BLOCK_M1']),)](x,out,mean,rs,w,b,m,n,1e-5,*x.stride(),*out.stride(),shape_key=both_key(m,N=n))
 tri();torch.cuda.synchronize();ref=[z.clone() for z in ((out,dw,db) if bwd else (out,mean,rs))]
 tg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(tg):tri()
 base=float(bench.bench_time(tg.replay,warmup=10,rep=50)['median_ms']);rows=[]
 configs=list(itertools.product((16,),(2,4,8,16),(1,2,3,4,5,6)))+list(itertools.product((32,),(2,4,8,16),(4,5))) if not a.quick else [(64,4,1),(64,8,2),(128,8,2)]
 for bm,nw,ns in configs:
  c=dict(BLOCK_M1=bm,BLOCK_K=n,num_warps=nw,num_stages=ns);record={'config':c}
  try:
   fn=prepare(x,dy,w,b,mean,rs,out,pdw,pdb,c,bwd)
   def full():
    fn()
    if bwd:torch.sum(pdw,dim=0,out=dw);torch.sum(pdb,dim=0,out=db)
   full();torch.cuda.synchronize();act=(out,dw,db) if bwd else (out,mean,rs)
   errors=[((z.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for z,r in zip(act,ref)]
   assert max(errors)<.004,errors
   cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):full()
   # Prove the captured graph contains the actual LN kernel: poison its outputs, replay, recheck.
   out.fill_(float('nan'));cg.replay();torch.cuda.synchronize()
   assert torch.isfinite(out).all(), 'normalization launch escaped CUDA graph capture'
   timings={'triton':[],'cute':[]}
   for rnd in range(3):
    for tag,graph in ([('triton',tg),('cute',cg)] if rnd%2==0 else [('cute',cg),('triton',tg)]):timings[tag].append(float(bench.bench_time(graph.replay,warmup=5,rep=30)['median_ms']))
   med={k:statistics.median(v) for k,v in timings.items()};record.update(passed=True,errors=errors,ms=med,speedup=med['triton']/med['cute']);del cg
  except Exception as e:
   record.update(passed=False,error=str(e)[-2000:]);print('ERROR',label,c,str(e),flush=True)
  rows.append(record);report['results'][label]={'baseline_ms':base,'rows':rows};(root/f'tma-round2-L{a.length}.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',label,json.dumps(record),flush=True)
 del tg,ref
