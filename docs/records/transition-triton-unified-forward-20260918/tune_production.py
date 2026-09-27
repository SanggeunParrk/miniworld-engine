import argparse,json,socket,time
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.b2b_residual import _kernel,_prune
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
from miniworld_engine.autotune.shape_key import pack,both_key
from miniworld_engine.autotune.cache import store_ranked_configs,gpu_key,config_space_hash,shape_bucket,op_identity
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);args=p.parse_args()
settings.configure(engine_backend='triton',autotune_miss_cap=24)
root=Path(__file__).parent;d=args.width;op='transition_b2b_residual_triton'
configs=_prune(_kernel.configs,{'D':d});rows=[]
torch.manual_seed(123)
wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)*d**-.5
wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)*(4*d)**-.5
g=torch.rand(d,device='cuda');b=torch.randn_like(g)*.1
for length in (384,768):
 m=length**2;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);rs,c1=stats_triton(x,1e-5)
 y=torch.empty_like(x);xn=torch.empty_like(x);empty=x.new_empty(0)
 for save in (False,True):
  key=pack(both_key(m),D=d);ranked=[];attempted=[]
  for i,c in enumerate(configs):
   row={'D':d,'L':length,'save':save,'config':{**c.kwargs,'num_warps':c.num_warps,'num_stages':c.num_stages}}
   try:
    def fn():return _kernel.fn[(triton.cdiv(m,c.kwargs['BLOCK_M1']),)](x,g,b,rs,c1,wa,wb,ws,y,xn if save else empty,m,d,save,key,**c.kwargs,num_warps=c.num_warps,num_stages=c.num_stages)
    kernel=fn();torch.cuda.synchronize()
    ms=triton.testing.do_bench_cudagraph(fn,rep=30)
    ranked.append((c,ms));row.update(status='ok',ms=ms,registers=kernel.n_regs,spills=kernel.n_spills)
   except Exception as e:
    row.update(status='failed',error=repr(e)[-2000:])
    if 'illegal memory' in str(e):raise
   attempted.append(c);rows.append(row);print(json.dumps(row),flush=True)
   (root/f'production-D{d}.json').write_text(json.dumps({'node':socket.gethostname(),'rows':rows},indent=2))
  ranked.sort(key=lambda t:t[1]);assert ranked
  # Parallel processes write separate widths to staging files; merge after both finish.
  import miniworld_engine.autotune.cache as cache
  cache._CACHE_ROOT=root/f'cache-D{d}'
  store_ranked_configs(op,gpu_key(),'bfloat16+float32',shape_bucket(shape_key=key,SAVE_XN=int(save)),ranked,config_space_hash(_kernel.configs),op_id=op_identity(_kernel),configs=_kernel.configs,entry_configs=attempted,measurement={'method':'cuda_graph','rep_ms':30,'M':m,'affine_dtype':'float32'})
  print('WINNER',length,save,ranked[0],flush=True)
