import argparse,json,socket
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.segmented_residual import _kernel
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
from miniworld_engine.autotune.shape_key import both_key,pack
from miniworld_engine.autotune.cache import store_ranked_configs,gpu_key,config_space_hash,op_identity,shape_bucket
import miniworld_engine.autotune.cache as cache
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
settings.configure(engine_backend='triton',autotune_miss_cap=24)
root=Path(__file__).parent;d=a.width;op='transition_segmented_b2b_triton';rows=[]
cache._CACHE_ROOT=root/f'cache-D{d}'
raw=json.loads((root/f'tune-D{d}.json').read_text())['rows']
torch.manual_seed(942)
g=torch.rand(d,device='cuda');b=torch.randn_like(g)*.1
wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)*d**-.5;wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)*(4*d)**-.5
for l in (384,768):
 m=l*l;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);rs,c1=stats_triton(x,1e-5)
 xn=((x.float()*rs[:,None]-c1[:,None])*g+b).bfloat16();y=torch.empty_like(x);saved=torch.empty_like(x);empty=x.new_empty(0)
 for mode,norm,save in [('separate',False,False),('fused_inference',True,False),('fused_training',True,True)]:
  configs=[r['config'] for r in sorted((r for r in raw if r['status']=='ok' and r['mode']==mode),key=lambda r:r['ms'])[:5]]
  ranked=[];tested=[];key=pack(both_key(m),D=d)
  for c in configs:
   tc=triton.Config(dict(BLOCK_M1=c['BM'],BLOCK_N=c['BN'],BLOCK_K=c['BK'],BLOCK_O=c['BO']),num_warps=c['num_warps'],num_stages=c['num_stages'])
   def fn():return _kernel.fn[(triton.cdiv(m,c['BM']),)](x if norm else xn,x,g if norm else empty,b if norm else empty,rs if norm else empty,c1 if norm else empty,wa,wb,ws,y,saved if save else empty,m,d,norm,save,key,**tc.kwargs,num_warps=tc.num_warps,num_stages=tc.num_stages)
   k=fn();torch.cuda.synchronize();ms=triton.testing.do_bench_cudagraph(fn,rep=50)
   ranked.append((tc,ms));tested.append(tc);row=dict(D=d,L=l,mode=mode,config=c,ms=ms,registers=k.n_regs,spills=k.n_spills,shared=k.metadata.shared);rows.append(row);print(json.dumps(row),flush=True)
  ranked.sort(key=lambda x:x[1])
  store_ranked_configs(op,gpu_key(),'bfloat16+float32' if norm else 'bfloat16',shape_bucket(shape_key=key,NORMALIZE=int(norm),SAVE_XN=int(save)),ranked,config_space_hash(_kernel.configs),op_id=op_identity(_kernel),configs=_kernel.configs,entry_configs=tested,measurement={'method':'cuda_graph','rep_ms':50,'M':m,'shortlist':'five best from explicit full grid at L384'})
(root/f'cache-bench-D{d}.json').write_text(json.dumps(dict(node=socket.gethostname(),rows=rows),indent=2)+'\n')
