import torch,json
from triton.testing import do_bench_cudagraph
from miniworld_engine import settings
settings.configure(engine_backend='triton',compile_wrap='custom_op')
from miniworld_engine.kernels.layernorm.triton.main import layer_norm_fwd_fused
results=[]
for m,d in [(128**2,128),(384**2,128),(768**2,128),(384,384),(768,768)]:
 dab=torch.randn(m,8*d,device='cuda',dtype=torch.bfloat16);wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16);wb=torch.randn_like(wa)
 packed=lambda:dab@torch.cat((wa,wb),0)
 split=lambda:dab[:,:4*d]@wa+dab[:,4*d:]@wb
 vals={'m':m,'d':d,'cat_ms':do_bench_cudagraph(lambda:torch.cat((wa,wb),0)), 'packed_ms':do_bench_cudagraph(packed),'split_ms':do_bench_cudagraph(split)}
 results.append(vals);print(vals,flush=True)
# Exercise both legal feature-tiling branches on offset inputs.
for dtype in (torch.bfloat16,torch.float32):
 for offset in (0,1000):
  torch.manual_seed(23);x=(torch.randn(37,128,device='cuda')+offset).to(dtype);w=torch.ones(128,device='cuda');b=torch.zeros_like(w)
  ref=torch.nn.functional.layer_norm(x.float(),(128,))
  for bk in (64,128):
   y=torch.empty_like(x);mean=torch.empty(37,device='cuda');rs=torch.empty_like(mean)
   layer_norm_fwd_fused.fn[(3,)](x,y,w,b,mean,rs,rs,128,1,37,128,1e-5,BLOCK_M1=16,BLOCK_K=bk,shape_key=37,HAS_ROWSCALE=False,num_warps=4)
   print('LN',str(dtype),offset,bk,'finite',bool(y.isfinite().all()),'relative',float((y.float()-ref).norm()/ref.norm()),flush=True)
open('/home/psk6950/MiniWorld/runs/transition_triton_audit_20260917/cat-bench.json','w').write(json.dumps(results,indent=2))
