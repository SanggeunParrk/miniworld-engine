import argparse
import torch,triton
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _ln_bwd_residual_kernel
from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent,_persistent_grid
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.layernorm.cute.tma_backward import prepare as b4
from residual_tma import prepare as b11
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['b4','b11'],default='b11');p.add_argument('--profile',action='store_true');a=p.parse_args()
m=768**2;n=256 if a.kind=='b4' else 128
x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t() if a.kind=='b4' else torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
dy=torch.randn_like(x);w=torch.randn(n,device='cuda');mean=x.float().mean(1);rs=(x.float().var(1,unbiased=False)+1e-5).rsqrt();out=torch.empty_like(x);dw=torch.empty(n,device='cuda');db=torch.empty_like(dw)
if a.kind=='b4':
 g=_persistent_grid(x.device);pw=torch.empty(g,n,device='cuda');pb=torch.empty_like(pw)
 fn=b4(x,dy,w,mean,rs,out,pw,pb,dict(BLOCK_M1=32,BLOCK_K=256,num_warps=8,num_stages=2))
 def tri():_ln_bwd_persistent.fn[(g,1)](out,pw,pb,dy,x,w,mean,rs,n,*x.stride(),m,n,BLOCK_M1=64,BLOCK_K=256,num_warps=8,num_stages=1,shape_key=both_key(m,N=n))
else:
 dr=torch.randn_like(x)
 fn=b11(x,dy,w,mean,rs,out,dw,db,dr,dict(BLOCK_M1=32,BLOCK_K=128,num_warps=4,num_stages=1))
 def tri():_ln_bwd_residual_kernel.fn[(triton.cdiv(m,64),)](out,dy,dw,db,dr,x,w,mean,rs,rs,1,1,*x.stride(),m,n,BLOCK_M1=64,BLOCK_K=128,num_warps=4,num_stages=1,shape_key=both_key(m,N=n),HAS_ROWSCALE=False)
for f in (tri,fn):dw.zero_();db.zero_();f()
torch.cuda.synchronize()
if a.profile:torch.cuda.cudart().cudaProfilerStart()
for f in (tri,fn):dw.zero_();db.zero_();f();torch.cuda.synchronize()
if a.profile:torch.cuda.cudart().cudaProfilerStop()
print('completed',a.kind)
