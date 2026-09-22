import torch
from miniworld_engine.kernels.trimul_inproj.cute.parity_dual_bwd import input_dual_bwd_sm90_impl
m,kg,kp,n=512,256,1024,128
g=torch.randn(m,kg,device='cuda',dtype=torch.bfloat16)*.1
f=(torch.randn(kp,m,device='cuda',dtype=torch.bfloat16)*.1).t()
w=(torch.randn(n,kg,device='cuda',dtype=torch.bfloat16)*.1).t()
v=torch.randn(kp,n,device='cuda',dtype=torch.bfloat16)*.1
y=input_dual_bwd_sm90_impl(g,f,w,v,128,dict(BLOCK_M1=64,BLOCK_N=128,BLOCK_K=64,GROUP_M=1,num_warps=4,num_stages=2))
r=(f.float()@v.float()+(g.float()@w.float()).bfloat16().float()).bfloat16()
torch.cuda.synchronize()
print('error',((y-r).float().norm()/r.float().norm()).item(),'max',(y-r).abs().max().item(),flush=True)
