import torch
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl
M,KP,KG,N,L=129,256,128,128,17
torch.manual_seed(1)
a=torch.randn(M,KP,device='cuda',dtype=torch.bfloat16)*.1
x=torch.randn(M,KG,device='cuda',dtype=torch.bfloat16)*.1
wp=torch.randn(N,KP,device='cuda',dtype=torch.bfloat16)*.1
wg=torch.randn(N,KG,device='cuda',dtype=torch.bfloat16).t()*.1
r=torch.randn(M,N,device='cuda',dtype=torch.bfloat16)
d=torch.ones(L,N,device='cuda',dtype=torch.bfloat16)
c=dict(BLOCK_M1=64,BLOCK_N=64,BLOCK_K=32,GROUP_M=2,num_warps=4,num_stages=2)
y,p,g=output_f567_impl(a,x,wp,wg,r,d,L,c)
torch.cuda.synchronize()
pr=a@wp.t(); gr=torch.sigmoid((x@wg).float()); yr=(pr.float()*gr+r.float()).bfloat16()
for name,v,ref in [('y',y,yr),('p',p,pr),('g',g,gr.bfloat16())]:
 print(name, ((v.float()-ref.float()).norm()/ref.float().norm()).item(),flush=True)
print('PASS',flush=True)
