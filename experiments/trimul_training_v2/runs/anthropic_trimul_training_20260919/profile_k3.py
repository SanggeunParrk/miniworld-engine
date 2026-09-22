import torch
from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_training import output_training,default_config
H=256;N=768;C=128;kw=dict(device='cuda',dtype=torch.bfloat16)
tri=torch.randn(H,N,N,**kw);xn=torch.randn(N,N,C,**kw)
wp=torch.randn(C,H,**kw)/H**.5;wg=torch.randn(C,C,**kw)/C**.5
gamma=torch.rand(H,device='cuda');beta=torch.randn(H,device='cuda')*.2
res=torch.randn(N*N,C,**kw);ds=(torch.rand(N,C,device='cuda')>.25).bfloat16()/.75
args=(tri,xn,wp,wg,gamma,beta,res,ds,1e-5,list(default_config(H)))
for _ in range(3):out=output_training(*args)
torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
out=output_training(*args)
torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
