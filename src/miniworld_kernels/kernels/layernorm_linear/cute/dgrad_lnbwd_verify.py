import torch, torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.cute.dgrad_lnbwd import dgrad_lnbwd_cute
D=torch.device("cuda"); dt=torch.bfloat16
def cos(a,b): return F.cosine_similarity(a.float().flatten(),b.float().flatten(),0).item()
M,K,N=8192,128,128; eps=1e-5; torch.manual_seed(0)
x=torch.randn(M,K,device=D,dtype=dt); g=torch.randn(K,device=D,dtype=dt)
w=(torch.randn(N,K,device=D,dtype=dt)*K**-0.5); dY=torch.randn(M,N,device=D,dtype=dt)
mean=x.float().mean(-1); rstd=torch.rsqrt(x.float().var(-1,unbiased=False)+eps)
xhat=((x.float()-mean[:,None])*rstd[:,None]).to(dt)
# reference dx
dxn=(dY.float()@w.float())            # (M,K)
dxh=dxn*g.float()
xh=(x.float()-mean[:,None])*rstd[:,None]
c2=dxh.mean(-1,keepdim=True); c1=(dxh*xh).mean(-1,keepdim=True)
dx_ref=rstd[:,None]*(dxh-c2-xh*c1)
dx=dgrad_lnbwd_cute(dY,w,xhat,g,rstd)
print("dx shape", tuple(dx.shape), "cos=", cos(dx,dx_ref), "max|abs|=", (dx.float()-dx_ref).abs().max().item())
