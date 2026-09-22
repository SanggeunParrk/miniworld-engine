import torch,json
from experiment import *
ln=ln_tma
for n in (64,72):
 torch.manual_seed(62);c=128;h=256
 x=torch.randn(1,n,n,c,device='cuda',dtype=torch.bfloat16);w=torch.randn(4*h,c,device='cuda',dtype=x.dtype)/c**.5
 mask=(torch.rand(n,n,device='cuda')>.2).float();g=torch.rand(c,device='cuda');b=torch.randn_like(g)
 xn,mu,rs=ln(x,g,b)
 ab,pre=S.front_training(xn.reshape_as(x),w,mask,list(S.front_default(h)))
 af,pf,xf,mf,rf=front(x,w,mask,g,b)
 for name,a,z in zip(('ab','pre','xn','mu','rs'),(ab,pre,xn,mu,rs),(af,pf,xf.reshape_as(xn),mf,rf)):
  err=(a.float()-z.float()).abs().max().item();print(n,name,err,flush=True);assert torch.equal(a,z),(name,err)
 tri=torch.randn(h,n,n,device='cuda',dtype=x.dtype);gg=torch.rand(h,device='cuda');bb=torch.randn_like(gg)
 norm,mo,ro=ln(tri,gg,bb,trans=True,serial=1)
 wp=torch.randn(c,h,device='cuda',dtype=x.dtype)/h**.5;wg=torch.randn(c,c,device='cuda',dtype=x.dtype)/c**.5;ds=(torch.rand(n,c,device='cuda')>.25).bfloat16()/.75
 yy,pp,gate=S.output_training(norm.reshape(n,n,h),xn.reshape(n,n,c),wp,wg,x,ds,[2,64,4,1,232])
 yf,nf,mof,rof,pf,gf=T.output_training(tri,xn.reshape(n,n,c),wp,wg,gg,bb,x.reshape(n*n,c),ds,1e-5,[2,64,4,1,232,1])
 for name,a,z in zip(('out','norm','muout','rsout','proj','gate'),(yy.reshape_as(yf),norm,mo,ro,pp.reshape_as(pf),gate.reshape_as(gf)),(yf,nf,mof,rof,pf,gf)):
  err=(a.float()-z.float()).abs().max().item();print(n,name,err,flush=True);assert torch.equal(a,z),(name,err)
print('PASS',flush=True)
