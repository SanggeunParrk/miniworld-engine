import argparse,json,torch
from pathlib import Path
from miniworld_engine import settings
from miniworld_engine.kernels.transition.cuda.variants import extension,transition
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--d',type=int,default=128);a=p.parse_args()
settings.configure(engine_backend='triton',autotune_miss_cap=3)
torch.manual_seed(918);d=a.d
c=dict(bk=(1<<(d-1).bit_length()) if a.variant=='full_k' else 64,bn=32,bo=64,mgroups=1,ngroups=2 if d>=384 else 1,stages=2,min_blocks=1)
print('BUILD',a.variant,d,c,flush=True)
ext=extension(a.variant,d,c);print('BUILT',ext.resources(),ext.__file__,flush=True)
m=129;xn=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);r=torch.randn_like(xn)
wa=torch.randn(4*d,d,device='cuda',dtype=xn.dtype)*d**-.5;wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=xn.dtype)*(4*d)**-.5
aa=xn.float()@wa.float().T;bb=xn.float()@wb.float().T;hh=(torch.nn.functional.silu(aa)*bb).bfloat16()
ref=(hh.float()@ws.float().T).bfloat16()+r
y=ext.forward(xn,r,wa,wb,ws);torch.cuda.synchronize()
err=(y.float()-ref.float()).norm()/ref.float().norm();print('FWD',err.item(),flush=True);assert err<.02
dh=torch.randn_like(hh);h,dab=ext.gate_backward(xn,wa,wb,dh);torch.cuda.synchronize()
sig=aa.sigmoid();da=(dh.float()*bb*(sig+aa*sig*(1-sig))).bfloat16();db=(dh.float()*aa*sig).bfloat16()
for name,out,expected in [('h',h,hh),('da',dab[:,:4*d],da),('db',dab[:,4*d:],db)]:
 err=(out.float()-expected.float()).norm()/expected.float().norm();print(name,err.item(),flush=True);assert err<.02
x=r.view(1,m,d).requires_grad_();g=torch.rand(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True)
leaves=(x,g,b,wa.requires_grad_(),wb.requires_grad_(),ws.requires_grad_());dy=torch.randn_like(x)
from miniworld_engine.kernels.transition.triton.residual import transition_residual
ref=transition_residual(*leaves,1e-5);rg=torch.autograd.grad(ref,leaves,dy)
y=transition(*leaves,variant=a.variant,forward_config=c,backward_config=c);gg=torch.autograd.grad(y,leaves,dy)
for name,actual,expected in zip(['y','dx','dg','db','dwa','dwb','dws'],(y,*gg),(ref,*rg)):
 err=(actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12);print('GRAD',name,err.item(),flush=True);assert torch.isfinite(actual).all() and err<.02
print('PASS',flush=True)
