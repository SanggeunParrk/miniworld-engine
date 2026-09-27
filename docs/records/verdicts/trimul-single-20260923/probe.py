import json, torch
import torch.nn.functional as F
from pathlib import Path
from initial_plan import Plan
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

torch.manual_seed(912)
torch.set_num_threads(8)
T._launch_module()._make_context_current(0)
L=384; D=128
bf=lambda *s: torch.randn(*s,device='cuda',dtype=torch.bfloat16,requires_grad=True)
x=bf(1,L,L,D)
w=[(torch.randn(D,D,device='cuda',dtype=torch.bfloat16)/D**.5).requires_grad_() for _ in range(6)]
p=[(1+torch.randn(D,device='cuda')*.1).requires_grad_(),(torch.randn(D,device='cuda')*.05).requires_grad_(),(1+torch.randn(D,device='cuda')*.1).requires_grad_(),(torch.randn(D,device='cuda')*.05).requires_grad_()]
mask=(torch.rand(L,L,device='cuda')>.15).bfloat16()
ds=(torch.rand(L,D,device='cuda')>.25).bfloat16()*(4/3)
dy=torch.randn_like(x)

def ref(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,outgoing):
    xn=F.layer_norm(x.float(),(D,),gi,bi,1e-5).bfloat16()
    a=(F.linear(xn,wl)*F.linear(xn,wlg).sigmoid())*mask[None,:,:,None]
    b=(F.linear(xn,wr)*F.linear(xn,wrg).sigmoid())*mask[None,:,:,None]
    t=torch.einsum('bikd,bjkd->bijd' if outgoing else 'bkid,bkjd->bijd',a,b)
    z=F.layer_norm(t.float(),(D,),go,bo,1e-5).bfloat16()
    return x+F.linear(z,wp)*F.linear(xn,wg).sigmoid()*ds[None,None]
ref=torch.compile(ref,fullgraph=True,options={'triton.cudagraphs':False})
for direction in [True,False]:
    plan=Plan(x,*w,*p,mask,ds,dy,outgoing=direction)
    with torch.no_grad(): y,g=plan()
    torch.cuda.synchronize()
    z=ref(x,*w,*p,direction)
    h=torch.autograd.grad(z,(x,*w,*p),dy)
    err=[float((a.float()-b.float()).norm()/b.float().norm()) for a,b in zip([y,*g],[z,*h])]
    print('CHECK',direction,err,flush=True)
    assert err[0]<.005 and max(err[1:])<.01
print('PASS',flush=True)
