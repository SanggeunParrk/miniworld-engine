import json,torch
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidirectional_trimul_triton
from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import trimul_triton

def rel(a,b):return ((a.float()-b.float()).square().sum()/b.float().square().sum().clamp_min(1e-30)).sqrt().item()
def setup(N,kind):
 torch.manual_seed(341);C=128;H=C*(2 if kind=='bidir' else 1);dt=torch.bfloat16;dev='cuda'
 x=torch.randn(1,N,N,C,device=dev,dtype=dt,requires_grad=True)
 weights=[(torch.randn(H,C,device=dev,dtype=dt)/C**.5).requires_grad_() for _ in range(4)]
 weights += [(torch.randn(C,C,device=dev,dtype=dt)/C**.5).requires_grad_(),(torch.randn(C,H,device=dev,dtype=dt)/H**.5).requires_grad_()]
 norms=[torch.rand(C,device=dev,requires_grad=True),torch.randn(C,device=dev,requires_grad=True),torch.rand(H,device=dev,requires_grad=True),torch.randn(H,device=dev,requires_grad=True)]
 with torch.no_grad():norms[0][0]=0;norms[2][0]=0
 leaves=[x,*weights,*norms]
 mask=torch.rand(1,N,N,device=dev)>.2
 ds=(torch.rand(1,1,N,C,device=dev)>.25).to(dt)/.75
 def call(backend):
  args=(*leaves,1e-5,1e-5,C)
  if kind=='bidir':return bidirectional_trimul_triton(*args,mask=mask,dropscale=ds)
  return trimul_triton(*args,outgoing=kind=='outgoing',mask=mask,dropscale=ds)
 return leaves,call,mask,ds
