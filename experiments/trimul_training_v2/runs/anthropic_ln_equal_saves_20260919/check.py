from core_saved import *
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
with torch.no_grad():
 for n in (64,72):
  d=setup(n);cf=(2,64,6,1,0);a,sa=forward(d,True,cf);b,sb=forward(d,False,cf);dy=torch.randn_like(a)
  print('FWD',n,rel(a,b),[rel(u,v) for u,v in zip(sa[0].saved_tensors,sb[0].saved_tensors)],flush=True)
  ga=backward(d,sa,dy);gb=backward(d,sb,dy);es=[rel(u,v) for u,v in zip(ga,gb)];print('BWD',es,flush=True);assert max(es)<.001
  with torch.enable_grad():
   yr=d['reference']('anthropic_cuda');gr=torch.autograd.grad(yr,d['leaves'],dy)
  er=[rel(u,v) for u,v in zip((a,*ga),(yr,*gr))];print('REFERENCE',er,flush=True);assert max(er)<.015
print('PASS',flush=True)
