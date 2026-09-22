from core_saved import *
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
records=[]
with torch.no_grad():
 for row in json.loads((R/'results.json').read_text()):
  for n in (64,72):
   d=setup(n);cf=row['configs'];a,sa=forward(d,True,tuple(cf['fused']),tuple(cf['ln']));b,sb=forward(d,False,tuple(cf['split']),tuple(cf['ln']));assert torch.equal(a,b)
   assert all(torch.equal(u,v) for u,v in zip(sa[0].saved_tensors,sb[0].saved_tensors));assert all(torch.equal(u,v) for u,v in zip(sa[1:],sb[1:]))
   dy=torch.randn_like(a);ga=backward(d,sa,dy);gb=backward(d,sb,dy);es=[rel(u,v) for u,v in zip(ga,gb)];assert max(es)<.001
   with torch.enable_grad():
    yr=d['reference']('anthropic_cuda');gr=torch.autograd.grad(yr,d['leaves'],dy)
   er=[rel(u,v) for u,v in zip((a,*ga),(yr,*gr))];assert max(er)<.015
   d['ds'].zero_();z,_=forward(d,True,tuple(cf['fused']),tuple(cf['ln']));assert torch.equal(z,d['x'])
   records.append(dict(selected_for=row['N'],tested_N=n,forward_and_all_saves_bitwise_equal=True,gradient_relative_l2=es,engine_reference_relative_l2=er,zero_dropout_scale_returns_residual=True))
 (R/'validation.json').write_text(json.dumps(records,indent=2))
print('PASS',flush=True)
