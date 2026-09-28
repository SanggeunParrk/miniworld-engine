import importlib.util,torch
from miniworld_engine import settings
spec=importlib.util.spec_from_file_location('cases','tests/integrations/test_trimul_single_h100_gpu.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
settings.configure(engine_backend='auto')
for n in (384,768):
 for direction in (True,False):
  m,x,mask,ds=c.setup(n,direction)
  y=m(x,mask);g=torch.autograd.grad(y,(x,*m.parameters()),torch.randn_like(y))
  torch.cuda.synchronize()
  assert all(torch.isfinite(v).all() for v in (y,*g))
  print('SANITIZED',n,direction,flush=True)
