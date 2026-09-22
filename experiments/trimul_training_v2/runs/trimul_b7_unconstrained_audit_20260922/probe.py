from pathlib import Path
import sys,importlib.util,torch,os
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
with torch.no_grad():
 a,m=H.setup(384);ref=m.p7;dl,dr,dg,dy=ref.inputs[:4]
 S=module('fast',R.parent/'trimul_b7_nextrow_20260921/gate_policy.py');fast=S.Training(a);fast()
 d=ref.d;ref.bind(dl,dr,dg,dy,xn=ref.xn);fast.p7.d=d;fast.p7.mask=ref.mask;fast.p7.bind(dl,dr,dg,dy,xn=ref.xn)
 if os.environ.get('VARIANT')=='baseline':p=fast.p7
 else:
  folder='trimul_b7_split_lnreduce_20260922' if os.environ.get('VARIANT')=='parallel' else 'trimul_b7_split_reduce_20260922';A=module('selected',R.parent/folder/'role_plan.py');cfg=dict(fast.p7.cfg);cfg.pop('saved');cfg.update(slices=1,dxctas=256)
  p=A.Plan(d,dy,dl,dr,dg,xn=ref.xn,split=True,**cfg);p.mask=ref.mask;p.bind(dl,dr,dg,dy,xn=ref.xn)
 reference=tuple(x.clone() for x in ref());torch.cuda.synchronize()
 for _ in range(2 if os.environ.get('SANITIZE') else 12):p()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();out=p();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 e=[H.rel(x,y) for x,y in zip(out,reference)];print('ERRORS',e,flush=True);assert all(x<=l for x,l in zip(e,[2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6]))
