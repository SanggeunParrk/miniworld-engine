"""L384 single-B7 correction: preserve the validated gate-gradient rounding order."""
from pathlib import Path
import importlib.util,sys,os,json,hashlib
R=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
P=load('ln_fix_previous_policy',R.parent/'trimul_full_latest_20260922/policy.py')
F=load('ln_fixed_plan',R/'fixed/plan.py')
Q,H=P.Q,P.H

def make_fixed(model):
 old=model.p7;dl,dr,dg,dy=old.inputs[:4];env=P.SELECTION['environment'];saved={k:os.environ.get(k) for k in env}
 try:
  os.environ.update(env);p=F.Plan(model.d,dy,dl,dr,dg,xn=old.xn,**P.SELECTION['kwargs'])
 finally:
  for k,v in saved.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
 p.mask=old.mask;p.bind(dl,dr,dg,dy,xn=old.xn)
 return p

class Fixed(P.B1.Training):
 def __init__(self,a):
  super().__init__(a);self.p7=make_fixed(self)
