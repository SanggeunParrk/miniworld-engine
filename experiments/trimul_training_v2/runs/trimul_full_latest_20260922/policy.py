"""Explicit L384 latest validated selections; no production dispatch change."""
from pathlib import Path
import importlib.util,json,os,hashlib
R=Path(__file__).resolve().parent

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B1=module('full_latest_b1',R.parent/'trimul_b1_gate_demote_20260922/policy.py')
HistoricalB7=module('full_historical_b7',R.parent/'trimul_b7_unconstrained_audit_20260922/selected.py')
LatestB7=module('full_latest_b7',R.parent/'trimul_b7_weight_batch128_20260922/plan.py')
SELECTION=json.loads((R.parent/'trimul_b7_weight_batch128_20260922/selected.json').read_text())
Q=B1.BASE.OLD.Q;H=B1.BASE.OLD.H

def make_latest_b7(model):
 assert model.d['n']==384
 for path,wanted in SELECTION['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==wanted,path
 p=model.p7;dl,dr,dg,dy=p.inputs[:4]
 saved={k:os.environ.get(k) for k in SELECTION['environment']}
 try:
  os.environ.update(SELECTION['environment'])
  q=LatestB7.Plan(model.d,dy,dl,dr,dg,xn=p.xn,**SELECTION['kwargs'])
 finally:
  for k,v in saved.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
 assert hashlib.sha256(Path(q.k.unit.cubin_path).read_bytes()).hexdigest()==SELECTION['cubin']['sha256'],'B7 selected cubin mismatch'
 q.mask=p.mask;q.bind(dl,dr,dg,dy,xn=p.xn)
 return q

class Historical(B1.B.Training):
 """Same B1/B7 composition as jobs14919/14927 (~1048-1076us)."""
 def __init__(self,a):
  super().__init__(a);p=self.p7
  self.p7=HistoricalB7.make(self.d,p.inputs[3],p.inputs[0],p.inputs[1],p.inputs[2],p.xn,mask=p.mask)

Regression=B1.Training
class Latest(B1.Training):
 def __init__(self,a):
  super().__init__(a);self.p7=make_latest_b7(self)
