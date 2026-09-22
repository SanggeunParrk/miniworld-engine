from pathlib import Path
import sys,json,importlib.util
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_wait_folding_20260921'
sys.path.insert(0,str(S));import wait_policy as B
BASE=B.BASE
sp=importlib.util.spec_from_file_location('b1_late_weight_plan',R/'replace_plan.py');RP=importlib.util.module_from_spec(sp);sp.loader.exec_module(RP)
def make_plan(model,a,saves):
 cfg=json.loads((S/('selected-L%d.json'%model.d['n'])).read_text())['config']
 cfg['defines']['B1_LATE_WEIGHT']=3;cfg['defines']['B1_TMA_PRIORITY']=2
 cfg['defines']['B1_GATE_DEMOTE']=3 if model.d['n']==384 else 1
 return RP.Plan(dict(model.d,x=saves[-1]),a['dy'],saves[1],saves[3],**cfg)
class Training(B.Training):
 def __init__(self,a):
  super().__init__(a)
  _,k=self.forward();self.p1=make_plan(self,a,k)
