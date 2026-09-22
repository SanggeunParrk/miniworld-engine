from pathlib import Path
import sys,json,importlib.util
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_wait_folding_20260921'
sys.path.insert(0,str(S));import wait_policy as B
BASE=B.BASE
sp=importlib.util.spec_from_file_location('b1_late_weight_plan',R/'replace_plan.py');RP=importlib.util.module_from_spec(sp);sp.loader.exec_module(RP)
class Training(B.Training):
 def __init__(self,a):
  super().__init__(a)
  _,k=self.forward();cfg=json.loads((S/('selected-L%d.json'%self.d['n'])).read_text())['config']
  cfg['defines']['B1_LATE_WEIGHT']=3
  self.p1=RP.Plan(dict(self.d,x=k[-1]),a['dy'],k[1],k[3],**cfg)
