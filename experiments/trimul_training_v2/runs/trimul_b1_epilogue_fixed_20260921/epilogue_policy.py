"""Experimental B1 split dNorm staging; saved activation policy unchanged."""
from pathlib import Path
import sys,json,importlib.util
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b1_sol90_selected_20260921'))
import sol_policy as B
spec=importlib.util.spec_from_file_location('b1_epilogue_plan',R/'replace_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
BASE=B.B.B.BASE
class Training(B.Training):
 def __init__(self,a):
  super().__init__(a)
  _,k=self.forward()
  cfg=json.loads((R/('selected-L%d.json'%self.d['n'])).read_text())['config']
  self.p1=RP.Plan(dict(self.d,x=k[-1]),a['dy'],k[1],k[3],**cfg)
