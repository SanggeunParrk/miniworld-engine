"""Experimental B7 shared-store elision, LN register reuse, early xn TMA and mask reuse; selected B1 unchanged."""
from pathlib import Path
import sys,json,importlib.util
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b1_wait_folding_20260921'));import wait_policy as B
spec=importlib.util.spec_from_file_location('b7_dw_candidate',R/'role_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
BASE=B.BASE
class Training(B.Training):
 def __init__(self,a):
  super().__init__(a)
  self();old=self.p7
  self.p7=RP.Plan(self.d,old.inputs[3],old.inputs[0],old.inputs[1],old.inputs[2],xn=old.xn,split=True,skip_restore=1,overlap=0,ln_mode=2,xn_early=1,mask_hoist=1,dw_mask_early=1,weight_prefetch=1,**{k:v for k,v in old.cfg.items() if k!='saved'})
  self.p7.mask=a['mask']
