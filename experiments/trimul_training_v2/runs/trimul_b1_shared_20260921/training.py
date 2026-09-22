"""Development-only TriMul: unchanged saved-x_n forward/B7, shared B1-B4.

The independent-reference L768 B7 accuracy issue still prevents production
promotion. See selected-L*.json and module-L*.json before using measurements.
"""
import importlib.util,json
from pathlib import Path
import plan as N
spec=importlib.util.spec_from_file_location('trimul_previous_training',N.OLD/'bench.py')
OLD=importlib.util.module_from_spec(spec);spec.loader.exec_module(OLD)
class Training(OLD.Training):
 def __init__(self,a):
  super().__init__(a,OLD.CONFIGS['split_xn_pc1'])
  _,kept=self.forward()
  cfg=json.loads((N.R/('selected-L%d.json'%self.d['n'])).read_text())['config']
  self.p1=N.Plan(dict(self.d,x=kept[-1]),a['dy'],kept[1],**cfg)
