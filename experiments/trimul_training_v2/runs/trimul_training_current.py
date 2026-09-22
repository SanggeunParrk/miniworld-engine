"""Current explicit CUDA development route for all five pair widths.

B=1, L384/768, BF16, direction hidden=D, concatenated hidden=2D.
D128 retains cache-policy B1 and validated single B7 at both lengths.
D64/256/384/512 reuse CUDA B1 and the optimized single CUDA B7.
The wide port is functional but slower than Triton; see the current report.
Engine automatic dispatch is not changed by this development entry.
"""
from pathlib import Path
import importlib.util,sys
from functools import lru_cache
_R=Path(__file__).resolve().parent
@lru_cache(None)
def _width_module():
 root=_R/'trimul_cuda_widths_opt_20260923'
 if str(root) not in sys.path:sys.path.insert(0,str(root))
 import width_plan
 import width_autograd as entry
 return width_plan,entry
class _WidthTraining:
 def __init__(self,a):
  self.a,self.d=a,a['d'];d=self.d;P,_=_width_module()
  self.model=P.Training(*d['leaves'],d['mask'],d['ds'],a['dy'])
 def forward(self):return self.model.forward(),None
 def backward(self,kept=None):return self.model.backward(self.a['dy'])
 def __call__(self):y,_=self.forward();return y,self.backward()
class Training:
 def __new__(cls,a):
  if a['d']['x'].shape[-1]==128:
   _,W=_width_module();return W.training128(a)
  return _WidthTraining(a)
def bidirectional_trimul_cuda(*args,**kwargs):
 """PyTorch autograd entry for D64/128/256/384/512. No Triton bwd fallback."""
 return _width_module()[1].bidirectional_trimul_cuda(*args,**kwargs)
__all__=['Training','bidirectional_trimul_cuda']
