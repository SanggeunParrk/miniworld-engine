"""Development selection: BF16 tri + FP32 mean/rstd; input affine x_n retained."""
from pathlib import Path
import importlib.util,sys
R=Path(__file__).resolve().parent
SHARED=R.parent/'trimul_b1_shared_20260921'
STATS=R.parent/'trimul_ln_policy_v4_20260921'
NATIVE=R.parent/'trimul_xhat_native2_20260921'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sys.path.insert(0,str(SHARED))
BASE=load('tri_restore_shared_training',SHARED/'training.py')
BASE.N=load('tri_restore_shared_plan',SHARED/'plan.py')
sys.path.insert(0,str(STATS))
S=load('tri_restore_stats_training',STATS/'final.py')
S.RP=load('tri_restore_stats_plan',STATS/'replace_plan.py')
S.RC=load('tri_restore_stats_core',STATS/'replace_core.py');S.BASE=BASE
class Training(S.Replacement):
 """No output LN activation buffer. Reconstruct h from tri and saved statistics."""
 def __init__(self,inputs):super().__init__(inputs,-1)
def normalized_comparison():
 n=load('tri_restore_native_comparison',NATIVE/'bench.py')
 n.RP=load('tri_restore_native_plan',NATIVE/'replace_plan.py')
 n.RC=load('tri_restore_native_core',NATIVE/'replace_core.py');n.BASE=BASE
 return n
