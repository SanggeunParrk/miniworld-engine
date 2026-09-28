"""Apply the candidate to the original strict/graph/independent oracle harness."""
import argparse, hashlib, json, os, pathlib, runpy, sys
p=argparse.ArgumentParser();p.add_argument('--splits',type=int,default=4);p.add_argument('--index',type=int,default=0);p.add_argument('--sanitize',action='store_true');a=p.parse_args()
root=pathlib.Path('/workspace/experiments/trimul-large-d/runs');stage=root/'trimul_d256_bwd_sol90_stage2_20260923'
pre=root/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(pre))
ns={'__file__':str(pre/'check.py')};exec(compile((pre/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0],str(pre/'check.py'),'exec'),ns);sys.path.insert(0,str(stage))
from runtime import pin_lt_library
pin_lt_library()
import wide_checkpoint24 as checkpoint
from input_split import attach
old_training,old_configuration=checkpoint.Training,checkpoint.configuration
class Candidate(old_training):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);op=attach(self,a.splits,a.index)
  self.artifacts.append(self.dx.reduce_only.cubin)
  self.input_split_selection={'splits':a.splits,'index':a.index,'algo':list(op.heuristics[op.index].algo.data)}
def configuration(plan):
 r=old_configuration(plan);r['vast_input_split']=plan.input_split_selection
 r['vast_sources']={name:hashlib.sha256(pathlib.Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('input_split.py','qualify_input_split.py','runtime.py')}
 return r
checkpoint.Training=Candidate;checkpoint.configuration=configuration
os.environ.update(SLURM_ARRAY_TASK_ID='2',SLURM_JOB_ID=f'vast-split{a.splits}-index{a.index}')
if a.sanitize:os.environ['SANITIZE']='1'
runpy.run_path(str(stage/'validate_wide_checkpoint24.py'),run_name='__main__')
f=stage/f'validation-wide-checkpoint24-D512-L384-{os.environ["SLURM_JOB_ID"]}.json'
out=pathlib.Path('/workspace/vast-results/trimul-large-d');(out/f.name).write_bytes(f.read_bytes())
