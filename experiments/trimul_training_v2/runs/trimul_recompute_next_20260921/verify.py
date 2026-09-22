"""Profile final selected B1 and run CUDA memory/race checks on node02."""
from pathlib import Path
import argparse,json,subprocess,sys
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);a=ap.parse_args();n=a.length
cfg=json.loads((R/('selected-L%d.json'%n)).read_text())['b1']
check=[sys.executable,'-B',str(R/'check.py'),'--kind','b1','--length',str(n),'--splits',str(cfg['splits']),'--defines',json.dumps(cfg['defines']),'--check-only']
def run(cmd,log):
 print('START',log,flush=True)
 with (R/log).open('w') as out:
  p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
 print('END',log,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(log)
stem='ncu-b1-L%d'%n
run(['ncu','--set','full','--metrics','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','--cache-control','none','--clock-control','none','--kernel-name','regex:^b1_fused$','--launch-count','1','--force-overwrite','-o',str(R/stem),*check],stem+'.log')
for page in ('details','raw'):
 run(['ncu','--import',str(R/(stem+'.ncu-rep')),'--page',page,'--csv','--log-file',str(R/(stem+'-'+page+'.csv'))],'export-'+stem+'-'+page+'.log')
if n==384:
 for tool in ('memcheck','racecheck'):
  run(['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','kns=b1_fused',*check],'%s-b1-L%d.log'%(tool,n))
else:
 run(['compute-sanitizer','--tool','memcheck','--error-exitcode','99',sys.executable,'-B',str(R/'bench.py'),'--length',str(n),'--check-only'],'memcheck-module-L768.log')
print('DONE',n,flush=True)
