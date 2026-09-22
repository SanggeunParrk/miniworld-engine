from pathlib import Path
import argparse,subprocess,sys,json
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);a=ap.parse_args();n=a.length
cfg=json.loads((R/('selected-L%d.json'%n)).read_text())['config']
check=[sys.executable,'-B',str(R/'check.py'),'--length',str(n),'--config',json.dumps(cfg)]
def run(cmd,name):
 print('START',name,flush=True)
 with (R/name).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=360)
 print('END',name,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(name)
for variant in ('baseline','candidate'):
 stem='ncu-%s-L%d'%(variant,n)
 cmd=['ncu','--profile-from-start','off','--set','full','--metrics','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','--cache-control','none','--clock-control','none','--kernel-name','regex:^b1_fused$','--launch-count','1','--force-overwrite','-o',str(R/stem),*check,'--profile']
 if variant=='baseline':cmd+=['--baseline']
 run(cmd,stem+'.log')
 run(['ncu','--import',str(R/(stem+'.ncu-rep')),'--page','raw','--csv','--log-file',str(R/(stem+'-raw.csv'))],stem+'-export.log')
for tool in (('memcheck','racecheck') if n==384 else ('memcheck',)):
 run(['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','kns=b1_fused',*check],'%s-L%d.log'%(tool,n))
