from pathlib import Path
import argparse,subprocess,sys,json,os
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args();n=args.length
records=[]
def run(cmd,name):
 print('START',name,flush=True)
 with (R/name).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=480,env=({**{k:v for k,v in os.environ.items() if k != "PYTHONPATH"},"PYTHONNOUSERSITE":"1"} if "--import" in cmd else None))
 records.append(dict(log=name,returncode=p.returncode));(R/('verification-L%d.json'%n)).write_text(json.dumps(records,indent=2));print('END',name,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(name)
for variant in ('baseline','optimized'):
 stem='ncu-%s-L%d'%(variant,n)
 cmd=['ncu','--profile-from-start','off','--set','full','--metrics','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','--cache-control','none','--clock-control','none','--kernel-name','regex:^b1_fused$','--launch-count','1','--force-overwrite','-o',str(R/stem),sys.executable,'-B',str(R/'profile_one.py'),'--length',str(n),'--variant',variant]
 run(cmd,stem+'.log');run(['ncu','--import',str(R/(stem+'.ncu-rep')),'--page','raw','--csv','--log-file',str(R/(stem+'-raw.csv'))],stem+'-export.log')
for tool in (('memcheck','racecheck') if n==384 else ('memcheck',)):
 cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99']
 if tool=='racecheck':cmd+=['--kernel-name','kns=b1_fused','--kernel-name','kns=save_k3']
 run([*cmd,sys.executable,'-B',str(R/'bench.py'),'--length',str(n),'--check-only'],'%s-L%d.log'%(tool,n))
