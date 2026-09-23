import os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;repo=R.parents[1]
for phase in ('before','after'):
 for module in ('opm','pwa'):
  for L in (384,768):
   env=dict(os.environ);env['FUSION_BASELINE']='1' if phase=='before' else '0'
   env['PYTHONPATH']=str(repo/'.release-build/h100-installed' if phase=='before' else repo/'src')
   key=f'{module}-{L}-engine2';print(phase,key,flush=True)
   with (R/phase/(key+'.log')).open('w') as f:
    subprocess.run([sys.executable,str(R/'launch.py'),str(R/phase/'bench.py'),'--module',module,'--length',str(L),'--arm','engine2','--msa-depth','1024'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
for tool,case in [('memcheck','opm'),('racecheck','pwa')]:
 print('sanitizer',tool,case,flush=True)
 with (R/f'{tool}-{case}.log').open('w') as f:
  subprocess.run(['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool',tool,'--error-exitcode','86',sys.executable,str(R/'launch.py'),str(R/'sanitize.py'),case],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
