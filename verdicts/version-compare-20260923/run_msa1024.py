import os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;repo=R.parents[1]
subprocess.run([sys.executable,str(R/'block_gap.py')],check=False)
for module in ('opm','pwa'):
 for L in (384,768):
  for arm in ('engine2','pytorch','engine1'):
   env=dict(os.environ);env['PYTHONPATH']=str(R/'engine1/src' if arm=='engine1' else repo/'.release-build/h100-installed')
   key=f'{module}-{L}-{arm}';print('MSA1024',key,flush=True)
   with (R/(key+'.log')).open('w') as log:
    subprocess.run([sys.executable,str(R/'bench.py'),'--module',module,'--length',str(L),'--arm',arm,'--msa-depth','1024'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
