import os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
repo=R.parents[1]
for module in ('trimul','transition','block','single','opm','pwa','dit'):
    for L in (384,768):
        for arm in ('engine2','pytorch','cuequiv','engine1'):
            if arm=='cuequiv' and module not in ('trimul','single','block'):continue
            if module=='dit' and arm=='engine1':continue # No DiTBlock in the 1.0.0 tag.
            env=dict(os.environ)
            env['PYTHONPATH']=str(R/'engine1/src' if arm=='engine1' else repo/'.release-build/h100-installed')
            key=f'{module}-{L}-{arm}'
            if (R/(key+'.json')).exists():continue
            print('START',key,time.strftime('%H:%M:%S'),flush=True)
            with (R/(key+'.log')).open('w') as log:
                try:subprocess.run([sys.executable,str(R/'bench.py'),'--module',module,'--length',str(L),'--arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                except subprocess.TimeoutExpired:print('TIMEOUT',key,flush=True)
            print('END',key,time.strftime('%H:%M:%S'),flush=True)
