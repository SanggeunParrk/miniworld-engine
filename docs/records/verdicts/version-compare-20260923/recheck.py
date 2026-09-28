"""Recheck measurements made before fixing capture warmup and Transition compilation."""
import json,os,subprocess,sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
repo=R.parents[1]
archive=R/'initial';archive.mkdir(exist_ok=True)
for path in sorted(R.glob('*.json')):
    if path.name.endswith('trace.json') or path.name=='manifest.json':continue
    d=json.loads(path.read_text())
    if 'arm' not in d:continue
    if d.get('harness_revision')==2:continue
    env=dict(os.environ)
    env['PYTHONPATH']=str(R/'engine1/src' if d['arm']=='engine1' else repo/'.release-build/h100-installed')
    shutil.copyfile(path,archive/path.name)
    print('RECHECK',path.stem,flush=True)
    with path.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,str(R/'bench.py'),'--module',d['module'],'--length',str(d['L']),'--arm',d['arm']],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
subprocess.run([sys.executable,str(R/'report.py')],check=True)
