"""Profile the fastest validated upstream candidate and engine baseline per shape."""
import argparse,json,subprocess,sys,os
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--lane',type=int,default=0);p.add_argument('--lanes',type=int,default=2);p.add_argument('--only',default='');p.add_argument('--modules',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent
groups={}
for path in (root/'results').glob('*.json'):
    r=json.loads(path.read_text())
    if r.get('status')!='passed' or not r.get('ms'):continue
    if bool(r.get('module')) != a.modules:continue
    if a.only and r['family'] not in a.only.split(','):continue
    if r['family']=='trimul' and r.get('direction')!='outgoing':continue
    key=(r['family'],r['length'],r['width'])
    groups.setdefault(key,[]).append(r)
jobs=[]
for key,rows in sorted(groups.items()):
    up=[r for r in rows if r['row'] not in {'engine_triton','pytorch','pytorch_compile','cueq'}]
    base=[r for r in rows if r['row']=='engine_triton']
    if up:jobs.append(min(up,key=lambda r:r['ms']))
    if base:jobs.append(base[0])
(root/f'profile-plan-{a.lane}.json').write_text(json.dumps(jobs[a.lane::a.lanes],indent=2))
for r in jobs[a.lane::a.lanes]:
    name=f"{r['family']}{'-module' if a.modules else ''}-L{r['length']}-C{r['width']}-{r['row']}"
    dst=root/'ncu'/name;dst.parent.mkdir(exist_ok=True)
    if dst.with_suffix('.ncu-rep').exists():continue
    cmd=['/usr/local/cuda-12.9/bin/ncu','--target-processes','all','--profile-from-start','off',
         '--clock-control','none','--cache-control','none',
         '--section','SpeedOfLight','--section','SpeedOfLight_RooflineChart','--section','MemoryWorkloadAnalysis',
         '--section','SpeedOfLight_HierarchicalTensorRooflineChart','--section','SchedulerStats',
         '--section','LaunchStats','--section','Occupancy','--force-overwrite','--export',str(dst),
         sys.executable,str(root/'bench.py'),'--family',r['family'],'--row',r['row'],
         '--length',str(r['length']),'--width',str(r['width']),'--profile','--output',str(dst)+'.json']
    if a.modules:cmd.append('--module')
    print('PROFILE',name,flush=True)
    with open(str(dst)+'.log','w') as f:
        try:
            completed=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900,env={**os.environ,'PYTHONNOUSERSITE':'1'})
            print('DONE',name,completed.returncode,flush=True)
        except subprocess.TimeoutExpired:print('TIMEOUT',name,flush=True)
    if dst.with_suffix('.ncu-rep').exists():
        with open(str(dst)+'.csv','w') as f:
            subprocess.run(['/usr/local/cuda-12.9/bin/ncu','--import',str(dst)+'.ncu-rep','--page','raw','--csv'],stdout=f,check=True,env={**os.environ,"PYTHONNOUSERSITE":"1"})
