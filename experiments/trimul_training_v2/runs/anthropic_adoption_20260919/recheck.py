"""Recheck setup failures after ABI/build/fixture fixes; retain original records."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--lane',type=int,default=0);a=p.parse_args()
root=Path(__file__).resolve().parent
cases=[]
for path in sorted((root/'results').glob('*.json')):
    r=json.loads(path.read_text())
    retry=(r.get('status') in {None,'process_failed'} or
           (r['family']=='ln' and r['row'] in {'pytorch','engine_triton'}) or
           (r['row']=='engine_triton' and r.get('engine_backend')!='triton') or
           (r['row'] in {'native_rebuilt','esm_t16','flash_sm90a'}))
    if retry:cases.append((path,r))
for L in (384,768):
    for C in (64,256,384):
        for row in ('native_rebuilt','v4','tx_sm90a'):
            if row=='tx_sm90a' and C!=256:continue
            r=dict(family='trimul',length=L,width=C,row=row,direction='outgoing')
            path=root/'results'/f'trimul-outgoing-L{L}-C{C}-{row}.json'
            cases.append((path,r))
    for row in ('triattn_native','engine_triton','pytorch_compile'):
        r=dict(family='triattn',length=L,width=128,row=row,direction='outgoing',module=True)
        path=root/'results'/f'triattn-module-L{L}-C128-{row}.json'
        cases.append((path,r))
for path,r in cases[a.lane::2]:
    old=root/'setup-attempts'/path.name;old.parent.mkdir(exist_ok=True)
    if path.exists() and not old.exists():old.write_bytes(path.read_bytes())
    cmd=[sys.executable,str(root/'bench.py'),'--output',str(path)]
    for k in ['family','length','width','direction','row']:
        if k in r:cmd+=['--'+k,str(r[k])]
    if r.get('module'):cmd+=['--module']
    print('RECHECK',path.name,flush=True)
    with (root/'logs'/path.with_suffix('.log').name).open('w') as f:
        try:
            done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=360)
            if done.returncode and (not path.exists() or path.read_bytes()==old.read_bytes()):
                path.write_text(json.dumps({**r,'status':'process_failed','returncode':done.returncode}))
        except subprocess.TimeoutExpired:path.write_text(json.dumps({**r,'status':'timeout'}))
