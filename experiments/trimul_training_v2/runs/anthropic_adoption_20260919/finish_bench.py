"""Complete a frozen retry plan and additional imported primitive fixtures."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
cases=[]
for path in sorted((root/'results').glob('*.json')):
    r=json.loads(path.read_text())
    if r.get('status') in {None,'process_failed'} or r['row'] in {'flash_sm90a','esm_t16','tx_sm90a'}:
        if r['row']=='tx_sm90a' and r['width']!=256:continue
        cases.append((path,r))
for L in (384,768):
    for C in (64,256,384):
        for row in ('native_rebuilt','v4','tx_sm90a'):
            if row=='tx_sm90a' and C!=256:continue
            path=root/'results'/f'trimul-outgoing-L{L}-C{C}-{row}.json'
            if not path.exists():cases.append((path,dict(family='trimul',length=L,width=C,row=row,direction='outgoing')))
    for row in ('triattn_native','engine_triton','pytorch_compile'):
        path=root/'results'/f'triattn-module-L{L}-C128-{row}.json'
        if not path.exists():cases.append((path,dict(family='triattn',length=L,width=128,row=row,direction='outgoing',module=True)))
    for family in ('atom_window','gather','template','dtk'):
        r=dict(family=family,length=L,width=128,row='carried',direction='outgoing')
        cases.append((root/'results'/f'{family}-L{L}.json',r))
for path,r in cases:
    cmd=[sys.executable,str(root/'bench.py'),'--output',str(path)]
    for k in ('family','length','width','direction','row'):
        if k in r:cmd+=['--'+k,str(r[k])]
    if r.get('module'):cmd+=['--module']
    print('COMPLETE',path.name,flush=True)
    with (root/'logs'/path.with_suffix('.log').name).open('w') as f:
        try:
            done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=360)
            if done.returncode and (not path.exists() or json.loads(path.read_text()).get('status') is None):
                path.write_text(json.dumps({**r,'status':'process_failed','returncode':done.returncode}))
        except subprocess.TimeoutExpired:path.write_text(json.dumps({**r,'status':'timeout'}))
