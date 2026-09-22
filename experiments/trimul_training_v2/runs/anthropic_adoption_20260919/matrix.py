"""Serial candidates per GPU; invoke with --lane 0/1 on two independent GPUs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--lane',type=int,default=0);p.add_argument('--lanes',type=int,default=2)
p.add_argument('--only',default='');a=p.parse_args()
root=Path(__file__).resolve().parent
cases=[]
for L in (384,768):
    for direction in ('outgoing','incoming'):
        for row in ('native_rebuilt','v4','tmk3_fast','tmk3_exact','esm_shapes','esm_v5_fwd','tx_sm90a','engine_triton','pytorch_compile'):
            cases.append(dict(family='trimul',length=L,width=128,direction=direction,row=row))
    for C in (128,256,384,512):
        for row in ('v2','v1','pf','lnl','af3_fused','engine_triton','pytorch_compile')+ (('flash_sm90a','esm_t16','esm_fused_exact') if C==256 else ()):
            cases.append(dict(family='transition',length=L,width=C,row=row))
    for row in ('k2b','k2','flash','cuda_sm90a','triattn_native','engine_triton','cueq'):
        cases.append(dict(family='triattn',length=L,width=128,row=row))
    for C in (128,256,384,512):
        for row in ('fastln','exactln:triton','ln_rows','dtk_ln','engine_triton','pytorch'):
            cases.append(dict(family='ln',length=L,width=C,row=row))
    for row in ('apb_attn','fpf_apb','l3a','dtk_loop','sba','dit_exact'):
        cases.append(dict(family='apb',length=L,width=384,row=row))
jobs=[]
for i,c in enumerate(cases):
    if i%a.lanes!=a.lane or (a.only and c['family'] not in a.only.split(',')):continue
    name=f"{c['family']}-{c.get('direction','outgoing')}-L{c['length']}-C{c['width']}-{c['row']}"
    dst=root/'results'/f'{name}.json';log=root/'logs'/f'{name}.log'
    log.parent.mkdir(exist_ok=True)
    if dst.is_file():
        print('EXISTS',name,flush=True);continue
    cmd=[sys.executable,str(root/'bench.py'),'--output',str(dst)]
    for k,v in c.items():cmd.extend(['--'+k,str(v)])
    print('START',name,flush=True)
    with log.open('w') as f:
        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=360)
        except subprocess.TimeoutExpired:
            dst.parent.mkdir(exist_ok=True)
            dst.write_text(json.dumps({**c,'status':'timeout','reason':'360s per candidate'}));continue
    if not dst.is_file():dst.write_text(json.dumps({**c,'status':'process_failed','returncode':r.returncode}))
    d=json.loads(dst.read_text());print('END',name,d.get('status'),d.get('ms'),d.get('reason','')[:250],flush=True)
