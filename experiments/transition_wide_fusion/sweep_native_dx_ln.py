import argparse
import gc
import json
from pathlib import Path
import torch
from common import inputs,baseline,identity,capture,paired,rel,wide
from native_dx_ln import dx_ln,candidate

p=argparse.ArgumentParser()
p.add_argument('--width',type=int,default=512)
p.add_argument('--length',type=int,default=384)
args=p.parse_args()
d,L=args.width,args.length
out=Path('.bench/transition-wide-local/native-dx-ln')
out.mkdir(parents=True,exist_ok=True)
v=inputs(d,L)
expected=tuple(t.clone() for t in baseline(v))
bg,bo=capture(lambda:baseline(v))
x,ga,be,wa,wb,ws,dy=v
_,xn,rs,c1,_=wide._fwd_launch(x,ga,be,wa,wb,ws,1e-5,True)
hid,dab=wide._ext_for(x).gate(xn,dy,wide._pack(wa,wb,128),ws.t().contiguous(),True)
result=dict(D=d,L=L,identity=identity(),configs=[])
configs=[(64,2,0,1),(32,4,0,1),(32,4,1,1),(32,3,1,1)]
if d==384:
    configs.extend([(64,3,0,1),(64,2,1,1)])
for tbk,stages,stgdx,xb in configs:
    config=dict(tbk=tbk,stages=stages,stgdx=stgdx,xb=xb)
    row=dict(config=config)
    print('CONFIG',config,flush=True)
    try:
        *_,ker=dx_ln(dab,torch.cat((wa,wb)),x,dy,ga,rs,c1,**config)
        row['resources']=dict(regs=ker.regs,local_bytes=ker.lmem,shared=ker.smem)
        actual=candidate(v,config)
        row['errors']=[rel(g,w) for g,w in zip(actual,expected)]
        assert max(row['errors'])<1e-4,row['errors']
        cg,co=capture(lambda:candidate(v,config))
        row['graph_errors']=[rel(g,w) for g,w in zip(co,actual)]
        assert max(row['graph_errors'])<1e-5,row['graph_errors']
        row['times']=paired({'baseline':bg,'candidate':cg},35)
        print('RESULT',config,row['errors'],{k:t['median_ms'] for k,t in row['times'].items()},row['resources'],flush=True)
        del cg,co,actual
    except Exception as exc:
        row['error']=repr(exc)
        print('REJECT',repr(exc),flush=True)
    result['configs'].append(row)
    (out/f'D{d}-L{L}.json').write_text(json.dumps(result,indent=2))
    gc.collect()
    torch.cuda.empty_cache()
