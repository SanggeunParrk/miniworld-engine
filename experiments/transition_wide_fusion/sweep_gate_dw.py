import argparse
import gc
import json
from pathlib import Path
import torch
from common import inputs,baseline,identity,capture,paired,rel,wide
from gate_dw import gate_dw,candidate

p=argparse.ArgumentParser()
p.add_argument('--width',type=int,default=256)
p.add_argument('--length',type=int,default=384)
args=p.parse_args()
d,L=args.width,args.length
out=Path('.bench/transition-wide-local/gate-dw')
out.mkdir(parents=True,exist_ok=True)
v=inputs(d,L)
expected=tuple(t.clone() for t in baseline(v))
bg,bo=capture(lambda:baseline(v))
x,ga,be,wa,wb,ws,dy=v
_,xn,rs,c1,_=wide._fwd_launch(x,ga,be,wa,wb,ws,1e-5,True)
result=dict(D=d,L=L,identity=identity(),configs=[])
for bm,bh,warps,split in [(32,32,8,8),(32,32,8,16),(64,32,8,8),(32,16,4,4),(64,16,8,4),(64,64,8,8)]:
    config=dict(bm=bm,bh=bh,warps=warps,split=split)
    row=dict(config=config)
    print('CONFIG',config,flush=True)
    try:
        *_,ker=gate_dw(xn,dy,wa,wb,ws,**config)
        row['resources']=dict(regs=ker.n_regs,spills=ker.n_spills,shared=ker.metadata.shared)
        actual=candidate(v,config)
        row['errors']=[rel(g,w) for g,w in zip(actual,expected)]
        assert max(row['errors'])<.006,row['errors']
        cg,co=capture(lambda:candidate(v,config))
        row['graph_errors']=[rel(g,w) for g,w in zip(co,actual)]
        assert max(row['graph_errors'])<1e-5,row['graph_errors']
        row['times']=paired({'baseline':bg,'candidate':cg},25)
        print('RESULT',config,row['errors'],{k:t['median_ms'] for k,t in row['times'].items()},row['resources'],flush=True)
        del cg,co,actual
    except Exception as exc:
        row['error']=repr(exc)
        print('REJECT',repr(exc),flush=True)
    result['configs'].append(row)
    (out/f'D{d}-L{L}.json').write_text(json.dumps(result,indent=2))
    gc.collect()
    torch.cuda.empty_cache()
