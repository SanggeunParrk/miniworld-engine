import argparse
import gc
import json
from pathlib import Path
import torch
from common import inputs, baseline, identity, capture, paired, rel, wide
from dx_ln import dx_ln,candidate

p=argparse.ArgumentParser()
p.add_argument('--width',type=int,default=512)
p.add_argument('--length',type=int,default=384)
args=p.parse_args()
d,L=args.width,args.length
out=Path('.bench/transition-wide-local/dx-ln')
out.mkdir(parents=True,exist_ok=True)
v=inputs(d,L)
expected=tuple(t.clone() for t in baseline(v))
bg,bo=capture(lambda:baseline(v))
x,ga,be,wa,wb,ws,dy=v
_,xn,rs,c1,_=wide._fwd_launch(x,ga,be,wa,wb,ws,1e-5,True)
hid,dab=wide._ext_for(x).gate(xn,dy,wide._pack(wa,wb,128),ws.t().contiguous(),True)
wab=torch.cat((wa,wb))
result=dict(D=d,L=L,identity=identity(),configs=[])
for bm,bk,warps,stages in [(32,64,8,2),(64,64,8,2),(64,32,8,3),(32,64,4,2),(32,32,4,3),(16,64,4,2),(64,64,8,3),(128,32,8,3)]:
    config=dict(bm=bm,bk=bk,warps=warps,stages=stages)
    print('CONFIG',config,flush=True)
    row=dict(config=config)
    try:
        *_,ker=dx_ln(dab,wab,x,dy,ga,rs,c1,**config)
        row['resources']=dict(regs=ker.n_regs,spills=ker.n_spills,shared=ker.metadata.shared)
        actual=candidate(v,config)
        errors=[rel(g,w) for g,w in zip(actual,expected)]
        row['errors']=errors
        assert max(errors)<.003,errors
        cg,co=capture(lambda:candidate(v,config))
        row['graph_errors']=[rel(g,w) for g,w in zip(co,actual)]
        assert max(row['graph_errors'])<1e-5,row['graph_errors']
        row['times']=paired({'baseline':bg,'candidate':cg},30)
        print('RESULT',config,errors,{k:t['median_ms'] for k,t in row['times'].items()},row['resources'],flush=True)
        del cg,co,actual
    except Exception as exc:
        row['error']=repr(exc)
        print('REJECT',repr(exc),flush=True)
    result['configs'].append(row)
    (out/f'D{d}-L{L}.json').write_text(json.dumps(result,indent=2))
    gc.collect()
    torch.cuda.empty_cache()
