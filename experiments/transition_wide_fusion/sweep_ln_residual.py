import argparse
import gc
import json
from pathlib import Path
import torch
from common import inputs,baseline,identity,capture,paired,rel,wide
from ln_residual import ln_residual,candidate

p=argparse.ArgumentParser()
p.add_argument('--width',type=int,default=512)
p.add_argument('--length',type=int,default=384)
p.add_argument('--persistent',action='store_true')
args=p.parse_args()
d,L=args.width,args.length
out=Path('.bench/transition-wide-local/ln-residual')
out.mkdir(parents=True,exist_ok=True)
v=inputs(d,L)
expected=tuple(t.clone() for t in baseline(v))
bg,bo=capture(lambda:baseline(v))
x,ga,be,wa,wb,ws,dy=v
_,xn,rs,c1,_=wide._fwd_launch(x,ga,be,wa,wb,ws,1e-5,True)
hid,dab=wide._ext_for(x).gate(xn,dy,wide._pack(wa,wb,128),ws.t().contiguous(),True)
dxn=wide._mm_f32(dab,torch.cat((wa,wb))) if d==256 else dab@torch.cat((wa,wb))
result=dict(D=d,L=L,identity=identity(),configs=[])
configs=[(4,4,4),(4,4,8),(4,4,16),(8,4,4),(8,4,8),(4,8,4)] if args.persistent else [(4,4,0),(8,4,0),(16,4,0),(32,4,0),(32,8,0),(64,8,0)]
for bm,warps,waves in configs:
    config=dict(bm=bm,warps=warps,waves=waves)
    row=dict(config=config)
    print('CONFIG',config,flush=True)
    try:
        *_,ker=ln_residual(dxn,x,dy,ga,rs,c1,**config)
        row['resources']=dict(regs=ker.n_regs,spills=ker.n_spills,shared=ker.metadata.shared)
        actual=candidate(v,config)
        row['errors']=[rel(g,w) for g,w in zip(actual,expected)]
        assert max(row['errors'])<1e-4,row['errors']
        cg,co=capture(lambda:candidate(v,config))
        row['graph_errors']=[rel(g,w) for g,w in zip(co,actual)]
        assert max(row['graph_errors'])<1e-5,row['graph_errors']
        row['times']=paired({'baseline':bg,'candidate':cg},50)
        print('RESULT',config,row['errors'],{k:t['median_ms'] for k,t in row['times'].items()},row['resources'],flush=True)
        del cg,co,actual
    except Exception as exc:
        row['error']=repr(exc)
        print('REJECT',repr(exc),flush=True)
    result['configs'].append(row)
    suffix='-persistent' if args.persistent else ''
    (out/f'D{d}-L{L}{suffix}.json').write_text(json.dumps(result,indent=2))
    gc.collect()
    torch.cuda.empty_cache()
