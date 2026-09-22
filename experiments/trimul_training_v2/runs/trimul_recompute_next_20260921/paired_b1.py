"""Equal buffer, strict checks and interleaved B1 timing."""
from pathlib import Path
import argparse,json,torch
import plans as P
import compare_cueq_training as Q
from compare_all_training import BoundB1
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args()
configs={
 'previous':dict(splits=26,defines=dict(GATE_SPLITS=36,TRAIN_L=args.length)),
 'partition_only':dict(splits=24,defines=dict(GATE_SPLITS=36,TRAIN_L=args.length)),
 'projection_pipe':dict(splits=24,defines=dict(GATE_SPLITS=36,TRAIN_L=args.length,B1_DWPROJ_PIPE=1)),
}
with torch.no_grad():
 a=Q.setup(args.length);d=a['d'];old=BoundB1(d,a['dy'],a['s'],132,2,'dual_ln_prefetch');ref=tuple(t.clone() for t in old())
 gs={};ps={};es={};partw=torch.empty((132,16384),device='cuda');partln=torch.empty((132,512),device='cuda');counts=torch.zeros(134,device='cuda',dtype=torch.int32)
 for name,cfg in configs.items():
  p=P.B1(d,a['dy'],a['s'][0].saved_tensors[11],**cfg)
  if ps:p.outputs=next(iter(ps.values())).outputs
  p.partw=partw;p.partln=partln;p.counts=counts;p.bind(a['dy'],a['s'][0].saved_tensors[11])
  out=p();torch.cuda.synchronize()
  names=('dg','dwg','dt','dgo','dbo','dwp');limits=(0,5e-4,2e-5,5e-6,5e-6,5e-4)
  err={k:dict(rel=Q.rel(x,y),finite=bool(torch.isfinite(x).all()),limit=lim) for k,x,y,lim in zip(names,out,ref,limits)}
  assert all(e['finite'] and e['rel']<=e['limit'] for e in err.values()),err
  es[name]=err;ps[name]=p;gs[name],_=Q.capture_outputs(p)
 blocks=[Q.paired(gs,iterations=200) for _ in range(4)]
 result=dict(L=args.length,configs=configs,errors=es,blocks=blocks,times=Q.pool(blocks),same_buffers=True)
 (R/('paired-b1-L%d.json'%args.length)).write_text(json.dumps(result,indent=2));print('RESULT',{k:v['median_us'] for k,v in result['times'].items()},flush=True)
