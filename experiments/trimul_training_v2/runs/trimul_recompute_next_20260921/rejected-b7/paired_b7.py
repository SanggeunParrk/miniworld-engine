"""Final B7 variants: equal buffers, interleaved graph timing, strict gradients."""
from pathlib import Path
import argparse,json,torch
import plans as P
import compare_cueq_training as Q
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args()
base=dict(PIPE_DW=1,SMEM_BYTES=221184,DX_RESIDENT=1)
configs={
 'previous':dict(splits=4,defines=base),
 'stats':dict(splits=4,defines=dict(base,B7_REUSE_STATS=1)),
 'stats_early_mask':dict(splits=4,defines=dict(base,B7_REUSE_STATS=1,B7_WEIGHT_EARLY=1,B7_HOIST_MASK=1)),
}

with torch.no_grad():
 a=Q.setup(args.length);d=a['d'];ref=Q.baseline(a);gs={};ps={};es={}
 for name,cfg in configs.items():
  p=P.B7(d,a['dy'],a['dl'],a['dr'],a['dg'],**cfg)
  if ps:
   first=next(iter(ps.values()))
   for key in ('dx','dw','dgam','dbeta','partw','partln','counts'):setattr(p,key,getattr(first,key))
   p.outputs=(p.dx,*p.dw.unbind(),p.dgam,p.dbeta);p.bind(a['dl'],a['dr'],a['dg'],a['dy'])
  p.mask=a['mask'];p.bind(a['dl'],a['dr'],a['dg'],a['dy']);out=p();torch.cuda.synchronize()
  names=('dx','dwl','dwlg','dwr','dwrg','dgi','dbi');limits=(2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6)
  err={k:dict(rel=Q.rel(x,y),finite=bool(torch.isfinite(x).all()),limit=lim) for k,x,y,lim in zip(names,out,ref,limits)}
  assert all(e['finite'] and e['rel']<=e['limit'] for e in err.values()),err
  es[name]=err;ps[name]=p;gs[name],_=Q.capture_outputs(p)
 blocks=[Q.paired(gs,iterations=200) for _ in range(4)]
 result=dict(L=args.length,configs=configs,errors=es,blocks=blocks,times=Q.pool(blocks),same_buffers=True)
 (R/('paired-b7-L%d.json'%args.length)).write_text(json.dumps(result,indent=2));print('RESULT',{k:v['median_us'] for k,v in result['times'].items()},flush=True)
