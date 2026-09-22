"""Retune CTA role counts with shared addresses and the fixed numerical limits."""
from ring_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True)
ap.add_argument('--source',required=True);ap.add_argument('--ring',action='store_true')
ap.add_argument('--splits',type=int,nargs='+',required=True)
args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);cls=RingPlan if args.ring else WarpPlan
 ps=[cls(a,count=264,splits=s,source=args.source) for s in args.splits]
 shared=ps[0];pw=torch.empty(8*max(args.splits)*2*16384,device='cuda')
 pl=torch.empty((264-8*min(args.splits),256),device='cuda')
 gs={'baseline':capture(lambda:baseline(a))};records={}
 for sp,p in zip(args.splits,ps):
  for name in ('dx','dw','dgam','dbeta','counts'):
   setattr(p,name,getattr(shared,name))
  if args.ring:p.ring=shared.ring
  p.partw=pw[:8*sp*2*16384].view(8,sp*2,16384);p.partln=pl[:264-8*sp]
  p.outputs=(p.dx,*p.dw.unbind(0),p.dgam,p.dbeta)
  p.bind(a['dl'],a['dr'],a['dg'],a['dy']);p();torch.cuda.synchronize()
  es=errors(p.outputs,ref);key='splits'+str(sp)
  valid=all(e['finite'] and e['relative_l2']<=LIMITS[k] for k,e in es.items())
  records[key]=dict(splits=sp,errors=es,valid=valid)
  if valid:gs[key]=capture(p)
  print('CHECK',sp,valid,flush=True)
 blocks=[];keys=list(gs)
 for i in range(3):
  order=keys[i:]+keys[:i];t=paired({k:gs[k] for k in order})
  blocks.append(t);print('BLOCK',i,{k:v['median_us'] for k,v in t.items()},flush=True)
 times={}
 for key in keys:
  vals=sorted(v for b in blocks for v in b[key]['samples_us'])
  times[key]=dict(median_us=statistics.median(vals),samples_us=vals)
 out=dict(source=args.source,L=args.length,ring=args.ring,shared_addresses=True,
          records=records,times=times,blocks=blocks)
 (R/f'{args.source}-shared-role-tune-L{args.length}.json').write_text(json.dumps(out,indent=2))
 print('RESULT',{k:v['median_us'] for k,v in times.items()},flush=True)
