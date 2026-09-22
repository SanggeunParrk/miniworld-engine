"""Compare source candidates using identical tensor addresses and rotated order."""
from ring_plan import *
from multicast_plan import MulticastRingPlan,MulticastWarpPlan
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('--length',type=int,required=True)
ap.add_argument('--splits',type=int,required=True)
ap.add_argument('--source-splits',help='Comma-separated split counts in source order; compare each tuned schedule')
ap.add_argument('--source-counts',help='Comma-separated CTA counts in source order; cluster residency is checked')
ap.add_argument('--ring',action='store_true')
ap.add_argument('--vary-ring',action='store_true',help='Share one maximum allocation across different ring windows')
ap.add_argument('--label',required=True)
ap.add_argument('sources',nargs='+')
args=ap.parse_args()
split_counts=([int(v) for v in args.source_splits.split(',')]
              if args.source_splits else [args.splits]*len(args.sources))
if len(split_counts)!=len(args.sources):ap.error('--source-splits must match the number of sources')
cta_counts=([int(v) for v in args.source_counts.split(',')]
            if args.source_counts else [264]*len(args.sources))
if len(cta_counts)!=len(args.sources):ap.error('--source-counts must match the number of sources')
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);cls=RingPlan if args.ring else WarpPlan
 plans=[];records={};gs={'baseline':capture(lambda:baseline(a))}
 count_size=max(2+json.loads((R/(source+'.launch.json')).read_text()).get('extra_counts',0) for source in args.sources)
 shared_counts=torch.zeros(count_size,device='cuda',dtype=torch.int32)
 configs=[json.loads((R/(source+'.launch.json')).read_text()) for source in args.sources]
 max_weight_elements=max(8*s*c.get('wgrad_slices',1)*16384 for s,c in zip(split_counts,configs))
 shared_partw=torch.empty(max_weight_elements,device='cuda')
 shared_partln=torch.empty((max(c-8*s for c,s in zip(cta_counts,split_counts)),256),device='cuda')
 shared_ring=None
 if args.vary_ring:
  assert args.ring
  max_window=max(json.loads((R/(source+'.launch.json')).read_text())['ring_tiles'] for source in args.sources)
  shared_ring=torch.empty(1024*max_window*64,device='cuda',dtype=torch.bfloat16)
 for source,splits,count,config in zip(args.sources,split_counts,cta_counts,configs):
  try:
   candidate_cls=(MulticastRingPlan if args.ring else MulticastWarpPlan) if config.get('cluster_size') else cls
   p=candidate_cls(a,count=count,splits=splits,source=source)
   p.counts=shared_counts
   p.partw=shared_partw[:p.partw.numel()].view(p.partw.shape)
   p.partln=shared_partln[:count-8*splits]
   if shared_ring is not None:
    width=p.config['ring_tiles']*64;p.ring=shared_ring[:1024*width].view(1024,width)
   if plans:
    shared=plans[0]
    for name in ('dx','dw','dgam','dbeta','counts'):
     assert getattr(p,name).shape==getattr(shared,name).shape
     setattr(p,name,getattr(shared,name))
    if args.ring and shared_ring is None:
     assert p.ring.shape==shared.ring.shape
     p.ring=shared.ring
    p.debugdc=p.dx;p.debugxn=p.dx
    p.outputs=(p.dx,*p.dw.unbind(0),p.dgam,p.dbeta)
   p.bind(a['dl'],a['dr'],a['dg'],a['dy'])
   p();torch.cuda.synchronize();es=errors(p.outputs,ref)
   valid=all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items())
   valid=valid and not bool(p.counts.any())
   records[source]=dict(count=count,splits=splits,valid=valid,errors=es)
   if valid:gs[source]=capture(p)
   plans.append(p)
   print('CHECK',source,valid,flush=True)
  except RuntimeError as e:
   records[source]=dict(valid=False,reason=str(e));print('REJECTED',source,str(e),flush=True)
 keys=list(gs);blocks=[]
 for i in range(3):
  order=keys[i:]+keys[:i];b=paired({k:gs[k] for k in order});blocks.append(b)
  print('BLOCK',i,{k:v['median_us'] for k,v in b.items()},flush=True)
 ts={}
 for k in gs:
  vals=sorted(v for b in blocks for v in b[k]['samples_us'])
  ts[k]=dict(median_us=statistics.median(vals),samples_us=vals)
 out=dict(L=args.length,splits=args.splits,source_splits=dict(zip(args.sources,split_counts)),source_counts=dict(zip(args.sources,cta_counts)),ring=args.ring,shared_addresses=True,
          records=records,times=ts,blocks=blocks)
 (R/f'{args.label}-L{args.length}.json').write_text(json.dumps(out,indent=2))
 print('RESULT',{k:v['median_us'] for k,v in ts.items()},flush=True)
