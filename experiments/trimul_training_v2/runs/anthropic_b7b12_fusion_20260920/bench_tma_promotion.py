"""Same allocation and cubin, varying only TMA L2-promotion descriptor fields."""
from ring_plan import *
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--length',type=int,required=True)
ap.add_argument('--ring',action='store_true')
ap.add_argument('--source')
args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a)
 cls=RingPlan if args.ring else WarpPlan
 source=args.source or ('front_ring96_cache3' if args.ring else 'front_prefetch_lnpair_storepipe')
 p=cls(a,count=264,splits=20 if args.ring else 13,source=source)
 base=dict(p.config);variants={
  'default128':{}, 'allnone':dict(tma_l2_promotion='none'),
  'all64':dict(tma_l2_promotion='64B'), 'all256':dict(tma_l2_promotion='256B'),
  'weightnone':dict(tma_weight_l2='none'), 'weight256':dict(tma_weight_l2='256B'),
 }
 if args.ring:variants.update(ringnone=dict(tma_ring_l2='none'),ring256=dict(tma_ring_l2='256B'),weight256_ringnone=dict(tma_weight_l2='256B',tma_ring_l2='none'))
 gs={};rec={}
 for name,options in variants.items():
  p.config=base|options;p.bind(a['dl'],a['dr'],a['dg'],a['dy'])
  p();torch.cuda.synchronize();es=errors(p.outputs,ref)
  assert all(e['finite'] and e['relative_l2']<=LIMITS[k] for k,e in es.items()),(name,es)
  gs[name]=capture(p);rec[name]=dict(options=options,errors=es)
 blocks=[]
 keys=list(gs)
 for j in range(3):
  order=keys[j:]+keys[:j]
  times=paired({k:gs[k] for k in order})
  blocks.append(dict(order=order,times=times))
  print('BLOCK',j,{k:v['median_us'] for k,v in times.items()},flush=True)
 for key in keys:
  values=sorted(x for b in blocks for x in b['times'][key]['samples_us'])
  rec[key]['median_us']=statistics.median(values)
 record=dict(source=source,L=args.length,ring=args.ring,shared_buffers=True,shared_cubin=True,variants=rec,blocks=blocks)
 (R/f'{source}-tma-promotion-L{args.length}.json').write_text(json.dumps(record,indent=2))
 print('RESULT',{k:v['median_us'] for k,v in rec.items()},flush=True)
