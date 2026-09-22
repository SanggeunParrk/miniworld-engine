"""Vary ring row pitch with one cubin and identical buffer base addresses."""
from ring_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=768)
ap.add_argument('--source',default='front_ring96_cache3')
ap.add_argument('--pads',type=int,nargs='+',default=[0,64,128,256,512,1024,2048,4096])
args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);p=RingPlan(a,count=264,splits=20,source=args.source)
 width=p.config['ring_tiles']*64
 backing=torch.empty((1024,width+max(args.pads)),device='cuda',dtype=torch.bfloat16)
 records={};gs={'baseline':capture(lambda:baseline(a))}
 for pad in args.pads:
  assert pad>=0 and pad%64==0
  p.ring=backing.as_strided((1024,width),(width+pad,1))
  p.bind(a['dl'],a['dr'],a['dg'],a['dy']);p();torch.cuda.synchronize()
  es=errors(p.outputs,ref);valid=all(e['finite'] and e['relative_l2']<=LIMITS[k] for k,e in es.items()) and not bool(p.counts.any())
  key='pad'+str(pad);records[key]=dict(pad_elements=pad,valid=valid,errors=es)
  if valid:gs[key]=capture(p)
  print('CHECK',pad,valid,flush=True)
 keys=list(gs);blocks=[]
 for i in range(3):
  order=keys[i:]+keys[:i];ts=paired({k:gs[k] for k in order});blocks.append(ts)
  print('BLOCK',i,{k:v['median_us'] for k,v in ts.items()},flush=True)
 times={}
 for key in keys:
  vals=sorted(v for b in blocks for v in b[key]['samples_us'])
  times[key]=dict(median_us=statistics.median(vals),samples_us=vals)
 out=dict(L=args.length,source=args.source,shared_addresses=True,records=records,times=times,blocks=blocks)
 (R/f'{args.source}-ring-pitch-L{args.length}.json').write_text(json.dumps(out,indent=2))
 print('RESULT',{k:v['median_us'] for k,v in times.items()},flush=True)
