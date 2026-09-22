"""Identical-source controls: distinguish candidate gains from workspace effects."""
from ring_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--source',default='front_ring112_ctasync_consume');args=ap.parse_args()
with torch.no_grad():
 a=setup(768);ref=baseline(a);ps={};gs={};meta={}
 for k in ['first','second','third']:
  p=RingPlan(a,count=264,splits=20,source=args.source);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
  assert all(v['finite'] and v['relative_l2']<=LIMITS[key] for key,v in es.items()),es
  ps[k]=p;gs[k]=capture(p);meta[k]=dict(errors=es,ring_offset_mod32MiB=p.ring.data_ptr()%(32*1024*1024),dx_offset_mod32MiB=p.dx.data_ptr()%(32*1024*1024))
 records=[]
 for order in [['first','second','third'],['third','first','second'],['second','third','first']]:
  ts=paired({k:gs[k] for k in order});records.append(dict(order=order,times=ts));print('IDENTICAL',order,{k:v['median_us'] for k,v in ts.items()},flush=True)
 (R/f'{args.source}-allocation-controls.json').write_text(json.dumps(dict(source=args.source,metadata=meta,records=records),indent=2))
