"""Query cluster residency before launching strict small-shape checks."""
from multicast_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--ring',action='store_true')
args=ap.parse_args()
with torch.no_grad():
 a=setup(64);ref=baseline(a)
 base='front_ring96_cache3' if args.ring else 'front_prefetch_lnpair_storepipe'
 cls=MulticastRingPlan if args.ring else MulticastWarpPlan
 records={}
 for size in (4,8):
  source=base+'_mcastasync_xn'+str(size)
  splits=20 if args.ring else 13
  k,_=load(240,splits,2,0,source)
  config=json.loads((R/(source+'.launch.json')).read_text())
  cap=active_clusters(k,240,config['shared'],config['threads'])
  count=min(size*cap,264//size*size)
  print('CAPACITY',source,cap,count,flush=True)
  p=cls(a,count=count,splits=splits,source=source)
  p();torch.cuda.synchronize();e=errors(p.outputs,ref)
  valid=all(v['finite'] and v['relative_l2']<=LIMITS[name] for name,v in e.items())
  assert valid,e
  assert not bool(p.counts.any())
  records[source]=dict(count=count,capacity=cap,splits=splits,errors=e)
  print('CHECK',source,valid,flush=True)
 (R/('multicast-wide-ring-probe.json' if args.ring else 'multicast-wide-direct-probe.json')).write_text(json.dumps(records,indent=2))
