from ring_plan import *
from cluster_driver import active_clusters
class MulticastMixin:
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw)
  size=self.config['cluster_size']
  assert size in (2,4,8)
  assert self.count%size==0 and self.a['d']['n']**2//64%size==0
  self.cluster_capacity=active_clusters(self.k,self.count,self.shared,self.threads)
  assert self.count<=size*self.cluster_capacity,(self.count,self.cluster_capacity,size)
class MulticastRingPlan(MulticastMixin,RingPlan):pass
class MulticastWarpPlan(MulticastMixin,WarpPlan):pass
if __name__=='__main__':
 import argparse,faulthandler
 faulthandler.dump_traceback_later(30,repeat=True)
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--length',type=int,default=64);ap.add_argument('--count',type=int,default=264);ap.add_argument('--splits',type=int,default=20);ap.add_argument('--ring',action='store_true');ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref=baseline(a);cls=MulticastRingPlan if args.ring else MulticastWarpPlan
  p=cls(a,count=args.count,splits=args.splits,source=args.source);print('CAPACITY',p.cluster_capacity,flush=True)
  p();torch.cuda.synchronize();es=errors(p.outputs,ref);print('ERRORS',es,flush=True)
  assert all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items()),es
  assert torch.equal(p.counts,torch.zeros_like(p.counts))
  if args.bench:
   ts=paired({'baseline':capture(lambda:baseline(a)),'multicast':capture(p)})
   (R/f'{args.source}-L{args.length}-c{args.count}.json').write_text(json.dumps(dict(errors=es,times=ts),indent=2));print('TIMES',{k:v['median_us'] for k,v in ts.items()},flush=True)
 faulthandler.cancel_dump_traceback_later()
