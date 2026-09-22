from front_plan import *
from cluster_driver import active_clusters
from check_front import LIMITS
class ClusterPlan(Plan):
 def __init__(self,a,count=120,source='front_cluster'):
  super().__init__(a,count=count,splits=count//8,source=source)
  self.threads=self.config.get("threads",256)
  self.active_clusters=active_clusters(self.k,count,196608,self.threads)
  assert count<=8*self.active_clusters,(count,self.active_clusters)
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;packed=L._Packed([self.p]);stream=int(torch.cuda.current_stream().cuda_stream)
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.count,1,1,self.threads,1,1,196608,drv.d.CUstream(stream),ctypes.addressof(packed.array)))
  return self.outputs
if __name__=='__main__':
 import argparse,faulthandler
 faulthandler.dump_traceback_later(25,repeat=True)
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--count',type=int,default=120);ap.add_argument('--source',default='front_cluster');ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref=baseline(a);p=ClusterPlan(a,args.count,args.source);print('LAUNCH',p.active_clusters,flush=True);p();torch.cuda.synchronize();es=errors(p.outputs,ref);print('CLUSTER_ERRORS',es,'COUNTS',p.counts.tolist(),flush=True);assert all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in es.items()),es
  if args.bench:
   ts=paired({'baseline':capture(lambda:baseline(a)),'cluster':capture(p)});(R/f'{args.source}-L{args.length}-c{args.count}.json').write_text(json.dumps(dict(errors=es,times=ts),indent=2));print('TIMES',{k:v['median_us'] for k,v in ts.items()},flush=True)
 faulthandler.cancel_dump_traceback_later()
