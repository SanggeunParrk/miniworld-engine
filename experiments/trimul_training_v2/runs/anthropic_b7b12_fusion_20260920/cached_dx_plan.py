from front_plan import *
from check_front import LIMITS
class CachedDxPlan(Plan):
 def __init__(self,a,count=132,source='front_cached_dx',debug=0):
  super().__init__(a,count=count,splits=1,source=source,debug=debug)
  self.partln=torch.empty((count,256),device=self.dx.device)
  if debug:self.partw=torch.empty_like(self.dx,dtype=torch.float32)
  self.bind(a['dl'],a['dr'],a['dg'],a['dy'])
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;packed=L._Packed([self.p]);stream=int(torch.cuda.current_stream().cuda_stream)
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.count,1,1,512,1,1,229376,drv.d.CUstream(stream),ctypes.addressof(packed.array)))
  return self.outputs
if __name__=='__main__':
 import argparse,faulthandler
 faulthandler.dump_traceback_later(30,repeat=True)
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--count',type=int,default=80);ap.add_argument('--source',default='front_cached_dx');ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref=baseline(a);p=CachedDxPlan(a,args.count,args.source);print('LAUNCH',flush=True);p();torch.cuda.synchronize();es=errors(p.outputs,ref);es={k:v for k,v in es.items() if k in ('dx','dgamma','dbeta')};print('DX_ERRORS',es,'COUNTS',p.counts.tolist(),flush=True)
  assert all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in es.items()),es
  if args.bench:
   ts=paired({'dx_only':capture(p)});(R/f'{args.source}-L{args.length}-c{args.count}.json').write_text(json.dumps(dict(errors=es,times=ts,scope='dX only, no dW'),indent=2));print('DX_ONLY_US',ts['dx_only']['median_us'],flush=True)
 faulthandler.cancel_dump_traceback_later()
