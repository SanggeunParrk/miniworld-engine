from ring_plan import *
class PairCTAPlan(RingPlan):
 def __init__(self,a,count=132,splits=20,part=2,source='front_ring_paircta_u8'):
  self.a=a;d=a['d'];n=d['n'];m=n*n
  assert m%128==0 and count>4*splits and count<=torch.cuda.get_device_properties(0).multi_processor_count
  self.count=count;self.splits=splits;self.part=part;self.source=source;self.config={'direct_weights':True};self.debug=0
  self.config.update(json.loads((R/(source+'.launch.json')).read_text()));self.threads=512;self.shared=self.config['shared']
  self.k,self.reduce=load(count,splits,part,0,source)
  self.dx=torch.empty((m,128),device=d['x'].device,dtype=torch.bfloat16);self.dw=torch.empty((4,128,256),device=self.dx.device,dtype=self.dx.dtype)
  self.dgam=torch.empty(128,device=self.dx.device);self.dbeta=torch.empty_like(self.dgam)
  self.partw=torch.empty((8,splits*2,16384),device=self.dx.device);self.partln=torch.empty(((count-4*splits)*2,256),device=self.dx.device)
  self.counts=torch.zeros(2+self.config['extra_counts'],dtype=torch.int32,device=self.dx.device);self.debugdc=self.dx;self.debugxn=self.dx
  self.outputs=(self.dx,*self.dw.unbind(0),self.dgam,self.dbeta);self.bind(a['dl'],a['dr'],a['dg'],a['dy'])
if __name__=='__main__':
 import argparse,faulthandler
 faulthandler.dump_traceback_later(30,repeat=True)
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='front_ring_paircta_u8');ap.add_argument('--length',type=int,default=64);ap.add_argument('--splits',type=int,default=20);ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref=baseline(a);p=PairCTAPlan(a,source=args.source,splits=args.splits);print('LAUNCH',flush=True);p();torch.cuda.synchronize();es=errors(p.outputs,ref);print('ERRORS',es,flush=True)
  assert all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items()),es
  assert torch.equal(p.counts,torch.zeros_like(p.counts))
  if args.bench:
   ts=paired({'baseline':capture(lambda:baseline(a)),'paircta':capture(p)});(R/f'{args.source}-L{args.length}.json').write_text(json.dumps(dict(errors=es,times=ts),indent=2));print('TIMES',{k:v['median_us'] for k,v in ts.items()},flush=True)
 faulthandler.cancel_dump_traceback_later()
