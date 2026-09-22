from front_plan import *
class WarpPlan(Plan):
 def __init__(self,a,count=132,splits=8,part=2,source='front_warp'):
  # setmaxnreg may redistribute only the initial CTA register quota.
  # A legal per-SM total can still deadlock if it exceeds 384*168.
  reg=re.search(r'_p(\d+)c(\d+)',source)
  if reg and int(reg[1])+2*int(reg[2])>504:raise ValueError('Dynamic register allocation exceeds the 168*384 CTA quota')
  self.a=a;d=a['d'];n=d['n'];m=n*n
  assert m%64==0 and count>8*splits
  self.count=count;self.splits=splits;self.part=part;self.source=source;self.config={'direct_weights':True};self.debug=0
  cf=R/(source+'.launch.json')
  if cf.exists():self.config.update(json.loads(cf.read_text()))
  self.threads=self.config.get('threads',384)
  self.shared=self.config.get('shared',229376)
  assert count<=self.config.get('max_ctas_per_sm',1)*torch.cuda.get_device_properties(0).multi_processor_count
  self.k,self.reduce=load(count,splits,part,0,source)
  self.dx=torch.empty((m,128),device=d['x'].device,dtype=torch.bfloat16);self.dw=torch.empty((4,128,256),device=self.dx.device,dtype=self.dx.dtype)
  self.dgam=torch.empty(128,device=self.dx.device);self.dbeta=torch.empty_like(self.dgam)
  self.partw=torch.empty((8,splits*self.config.get("wgrad_slices",1),16384),device=self.dx.device);self.partln=torch.empty((count-8*splits,256),device=self.dx.device)
  self.counts=torch.zeros(2+self.config.get("extra_counts",0)+(self.config.get("trace_words",4)*count if source.endswith("_trace") else 0),dtype=torch.int32,device=self.dx.device);self.debugdc=self.dx;self.debugxn=self.dx
  self.outputs=(self.dx,*self.dw.unbind(0),self.dgam,self.dbeta);self.bind(a['dl'],a['dr'],a['dg'],a['dy'])
 def bind(self,dl,dr,dg,dy):
  a=self.a;d=a['d'];n=d['n'];m=n*n;L=T._launch_module()
  def tm(t,box,dims,strides):
   policy=self.config.get('tma_l2_promotion','128B')
   if any(t is a[k] for k in ('wlg','wl','wrg','wr','wg')):policy=self.config.get('tma_weight_l2',policy)
   if hasattr(self,'ring') and t is self.ring:policy=self.config.get('tma_ring_l2',policy)
   return L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2=policy)
  row=lambda t:tm(t,[64,64],[128,m],[256])
  maps=[tm(dl,[64,64],[m,256],[m*2]),tm(dr,[64,64],[m,256],[m*2]),tm(a['pre'],[64,128],[m,1024],[m*2]),row(a['xn']),row(dg),*[tm(a[k],[64,self.config.get('weight_tma_rows',64)],[256,128],[512]) for k in ['wlg','wl','wrg','wr']],tm(a['wg'],[64,64],[128,128],[256]),row(d['x']),row(dy),row(self.dx)]
  self.inputs=(dl,dr,dg,dy);self.p=L.Struct([*maps,a['mask'],a['mu'],a['rs'],d['gi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.debugdc,self.debugxn,m,n,m//64])
 def __call__(self):
  if self.part==2:
   L=T._launch_module();drv=self.k.unit.drv;packed=L._Packed([self.p]);stream=int(torch.cuda.current_stream().cuda_stream)
   drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.count,1,1,self.threads,1,1,self.shared,drv.d.CUstream(stream),ctypes.addressof(packed.array)))
  else:
   self.k.launch((self.count,1,1),(self.threads,1,1),[self.p],self.shared);self.reduce.launch(((131328+255)//256,1,1),(256,1,1),[self.p],0)
  return self.outputs
if __name__=='__main__':
 import argparse,faulthandler
 faulthandler.dump_traceback_later(45,repeat=True)
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--splits',type=int,default=8);ap.add_argument('--part',type=int,default=2);ap.add_argument('--count',type=int,default=132);ap.add_argument('--source',default='front_warp');ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref=baseline(a);p=WarpPlan(a,count=args.count,splits=args.splits,part=args.part,source=args.source);print('LAUNCH',flush=True)
  p();torch.cuda.synchronize();e=errors(p.outputs,ref);print('ERRORS',e,flush=True);print('COUNTERS',p.counts.tolist(),flush=True)
  if args.bench:
   from check_front import LIMITS
   assert all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in e.items()),e
   gs={'baseline':capture(lambda:baseline(a)),'cuda':capture(p)};t=paired(gs);(R/f'{args.source}-L{args.length}-s{args.splits}.json').write_text(json.dumps(dict(L=args.length,errors=e,times=t),indent=2));print('TIMES',{k:v['median_us'] for k,v in t.items()},flush=True)
 faulthandler.cancel_dump_traceback_later()
