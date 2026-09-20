from front_plan import *
class WarpPlan(Plan):
 def __init__(self,a,count=264,splits=13,part=2,source='front_prefetch_lnpair_storepipe'):
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
  tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
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
