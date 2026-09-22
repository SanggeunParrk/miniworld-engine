from pathlib import Path
import ctypes,hashlib,json,subprocess,sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_recompute_next_20260921'));import plans as P
T=P.T
class Plan:
 def __init__(self,d,dy,dl,dr,dg,xn,clusters=16):
  self.d=d;self.xn=xn;self.clusters=clusters;self.count=clusters*8
  inc=T._upstream()/'csrc';flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-I'+str(R.parent/'anthropic_b7b12_fusion_20260920'),'-I'+str(R.parent/'trimul_b7_nextrow_20260921')]
  self.source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (R/'joint.cu',R/'single_wg.inc')};key=hashlib.sha256((json.dumps(self.source_sha256)+str(flags)).encode()).hexdigest();out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
  if not out.exists():
   p=subprocess.run(['nvcc',*flags,str(R/'joint.cu'),'-o',str(out)],capture_output=True,text=True);out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError(p.stderr)
  self.compiler_log=out.with_suffix('.ptxas.log').read_text();print('CUBIN',str(out),self.compiler_log[-1500:],flush=True)
  L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());unit=L.Unit('joint','sm_90a',0,drv.drv,mod,{},str(out));self.k=unit.kernel('b7_joint');self.k.set_max_dynamic_smem(180224)
  try:
   cfg=drv.drv.d.CUlaunchConfig();cfg.gridDimX=self.count;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=384;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=180224
   result=drv.drv._unwrap('cuOccupancyMaxActiveClusters',drv.drv.d.cuOccupancyMaxActiveClusters(drv.drv.d.CUfunction(int(self.k.handle)),cfg));print('ACTIVE_CLUSTERS',result,flush=True)
   if isinstance(result,tuple):result=result[0]
   self.clusters=min(clusters,int(result));self.count=self.clusters*8
  except AttributeError as e:print('OCCUPANCY_API_UNAVAILABLE',str(e),flush=True)
  assert self.clusters>0
  x=d['x'];m=d['n']**2;self.dx=torch.empty((m,128),device=x.device,dtype=x.dtype);self.dw=torch.empty((4,128,256),device=x.device,dtype=x.dtype);self.dgam=torch.empty(128,device=x.device);self.dbeta=torch.empty_like(self.dgam)
  self.partw=torch.zeros((self.clusters,8,1,2,64,128),device=x.device);self.partln=torch.empty((self.clusters,256),device=x.device);self.counts=torch.zeros(2,device=x.device,dtype=torch.int32);self.mask=d['mask'].bfloat16().reshape(-1);self.outputs=(self.dx,*self.dw.unbind(),self.dgam,self.dbeta);self.bind(dl,dr,dg,dy,xn)
 def bind(self,dl,dr,dg,dy,xn=None):
  if xn is not None:self.xn=xn
  d=self.d;n=d['n'];m=n*n;L=T._launch_module();tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='256B');row=lambda t:tm(t,[64,64],[128,m],[256]);wg=d['wt'][4]
  self.params=L.Struct([row(self.xn),tm(d['w1'],[64,64],[128,1024],[256]),tm(dl,[64,32],[m,256],[m*2]),tm(dr,[64,32],[m,256],[m*2]),row(dg),tm(wg,[64,64],[128,128],[256]),row(d['x']),row(dy),row(self.dx),self.mask,d['gi'],d['bi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.dx.numel()//128,m//64]);self.inputs=(dl,dr,dg,dy,d['w1'],self.xn,wg)
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params]);drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.count,1,1,384,1,1,180224,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)));return self.outputs
