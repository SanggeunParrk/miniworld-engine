from pathlib import Path
import os,ctypes,hashlib,json,subprocess,sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_recompute_next_20260921'));import plans as P
T=P.T
class Plan:
 def __init__(self,d,dy,dl,dr,dg,xn,clusters=10,mode=0):
  self.d=d;self.xn=xn;self.clusters=clusters;self.consumers=int(os.environ.get("B7_CONSUMERS","8"));self.reuse_dp=1;self.count=clusters*(16+self.consumers)
  inc=T._upstream()/'csrc';flags=['-DB7_TRANSPOSE_PART='+str(bool(mode&16)*1),'-DB7_REDUCE_NATIVE='+str(bool(mode&32)*1),'-DB7_PRODUCER_REGS='+os.environ.get('B7_PRODUCER_REGS','48'),'-DB7_RING_CHUNK='+str(8192 if mode&8 else 16384 if mode&4 else 32768),'-DB7_GATEFIRST='+str(bool(mode&2)*1),'-DB7_RING_TENSOR='+str(mode&1),'-DB7_ROLLING='+str(bool(mode&4)*1),'-DB7_LNPREFETCH='+str(bool(mode&8)*1),'-DB7_PREFETCH='+str(mode&3),'-DB7_CONSUMERS='+str(self.consumers),'-DB7_REUSE_DP='+str(self.reuse_dp),'-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-I'+str(R.parent/'anthropic_b7b12_fusion_20260920'),'-I'+str(R.parent/'trimul_b7_nextrow_20260921')]
  self.source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (R/'joint.cu',R/'single_wg.inc')};key=hashlib.sha256((json.dumps(self.source_sha256)+str(flags)).encode()).hexdigest();out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
  if not out.exists():
   p=subprocess.run(['nvcc',*flags,str(R/'joint.cu'),'-o',str(out)],capture_output=True,text=True);out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError(p.stderr)
  self.compiler_log=out.with_suffix('.ptxas.log').read_text();print('CUBIN',str(out),self.compiler_log[-1500:],flush=True)
  L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());unit=L.Unit('joint','sm_90a',0,drv.drv,mod,{},str(out));self.k=unit.kernel('b7_joint');self.k.set_max_dynamic_smem(114688)
  assert self.clusters>0
  x=d['x'];m=d['n']**2;self.dx=torch.empty((m,128),device=x.device,dtype=x.dtype);self.dw=torch.empty((4,128,256),device=x.device,dtype=x.dtype);self.dgam=torch.empty(128,device=x.device);self.dbeta=torch.empty_like(self.dgam)
  self.partw=torch.zeros((self.clusters*2,16,64,128),device=x.device);self.partln=torch.empty((self.clusters*self.consumers,256),device=x.device);self.counts=torch.zeros(2,device=x.device,dtype=torch.int32);self.mask=d['mask'].bfloat16().reshape(-1);self.ring=torch.empty((self.clusters,8,131072),device=x.device,dtype=torch.uint8);self.xring=torch.empty((self.clusters,8,16384),device=x.device,dtype=torch.uint8);self.flags=torch.zeros((self.clusters,8,18),device=x.device,dtype=torch.int32);self.outputs=(self.dx,*self.dw.unbind(),self.dgam,self.dbeta);self.bind(dl,dr,dg,dy,xn)
 def bind(self,dl,dr,dg,dy,xn=None):
  if xn is not None:self.xn=xn
  d=self.d;n=d['n'];m=n*n;L=T._launch_module();tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='256B');row=lambda t:tm(t,[64,64],[128,m],[256]);wg=d['wt'][4]
  self.params=L.Struct([row(self.xn),tm(d['w1'],[64,64],[128,1024],[256]),tm(dl,[64,32],[m,256],[m*2]),tm(dr,[64,32],[m,256],[m*2]),row(dg),tm(wg,[64,128],[128,128],[256]),row(d['x']),row(dy),row(self.dx),*[tm(w,[64,128],[256,128],[512]) for w in (d['wt'][1],d['wt'][0],d['wt'][3],d['wt'][2])],L.tensor_map(self.ring.view(torch.bfloat16),[256,8,2],dims=[256,64,4*self.clusters*8],strides_bytes=[512,32768],swizzle='none'),self.mask,d['gi'],d['bi'],self.dx,self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.dx.numel()//128,m//64,self.ring,self.xring,self.flags]);self.inputs=(dl,dr,dg,dy,d['w1'],self.xn,wg)
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params]);drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.count,1,1,256,1,1,114688,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)));return self.outputs
