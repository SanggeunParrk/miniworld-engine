from core import *
@lru_cache(None)
def ukernel(count,part=0):
 inc=T._upstream()/'csrc';source=R/'dual.cu';flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-DROLE=0',f'-DUCOUNT={count}',f'-DPART_ONLY={part}']
 key=hashlib.sha256(source.read_bytes()+(R/'dual_primitives.cuh').read_bytes()+b''.join((inc/p).read_bytes() for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))+str(flags).encode()).hexdigest();path=R/'build'/(key+'.cubin')
 if not path.exists():
  z=subprocess.run(['nvcc',*flags,str(source),'-o',str(path)],capture_output=True,text=True);path.with_suffix('.ptxas.log').write_text(z.stdout+z.stderr)
  if z.returncode:raise RuntimeError(z.stderr)
  path.with_suffix('.json').write_text(json.dumps(dict(unified=True,count=count,flags=flags)))
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(path.read_bytes());unit=L.Unit('unified','sm_90a',0,drv.drv,mod,{},str(path));k=unit.kernel('dual_b1b4');k.set_max_dynamic_smem(231424);print('CUBIN',str(path),flush=True);return k,unit.kernel('unified_reduce')
class Unified:
 """Experimental BF16 C128/H256 B1-B4 plan; one stream and one invocation at a time."""
 def __init__(self,d,dy,s,count,part=2):
  assert part in (1,2) and 0<count
  assert torch.cuda.get_device_capability(0)==(9,0)
  assert count<=d['n']**2//64
  self._init_workspace(d,dy,s,count)
  self.k,self.reduce=ukernel(count,part);self.part=part;self.wgroups=2;self.smem=231424;self.grid=count

 def __call__(self):
  if self.part==2:
   import ctypes
   assert self.grid<=torch.cuda.get_device_properties(0).multi_processor_count
   L=T._launch_module();dr=self.k.unit.drv;d=dr.d;packed=L._Packed([self.p]);st=int(torch.cuda.current_stream().cuda_stream)
   dr._unwrap('cuLaunchCooperativeKernel',d.cuLaunchCooperativeKernel(d.CUfunction(int(self.k.handle)),self.grid,1,1,256,1,1,self.smem,d.CUstream(st),ctypes.addressof(packed.array)))
  else:self.k.launch((self.grid,1,1),(256,1,1),[self.p],self.smem)
  if self.part==1:self.reduce.launch((194,1,1),(256,1,1),[self.p],0)
  return self.outputs

 def _init_workspace(self,d,dy,s,count):
  ctx,_,_=s
  (xn,wl,wlg,wr,wrg,wg,wp,go,pre,lf,rf,tri,norm,mean,rs,gate,proj)=ctx.saved_tensors
  n=d['n'];m=n*n;assert n>=64 and m%64==0 and xn.shape[-1]==128 and tri.shape[0]==256
  self.n=n;self.splits=count;tiles=m//64;groups=(tiles+32-1)//32
  self.d=d;self.dy=dy;self.saved=s;self.wpt=wp.t().contiguous()
  dg=torch.empty((m,128),device='cuda',dtype=torch.bfloat16);dt=torch.empty_like(tri);dwg=torch.empty((128,128),device='cuda',dtype=dg.dtype);dwp=torch.empty((128,256),device='cuda',dtype=dg.dtype);dgamma=torch.empty(256,device='cuda');dbeta=torch.empty_like(dgamma)
  partw=torch.empty((12,count,4096),device='cuda');partln=torch.empty((count,512),device='cuda');group_ln=torch.empty((1,512),device='cuda');counts=torch.zeros(2,device='cuda',dtype=torch.int32)
  self.workspace=(partw,partln,group_ln,counts);self.outputs=(dg,dwg,dt,dgamma,dbeta,dwp)
  L=T._launch_module();tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
  row=lambda t,c:tm(t,[64,64],[c,m],[c*2])
  maps=[row(dy,128),row(gate,128),row(proj,128),row(xn,128),row(norm,256),tm(tri,[64,256],[m,256],[m*2]),tm(self.wpt,[64,64],[128,256],[256]),tm(dt,[64,16,1],[m,256,1],[m*2,m*512])]
  self.p=L.Struct([*maps,d['ds'],mean,rs,go,dg,dwg,dwp,dgamma,dbeta,partw,partln,group_ln,counts,m,n,tiles,groups]);self.grid=tiles+12*count
