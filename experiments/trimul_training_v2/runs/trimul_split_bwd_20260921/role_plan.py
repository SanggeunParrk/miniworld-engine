from pathlib import Path
from functools import lru_cache
import ctypes,fcntl,hashlib,json,re,subprocess,sys,torch
R=Path(__file__).resolve().parent;PREV=R.parent/'trimul_recompute_next_20260921';P7=R.parent/'anthropic_b7b12_fusion_20260920'
sys.path.insert(0,str(PREV));import plans as P
T=P.T
@lru_cache(None)
def build(role,saved,splits,pc=2,dxctas=132,prod=40,cons=232,resident=1,slices=2):
 if role==1 and 16*splits>132*(2 if pc==1 else 1):raise ValueError('Cooperative grid exceeds the measured H100 residency limit')
 count=132 if role==0 else 16*splits if role==1 else dxctas
 threads=256 if role==2 else 128*(pc+1)
 smem=(pc*49152+16384) if role==1 else (221184 if resident else 131072)
 defs=dict(SPLIT_ROLE=role,USE_SAVED_XN=int(saved),DW_SPLITS=splits,PIPE_CONSUMERS=pc,DX_CTAS=dxctas,DW_PROD_REGS=prod,DW_CONS_REGS=cons,UCOUNT=count,DX_RESIDENT=resident,WGRAD_SLICES=slices)
 inc=T._upstream()/'csrc';src=R/'b7_roles.cu'
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-I'+str(P7),'-I'+str(R)]+['-D%s=%s'%v for v in defs.items()]
 deps=[src,*sorted(R.glob('*.cuh')),R/'b7_pipe_dw.inc',R/'b7_packed_dx.inc',*[P7/p for p in ('warp_primitives.cuh','front_mn_primitives.cuh','front_primitives.cuh')],*[inc/p for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh')]]
 key=hashlib.sha256(b''.join(p.read_bytes() for p in deps)+str(flags).encode()).hexdigest();out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
 with out.with_suffix('.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not out.exists():
   tmp=out.with_suffix('.tmp.cubin');p=subprocess.run(['nvcc',*flags,str(src),'-o',str(tmp)],capture_output=True,text=True);out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError(p.stderr)
   tmp.replace(out);out.with_suffix('.json').write_text(json.dumps(dict(defines=defs,flags=flags,threads=threads,smem=smem),indent=2))
 log=out.with_suffix('.ptxas.log').read_text()
 if re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)',log):raise RuntimeError('spill '+str(out))
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());unit=L.Unit('b7_roles','sm_90a',0,drv.drv,mod,{},str(out));k=unit.kernel('front_b7b12');k.set_max_dynamic_smem(smem)
 print('CUBIN',str(out),defs,[s.strip() for s in log.splitlines() if 'spill' in s or 'Used ' in s],flush=True)
 return k,count,threads,smem

class Plan:
 def __init__(self,d,dy,dl,dr,dg,xn=None,split=False,splits=4,pc=2,dxctas=132,prod=40,cons=232,resident=1,slices=2):
  self.d=d;self.split=split;self.xn=xn;self.cfg=dict(saved=xn is not None,splits=splits,pc=pc,dxctas=dxctas,prod=prod,cons=cons,resident=resident,slices=slices)
  self.units=[build(role=role,**self.cfg) for role in ((1,2) if split else (0,))]
  m=d['n']**2;x=d['x'];self.dx=torch.empty((m,128),device=x.device,dtype=x.dtype);self.dw=torch.empty((4,128,256),device=x.device,dtype=x.dtype)
  self.dgam=torch.empty(128,device=x.device);self.dbeta=torch.empty_like(self.dgam)
  self.partw=torch.zeros((16,splits,pc,slices,8192),device=x.device);self.partln=torch.empty((dxctas if split else 132-16*splits,256),device=x.device)
  self.counts=[torch.zeros(2,device=x.device,dtype=torch.int32) for _ in self.units];self.mask=d['mask'].bfloat16().reshape(-1);self.outputs=(self.dx,*self.dw.unbind(),self.dgam,self.dbeta);self.bind(dl,dr,dg,dy,xn)
 def bind(self,dl,dr,dg,dy,xn=None):
  if xn is not None:self.xn=xn
  d=self.d;n=d['n'];m=n*n;L=T._launch_module();tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B');row=lambda t:tm(t,[64,64],[128,m],[256])
  wl,wlg,wr,wrg,wg=d['wt'];self.weights=(wl,wlg,wr,wrg,wg)
  maps=[tm(dl,[64,32],[m,256],[m*2]),tm(dr,[64,32],[m,256],[m*2]),tm(d['w1'],[64,64],[128,1024],[256]),row(self.xn if self.xn is not None else d['x']),row(dg),*[tm(w,[64,64],[256,128],[512]) for w in (wlg,wl,wrg,wr)],tm(wg,[64,64],[128,128],[256]),row(d['x']),row(dy),row(self.dx)]
  self.params=[L.Struct([*maps,self.mask,None,None,d['gi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,count,self.dx,self.dx,m,n,m//64,d['bi']]) for count in self.counts]
  self.inputs=(dl,dr,dg,dy,d['w1'],self.xn,*self.weights)
 def __call__(self):
  L=T._launch_module();stream=int(torch.cuda.current_stream().cuda_stream)
  for (k,count,threads,smem),p in zip(self.units,self.params):
   drv=k.unit.drv;args=L._Packed([p]);drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(k.handle)),count,1,1,threads,1,1,smem,drv.d.CUstream(stream),ctypes.addressof(args.array)))
  return self.outputs
