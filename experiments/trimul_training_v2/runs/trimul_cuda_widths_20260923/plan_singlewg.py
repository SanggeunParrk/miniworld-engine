"""Width-port of recomputing CUDA training: explicit TMA/WGMMA, all eleven grads.
Generic widths use cooperative phase grids and transient GEMM workspace. The
original D128 tuned implementation remains selected separately.
"""
from pathlib import Path
import torch,json,hashlib,subprocess,ctypes,sys,os
from functools import lru_cache
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_widths_20260922'))
from native import T,Front
from selected import pack_into,normalize_into
@lru_cache(None)
def build(D):
 inc=T._upstream()/'csrc';src=R/'widths.cu';flags=['-DPROFILE_STAGE='+os.environ.get('WIDTH_PROFILE','0'),'-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',f'-DWIDTH={D}',f'-DWEIGHT_SPLITS={128 if D==64 else 32}','-I'+str(inc)]
 key=hashlib.sha256(src.read_bytes()+str(flags).encode()+b''.join((inc/x).read_bytes() for x in ['tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'])).hexdigest();out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
 if not out.exists():
  z=subprocess.run(['nvcc',*flags,str(src),'-o',str(out)],capture_output=True,text=True);out.with_suffix('.log').write_text(z.stdout+z.stderr)
  if z.returncode:raise RuntimeError(z.stderr)
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());u=L.Unit('width_training','sm_90a',0,drv.drv,mod,{},str(out));ks={n:u.kernel('width_'+n) for n in ('forward','b1','b7','probe')}
 for k in ks.values():k.set_max_dynamic_smem(33024)
 return ks,str(out)
def launch(k,params,grid=132,cooperative=True):
 L=T._launch_module()
 if not cooperative:k.launch((grid,1,1),(128,1,1),[params],33024);return
 drv=k.unit.drv;args=L._Packed([params]);drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(k.handle)),grid,1,1,128,1,1,33024,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
def tm(t):
 assert t.ndim==2 and t.is_contiguous() and t.dtype==torch.bfloat16
 return T._launch_module().tensor_map(t,[64,64],dims=[t.shape[1],t.shape[0]],strides_bytes=[t.shape[1]*2],swizzle='128B',l2='128B')
class Training:
 def __init__(self,x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask,ds,dy):
  assert x.dtype==torch.bfloat16 and x.shape[0]==1 and x.shape[1]==x.shape[2]
  n=x.shape[1];D=x.shape[-1];H=2*D;M=n*n;assert D in (64,128,256,384,512) and n in (384,768)
  self.D,self.n,self.M=D,n,M;self.weights=(wl,wlg,wr,wrg);self.x,self.dy,self.ds=x,dy,ds;self.mask=mask.float().contiguous();self.ks,self.path=build(D);self.grid=132*min(4,min(int(k.unit.drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',k.unit.drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(k.unit.drv.d.CUfunction(int(k.handle)),128,33024))) for k in (self.ks['forward'],self.ks['b1'],self.ks['b7'])))
  self.w1=x.new_empty((8*D,D));pack_into(self.w1,*self.weights)
  if D==128:cfg=dict(k1=[2,64,8,2,1],input_ln='fused')
  else:cfg=json.loads((R.parent/'trimul_widths_20260922/selection.json').read_text())[f'{D}-{n}']
  self.separate=cfg['input_ln']=='separate';self.gi,self.bi=gi,bi
  self.xn=torch.empty_like(x) if self.separate else None
  if self.separate:normalize_into(self.xn,x,gi,bi)
  self.front=Front((self.xn if self.separate else x)[0],self.w1,self.mask,gi,bi,cfg['k1'],emit_xn=not self.separate,normalize=not self.separate)
  if not self.separate:self.xn=self.front.xn
  self.tri=x.new_empty((H,n,n));self.y=torch.empty_like(x);self.dx=torch.empty_like(x);self.dg=x.new_empty((M,D));self.dt=torch.empty_like(self.tri)
  self.gp=[x.new_empty((M,H)) for _ in range(4)];self.dw=[torch.empty_like(w) for w in self.weights];self.dwp=torch.empty_like(wp);self.dwg=torch.empty_like(wg)
  norm=x.new_empty((M,H));dp=x.new_empty((M,D));dn=x.new_empty((M,H));dxn=x.new_empty((M,D));self.dl=torch.empty_like(self.tri);self.dr=torch.empty_like(self.tri)
  self.tensors=[x,self.tri,dy,ds,self.y,self.xn,norm,dp,self.dg,dn,dxn,self.dx,self.dt,*self.gp,*self.dw,self.dwp,self.dwg,None]
  self.floats=[gi,bi,go,bo,self.mask,*[torch.empty(M,device=x.device) for _ in range(2)],torch.empty((128 if D==64 else 32,11*D*D),device=x.device),*[torch.empty(c,device=x.device) for c in (D,D,H,H)],torch.empty(32,device=x.device,dtype=torch.int64)]
  maps=[tm(self.xn.reshape(M,D)),tm(wp),tm(wg),tm(norm),tm(dp),tm(self.dg),*[tm(g) for g in self.gp],*[tm(w) for w in self.weights],tm(dxn),tm(dn)]
  L=T._launch_module();self.params=L.Struct([*maps,*self.tensors,*self.floats,M,n]);t7=self.tensors.copy();t7[22:24]=[self.dl,self.dr];maps7=maps.copy();maps7[14:16]=[tm(self.dl.reshape(H,M)),tm(self.dr.reshape(H,M))];self.params7=L.Struct([*maps7,*t7,*self.floats,M,n]);self.maps=maps
  self.outputs=(self.dx,*self.dw,self.dwg,self.dwp,*self.floats[8:12])
 def forward(self):
  pack_into(self.w1,*self.weights)
  if self.separate:normalize_into(self.xn,self.x,self.gi,self.bi)
  ab,_=self.front();d=self.D;h=2*d
  torch.bmm(ab[:d],ab[h:h+d].transpose(-1,-2),out=self.tri[:d]);torch.bmm(ab[d:h].transpose(-1,-2),ab[h+d:],out=self.tri[d:])
  launch(self.ks['forward'],self.params,self.grid);return self.y
 def backward(self):
  launch(self.ks['b1'],self.params,self.grid);ab=self.front.ab;d=self.D;h=2*d
  torch.bmm(self.dt[:d],ab[h:h+d],out=self.dl[:d]);torch.bmm(self.dt[:d].transpose(-1,-2),ab[:d],out=self.dr[:d]);torch.bmm(ab[h+d:],self.dt[d:].transpose(-1,-2),out=self.dl[d:]);torch.bmm(ab[d:h],self.dt[d:],out=self.dr[d:])
  launch(self.ks['b7'],self.params7,self.grid);return self.outputs
 def __call__(self):return self.forward(),self.backward()
