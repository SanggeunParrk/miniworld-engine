from native import *
R=Path(__file__).resolve().parent
@lru_cache(None)
def build_gp(D,cfg):
 bi,bj,slots,sk,mb=cfg;inc=headers();src='#include "native_gp.cuh"\nusing Cfg=tmn::K1Cfg<%d,%d,false,%d,%d,%d,%d>;\nextern "C" __global__ __launch_bounds__(Cfg::NTHR,Cfg::MINB) void native_gp(__grid_constant__ const tmn::sm90::GPParams p){tmn::sm90::gp_body<Cfg,true,0,false,false>(p);}\n'%(D,2*D,bi,bj,slots,sk)
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-I'+str(R),'-DMW_MINB='+str(mb)]
 key=hashlib.sha256(src.encode()+str(flags).encode()+(R/'native_gp.cuh').read_bytes()+b''.join((inc/f).read_bytes() for f in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest();out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
 if not out.exists():
  p=out.with_suffix('.cu');p.write_text(src);z=subprocess.run(['nvcc',*flags,str(p),'-o',str(out)],capture_output=True,text=True);out.with_suffix('.log').write_text(z.stdout+z.stderr)
  if z.returncode:raise RuntimeError(z.stderr)
 U=T._launch_module();drv=U.BlockDriver(device=0);mod=drv.load(out.read_bytes());unit=U.Unit('native_gp','sm_90a',0,drv.drv,mod,{},str(out));k=unit.kernel('native_gp');smem=k1_smem(D,cfg);k.set_max_dynamic_smem(smem);return k,smem,str(out)
class GP:
 def __init__(self,m,cfg=None,parameters_only=False):
  n=m.n;D=m.D;cfg=tuple(cfg or m.front.cfg);bi,bj,slots,sk,mb=cfg;self.cfg=cfg
  if parameters_only:
   from width_plan import gp_smem
   self.k,self.smem,self.path=None,gp_smem(D,cfg),None
  else:self.k,self.smem,self.path=build_gp(D,cfg)
  U=T._launch_module();tm=lambda t,box,dims,strides:U.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
  mz=tm(m.xn,[64,bj,bi],[D,n,n],[D*2,n*D*2]);mw=tm(m.w1,[64,64],[D,8*D],[D*2]);ma=tm(m.front.ab,[64,1,32],[n,n,4*D],[n*2,n*n*2]);tj=n//bj;tiles=(n//bi)*tj
  self.params=U.Struct([mz,mw,ma,m.mask,m.gi,m.bi,m.front.ab,None,None,n,n,tj,tiles,1,n,1,1e-5,n*D,D,0,0,*([0]*8),tm(m.dl,[64,1,32],[n,n,2*D],[n*2,n*n*2]),tm(m.dr,[64,1,32],[n,n,2*D],[n*2,n*n*2]),tm(m.gp_all,[64,1,32],[n,n,8*D],[n*2,n*n*2])]);self.grid=min(tiles,torch.cuda.get_device_properties(0).multi_processor_count*mb);self.threads=128*(bi*bj//64+1)
 def __call__(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
