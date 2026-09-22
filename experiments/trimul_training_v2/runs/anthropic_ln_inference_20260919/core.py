from pathlib import Path
from functools import lru_cache
import hashlib,subprocess,json,torch,itertools
from miniworld_engine.kernels.trimul_inproj.cuda import anthropic_training as T
R=Path(__file__).resolve().parent

def k1_smem(cfg):
 bi,bj,slots,sk,sched,regs,offset=cfg;ng=bi*bj//64;mb=2 if ng==1 else 1
 size=bi*bj*256+slots*sk*8192+ng*8192+1024+((2+2*slots)*8+127)//128*128
 if slots<2*(2//sk) or size*mb>232448:raise ValueError('resources')
 if sched==0 and slots<3*(2//sk):raise ValueError('pipeline')
 return size

def default_regs(bi,bj):
 ng=bi*bj//64;nt=128*(ng+1);mb=2 if ng==1 else 1;lr=(65536//(nt*mb))//8*8
 return min(232,((lr*nt-128*40)//(128*ng))//8*8)

def k1_candidates():
 for (bi,bj),slot,sk in itertools.product(((1,64),(2,32),(2,64),(4,32),(1,128),(3,64),(6,32),(1,192),(4,64),(2,128)),(2,4,6,8),(1,2)):
  cfg=(bi,bj,slot,sk,-1,default_regs(bi,bj),2)
  try:k1_smem(cfg)
  except ValueError:continue
  yield cfg

def k3_smem(cfg):
 bi,bj,slot,acc,r240,serial=cfg
 # Original native config, no extra saved tensor stages.
 bmt=bi*bj;size=bmt*256*2+bmt*128*2+slot*256*64+8*2048+(2*128+2*256)*4+((2*2+2+2*slot)*8+127)//128*128
 if size>232448 or (bmt==64 and acc!=1):raise ValueError('resources')
 return size

def k3_candidates():
 for (bi,bj),slot,acc,r240,serial in itertools.product(((1,64),(2,64),(1,128)),(4,6,8),(1,2),(0,1),(0,1)):
  cfg=(bi,bj,slot,acc,r240,serial)
  try:k3_smem(cfg)
  except ValueError:continue
  yield cfg

@lru_cache(None)
def build(kind,fused,cfg):
 inc=T._upstream()/'csrc';source=R/(kind+'.cu')
 if kind=='k1':
  bi,bj,slot,sk,sched,regs,offset=cfg
  defs=dict(MW_BI=bi,MW_BJ=bj,MW_SLOT=slot,MW_SK=sk,MW_SCHED=sched,MW_REGS=regs,TMN_K1_START_OFFSET=offset,MW_FUSED=int(fused))
 elif kind in ('k3','k3_tma'):
  bi,bj,slot,acc,r240,serial=cfg;defs=dict(MW_BI=bi,MW_BJ=bj,MW_SLOT=slot,MW_ACC=acc,TMN_K3_REGS_24_240=r240,MW_SERIAL=serial,MW_FUSED=int(fused))
 else:
  typ,threads,serial,bulk=cfg;source=R/('separate_ln_tma.cu' if typ=='tma' else 'separate_ln.cu')
  defs=dict(MW_C=128,MW_TRANSPOSE=0,MW_THREADS=threads,MW_SERIAL=serial,MW_BULK_STORE=bulk)
 if fused:
  source=R/('original_'+kind+'.cu');defs={}
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc)]+['-D%s=%s'%v for v in defs.items()]
 digest=hashlib.sha256(source.read_bytes()+str(flags).encode()+b''.join((inc/p).read_bytes() for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest()
 out=R/'build'/(digest+'.cubin');out.parent.mkdir(exist_ok=True)
 if not out.exists():
  proc=subprocess.run(['nvcc',*flags,str(source),'-o',str(out)],capture_output=True,text=True)
  out.with_suffix('.ptxas.log').write_text(proc.stdout+proc.stderr)
  if proc.returncode:raise RuntimeError(proc.stderr)
  out.with_suffix('.json').write_text(json.dumps(dict(kind=kind,fused=fused,config=cfg,flags=flags,source=str(source)),indent=2))
 return out
@lru_cache(None)
def kernel(kind,fused,cfg):
 path=build(kind,fused,cfg);L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(path.read_bytes());unit=L.Unit(kind,'sm_90a',0,drv.drv,mod,{},str(path));k=unit.kernel('mw_ln' if kind=='ln' else 'infer_'+kind)
 smem=k1_smem(cfg) if kind=='k1' else k3_smem(cfg) if kind in ('k3','k3_tma') else 0
 if smem:k.set_max_dynamic_smem(smem)
 return k,smem

def ln(x,g,b,cfg):
 typ,threads,serial,bulk=cfg;m=x.numel()//128;y=torch.empty_like(x)
 k,_=kernel('ln',False,tuple(cfg));L=T._launch_module()
 if typ=='tma':
  tx=L.tensor_map(x,[64,64],dims=[128,m],strides_bytes=[256],swizzle='128B',l2='128B')
  ty=L.tensor_map(y,[64,16,1],dims=[128,m,1],strides_bytes=[256,m*256],swizzle='128B',l2='128B')
  p=L.Struct([tx,ty,g,b,y,None,None,m,1e-5]);grid=(m+63)//64
 else:p=L.Struct([x,g,b,y,None,None,m,1e-5]);grid=(m+threads//2-1)//(threads//2)
 k.launch((grid,1,1),(threads,1,1),[p],0);return y

def front(x,w,mask,g,b,fused,cfg):
 n=x.shape[1];ab=torch.empty((512,n,n),device=x.device,dtype=x.dtype);bi,bj=cfg[:2];ng=bi*bj//64;mb=2 if ng==1 else 1
 k,smem=kernel('k1',fused,tuple(cfg));L=T._launch_module()
 tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
 mz=tm(x,[64,bj,bi],[128,n,n],[256,n*256]);mw=tm(w,[64,64],[128,1024],[256]);ma=tm(ab,[64,1,32],[n,n,512],[n*2,n*n*2])
 tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
 p=L.Struct([mz,mw,ma,mask,g,b,ab,None,None,n,n,tj,tiles,1,n,1,1e-5,n*128,128,0,0])
 k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count*mb),1,1),(128*(ng+1),1,1),[p],smem);return ab

def output(tri,x,wp,wg,g,b,go,bo,res,fused,cfg,tma_residual=False):
 n=x.shape[1];y=torch.empty_like(x);bi,bj=cfg[:2];k,smem=kernel('k3_tma' if tma_residual else 'k3',fused,tuple(cfg));L=T._launch_module()
 tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
 mz=tm(x,[64,bj,bi],[128,n,n],[256,n*256]);mx=tm(tri,[64,1,64],[n,n,256],[n*2,n*n*2])
 mg=tm(wg,[64,32],[128,128],[256]);mp=tm(wp,[64,32],[256,128],[512]);my=tm(y,[64,16,1],[128,n,n],[256,n*256])
 tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
 p=L.Struct([mz,mx,mg,mp,my,g,b,go,bo,res,y,None,n,n,tj,tiles,1,0,1e-5,0])
 if tma_residual:
  mr=tm(res,[64,bj,bi],[128,n,n],[256,n*256]);p=L.Struct([p,mr]);assert len(p.pack())==896
 k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count),1,1),(384,1,1),[p],smem);return y

def triton_ln(x,g,b,cfg):
 from triton_ln import layer_norm_fwd_fused
 y=torch.empty_like(x);m=x.numel()//128;bm,bk,warps,stages=cfg
 layer_norm_fwd_fused[((m+bm-1)//bm,)](x,y,g,b,None,None,None,128,1,m,128,1e-5,bm,bk,0,False,num_warps=warps,num_stages=stages)
 return y
