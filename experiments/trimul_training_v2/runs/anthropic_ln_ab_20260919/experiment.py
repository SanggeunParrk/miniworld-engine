"""Isolated A/B: unchanged Anthropic ln_fragment in both paths; all training saves retained."""
from pathlib import Path
from functools import lru_cache
import hashlib,subprocess,json,torch
from miniworld_engine.kernels.trimul_inproj.cuda import anthropic_saved as S
from miniworld_engine.kernels.trimul_inproj.cuda import anthropic_training as T
R=Path(__file__).resolve().parent
@lru_cache(None)
def kernel(kind,c,h=256,cfg=(),trans=0,threads=128,serial=0,bulk=0):
 inc=T._upstream()/'csrc'
 if kind=='front':
  source=R/'fused_front.cu';name='mw_saved_front';smem=S.front_smem(c,h,cfg)
  defines=dict(MWK1_CZ=c,MWK1_CH=h,**dict(zip(('MWK1_BI','MWK1_BJ','MWK1_NSLOT','MWK1_SKCH','MWK1_SCHED'),cfg)))
 else:
  source=R/('separate_ln_tma.cu' if kind=='ln_tma' else 'separate_ln.cu');name='mw_ln';smem=0
  defines=dict(MW_C=c,MW_TRANSPOSE=trans,MW_THREADS=threads,MW_SERIAL=serial,MW_BULK_STORE=bulk)
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc)]+['-D%s=%s'%x for x in defines.items()]
 key=hashlib.sha256(source.read_bytes()+str(flags).encode()+(inc/'tmn_kernels.cuh').read_bytes()).hexdigest()
 out=R/(key+'.cubin')
 if not out.exists():
  p=subprocess.run(['nvcc',*flags,str(source),'-o',str(out)],capture_output=True,text=True)
  out.with_suffix('.log').write_text(p.stdout+p.stderr)
  if p.returncode:raise RuntimeError(p.stderr)
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());unit=L.Unit(name,'sm_90a',0,drv.drv,mod,{},str(out));k=unit.kernel(name)
 if smem:k.set_max_dynamic_smem(smem)
 return k,smem

def ln(x,g,b,eps=1e-5,trans=False,threads=128,serial=0):
 c=g.numel();m=x.numel()//c;y=torch.empty((m,c),device=x.device,dtype=x.dtype);mu=torch.empty(m,device=x.device);rs=torch.empty_like(mu)
 k,_=kernel('ln',c,trans=int(trans),threads=threads,serial=serial)
 p=T._launch_module().Struct([x,g,b,y,mu,rs,m,float(eps)])
 k.launch(((m+threads//2-1)//(threads//2),1,1),(threads,1,1),[p],0)
 return y,mu,rs

def front(x,w1,mask,g,b,eps=1e-5,cfg=None):
 n,c=x.shape[1],x.shape[-1];h=w1.shape[0]//4;cfg=tuple(cfg or S.front_default(h));bi,bj,slots,skch,sched=cfg;groups=bi*bj//64;minb=2 if groups==1 else 1
 ab=torch.empty((2*h,n,n),device=x.device,dtype=x.dtype);pre=torch.empty((4*h,n*n),device=x.device,dtype=x.dtype)
 xn=torch.empty_like(x);stats=torch.empty((2,n*n),device=x.device)
 L=T._launch_module();k,smem=kernel('front',c,h,cfg)
 tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
 mz=tm(x,[64,bj,bi],[c,n,n],[c*2,n*c*2]);mw=tm(w1,[64,64],[c,4*h],[c*2]);ma=tm(ab,[64,1,32],[n,n,2*h],[n*2,n*n*2])
 mg=tm(pre,[64,1,32],[n,n,2*h],[n*2,n*n*4]);mp=tm(pre[1],[64,1,32],[n,n,2*h],[n*2,n*n*4])
 tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
 base=L.Struct([mz,mw,ma,mask,g,b,ab,stats,xn,n,n,tj,tiles,1,n,1,float(eps),c*n,c,0,0]);p=L.Struct([base,mg,mp]);assert len(p.pack())==768
 k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count*minb),1,1),(128*(groups+1),1,1),[p],smem)
 return ab,pre,xn,stats[0],stats[1]


def ln_tma(x,g,b,eps=1e-5,trans=False,threads=128,serial=0,bulk=0):
 c=g.numel();m=x.numel()//c;y=torch.empty((m,c),device=x.device,dtype=x.dtype);mu=torch.empty(m,device=x.device);rs=torch.empty_like(mu)
 k,_=kernel('ln_tma',c,trans=int(trans),threads=128,serial=serial,bulk=bulk)
 L=T._launch_module()
 dims=[m,c] if trans else [c,m];stride=m*2 if trans else c*2
 tx=L.tensor_map(x,[64,64],dims=dims,strides_bytes=[stride],swizzle='128B',l2='128B')
 ty=L.tensor_map(y,[64,16,1],dims=[c,m,1],strides_bytes=[c*2,m*c*2],swizzle='128B',l2='128B')
 p=L.Struct([tx,ty,g,b,y,mu,rs,m,float(eps)])
 k.launch(((m+63)//64,1,1),(128,1,1),[p],0)
 return y,mu,rs
