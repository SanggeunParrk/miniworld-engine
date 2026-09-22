from pathlib import Path
from functools import lru_cache
import sys,torch,json,hashlib,subprocess
from types import SimpleNamespace
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'anthropic_ln_inference_20260919'))
import core as I
from miniworld_engine.kernels.trimul_inproj.cuda import anthropic_saved as S,anthropic_training as T
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual_bwd
from miniworld_engine.autotune.shape_key import both_key
@lru_cache(None)
def build(kind,fused,cfg):
 inc=T._upstream()/'csrc';source=R/(kind+'.cu')
 if kind=='front':
  defs=dict(MWK1_CZ=128,MWK1_CH=256,MW_FUSED=int(fused));defs.update(zip(('MWK1_BI','MWK1_BJ','MWK1_NSLOT','MWK1_SKCH','MWK1_SCHED'),cfg));smem=S.front_smem(128,256,cfg)
 else:defs=dict(MW_C=128,MW_TRANSPOSE=0,MW_SERIAL=cfg[0],MW_BULK_STORE=cfg[1]);smem=0
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc)]+['-D%s=%s'%v for v in defs.items()]
 key=hashlib.sha256(source.read_bytes()+str(flags).encode()+b''.join((inc/p).read_bytes() for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest();p=R/'build'/(key+'.cubin');p.parent.mkdir(exist_ok=True)
 if not p.exists():
  z=subprocess.run(['nvcc',*flags,str(source),'-o',str(p)],capture_output=True,text=True);p.with_suffix('.ptxas.log').write_text(z.stdout+z.stderr)
  if z.returncode:raise RuntimeError(z.stderr)
  p.with_suffix('.json').write_text(json.dumps(dict(kind=kind,fused=fused,config=cfg,flags=flags)))
 return p,smem

@lru_cache(None)
def kernel(kind,fused,cfg):
 p,smem=build(kind,fused,cfg)
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(p.read_bytes());u=L.Unit(kind,'sm_90a',0,drv.drv,mod,{},str(p));k=u.kernel('mw_saved_front' if kind=='front' else 'mw_ln')
 if smem:k.set_max_dynamic_smem(smem)
 return k,smem

def ln(x,g,b,cfg):
 m=x.numel()//128;y=torch.empty_like(x);mu=torch.empty(m,device=x.device);rs=torch.empty_like(mu);L=T._launch_module();k,_=kernel('ln',False,tuple(cfg))
 tx=L.tensor_map(x,[64,64],dims=[128,m],strides_bytes=[256],swizzle='128B',l2='128B');ty=L.tensor_map(y,[64,16,1],dims=[128,m,1],strides_bytes=[256,m*256],swizzle='128B',l2='128B')
 p=L.Struct([tx,ty,g,b,y,mu,rs,m,1e-5]);k.launch(((m+63)//64,1,1),(128,1,1),[p],0);return y,mu,rs

def front(x,w,mask,g,b,fused,cfg,lc=(0,1)):
 n=x.shape[1];m=n*n
 if fused:z=x;xn=torch.empty_like(x);mu=torch.empty(m,device=x.device);rs=torch.empty_like(mu)
 else:xn,mu,rs=ln(x,g,b,lc);z=xn
 ab=torch.empty((512,n,n),device=x.device,dtype=x.dtype);pre=torch.empty((1024,m),device=x.device,dtype=x.dtype)
 L=T._launch_module();k,smem=kernel('front',fused,tuple(cfg));bi,bj=cfg[:2];ng=bi*bj//64;mb=2 if ng==1 else 1
 tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
 mz=tm(z,[64,bj,bi],[128,n,n],[256,n*256]);mw=tm(w,[64,64],[128,1024],[256]);ma=tm(ab,[64,1,32],[n,n,512],[n*2,m*2]);mg=tm(pre,[64,1,32],[n,n,512],[n*2,m*4]);mp=tm(pre[1],[64,1,32],[n,n,512],[n*2,m*4]);tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
 base=L.Struct([mz,mw,ma,mask,g,b,ab,mu,xn,n,n,tj,tiles,1,n,1,1e-5,n*128,128,0,0]);p=L.Struct([base,mg,mp,rs]);assert len(p.pack())==832
 k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count*mb),1,1),(128*(ng+1),1,1),[p],smem);return xn,mu,rs,ab,pre

def setup(n):
 sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'));from check_full import setup as old
 leaves,call,mask,ds=old(n,'bidir');x,*rest=leaves;wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo=rest
 wt=[q.t().contiguous() for q in (wl,wlg,wr,wrg,wg)]
 gate=torch.cat((wlg,wrg),0);proj=torch.cat((wl,wr),0);w1=torch.stack((gate.reshape(-1,32,128),proj.reshape(-1,32,128)),1).reshape(1024,128)
 return dict(n=n,x=x,leaves=leaves,reference=call,mask=mask.reshape(n,n).float(),ds=ds.reshape(n,128),wt=wt,wp=wp,gi=gi,bi=bi,go=go,bo=bo,w1=w1)

def forward(d,fused,cfg,lc=(0,1)):
 n=d['n'];m=n*n;x=d['x'];xn,mu,rs,ab,pre=front(x,d['w1'],d['mask'],d['gi'],d['bi'],fused,cfg,lc)
 lf,rf=ab[:256],ab[256:];tri=B.packed_forward(lf,rf,128)
 y,norm,mo,ro,proj,gate=T.fused_output(tri,xn,d['wp'],d['wt'][4],d['go'],d['bo'],x.reshape(m,128),d['ds'],1e-5)
 ctx=SimpleNamespace(saved_tensors=(xn,*d['wt'],d['wp'],d['go'],pre,lf,rf,tri,norm,mo,ro,gate,proj),eps=1e-5,h=128,mm=d['mask'],sm90_dual_bwd=False,dropscale=d['ds'],seq_len=n)
 return y.reshape_as(x),(ctx,mu,rs)

def backward(d,saved,dy):
 ctx,mu,rs=saved;r=B._BidirBackHalfTriton.backward(ctx,dy);dx,dgi,dbi=input_ln_residual_bwd(r[0].reshape(-1,128),d['x'].reshape(-1,128),d['gi'],mu,rs,r[12],both_key(d['n']**2))
 return (dx.reshape_as(d['x']),*[v.t() for v in r[1:6]],r[6],dgi,dbi,r[7],r[8])
