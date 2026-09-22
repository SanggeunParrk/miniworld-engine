from pathlib import Path
from functools import lru_cache
import fcntl,hashlib,json,subprocess,sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_ln_only_save_20260921'));import ln_save_core as LN
T,I,F,B=LN.T,LN.I,LN.F,LN.B
K1=(2,64,8,2,-1,232,2);K3=LN.DEFAULT
@lru_cache(None)
def kernel(kind,ln=0,stats=0,pg=False,method=0,cfg=None):
 inc=T._upstream()/'csrc';src=R/('save_'+kind+'.cu');defs=dict(MW_SAVE_PG=int(pg),MW_PG_METHOD=method,XHAT_FP32=method)
 if kind=='k1':smem=I.k1_smem(K1)+(32768 if pg and method==0 else 0);threads=384
 else:
  bi,bj,slots,acc,regs,serial=cfg or K3;defs.update(MW_SAVE_IN=ln&1,MW_SAVE_OUT=0,MW_STORE_METHOD=0,MW_SAVE_STATS_IN=stats&1,MW_SAVE_STATS_OUT=(stats>>1)&1,MW_BI=bi,MW_BJ=bj,MW_SLOT=slots,MW_ACC=acc,TMN_K3_REGS_24_240=regs,MW_SERIAL=serial,MW_FUSED=1);smem=I.k3_smem(cfg or K3)+(32768 if pg and method==0 else 0);threads=384
 assert smem<=232448
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc)]+['-D%s=%s'%v for v in defs.items()]
 deps=[src,*[inc/p for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh')]];key=hashlib.sha256(b''.join(p.read_bytes() for p in deps)+str(flags).encode()).hexdigest();out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
 with out.with_suffix('.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not out.exists():
   tmp=out.with_suffix('.tmp.cubin');p=subprocess.run(['nvcc',*flags,str(src),'-o',str(tmp)],capture_output=True,text=True);out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError(p.stderr)
   tmp.replace(out);out.with_suffix('.json').write_text(json.dumps(dict(kind=kind,ln=ln,stats=stats,pg=pg,method=method,defines=defs,smem=smem,threads=threads,flags=flags),indent=2))
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());unit=L.Unit('save_'+kind,'sm_90a',0,drv.drv,mod,{},str(out));k=unit.kernel('save_'+kind);k.set_max_dynamic_smem(smem)
 print('CUBIN',kind,ln,stats,pg,method,str(out),[x.strip() for x in out.with_suffix('.ptxas.log').read_text().splitlines() if 'spill' in x or 'Used ' in x],flush=True)
 return k,smem

def tm(t,box,dims,strides):return T._launch_module().tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
def front(d,pg=False,method=0,bufs=None):
 n=d['n'];x=d['x'];m=n*n;bi,bj=K1[:2];L=T._launch_module();k,smem=kernel('k1',pg=pg,method=method)
 ab,pre=bufs if bufs else (torch.empty((512,n,n),device=x.device,dtype=x.dtype),torch.empty((1024,m),device=x.device,dtype=x.dtype) if pg else None)
 mz=tm(x,[64,bj,bi],[128,n,n],[256,n*256]);mw=tm(d['w1'],[64,64],[128,1024],[256]);ma=tm(ab,[64,1,32],[n,n,512],[n*2,m*2])
 mg=tm(pre,[64,1,32],[n,n,512],[n*2,m*4]) if pg else ma;mp=tm(pre[1],[64,1,32],[n,n,512],[n*2,m*4]) if pg else ma;tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
 base=L.Struct([mz,mw,ma,d['mask'],d['gi'],d['bi'],ab,None,None,n,n,tj,tiles,1,n,1,1e-5,n*128,128,0,0]);p=L.Struct([base,mg,mp]);k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count),1,1),(384,1,1),[p],smem);return ab,pre

def output(d,tri,ln=1,stats=0,pg=False,method=0,bufs=None,cfg=None):
 n=d['n'];x=d['x'];m=n*n;bi,bj=(cfg or K3)[:2];L=T._launch_module();k,smem=kernel('k3',ln,stats,pg,method,cfg)
 if bufs is None:
  e=lambda c:torch.empty((m,c),device=x.device,dtype=x.dtype)
  f=lambda:torch.empty(m,device=x.device,dtype=torch.float32)
  saves=dict(xn=e(128) if ln&1 else None,xnout=tri if method==-1 else torch.empty((m//64,16,4,2,32,4),device=x.device,dtype=torch.float32 if method==1 else torch.float16 if method>=2 else x.dtype) if ln&2 else None,mi=f() if stats&1 else None,ri=f() if stats&1 else None,mo=None,ro=torch.empty(m*(2 if method in (-1,3) else 1),device=x.device,dtype=torch.float32) if stats&2 else None,proj=e(128) if pg else None,gate=e(128) if pg else None);y=torch.empty_like(x)
 else:y,saves=bufs
 if method in (-1,3):saves['mo']=saves['ro'][m:]
 if method==-1:saves['xnout']=tri
 maps=[tm(x,[64,bj,bi],[128,n,n],[256,n*256]),tm(tri,[64,1,64],[n,n,256],[n*2,m*2]),tm(d['leaves'][5],[64,32],[128,128],[256]),tm(d['wp'],[64,32],[256,128],[512]),tm(y,[64,16,1],[128,n,n],[256,n*256])]
 tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj;base=L.Struct([*maps,d['gi'],d['bi'],d['go'],d['bo'],x,y,None,n,n,tj,tiles,1,0,1e-5,0])
 om=lambda key,c:tm(saves[key],[64,16,1],[c,n,n],[c*2,n*c*2]) if saves[key] is not None else maps[-1]
 outmap=L.tensor_map(saves['xnout'],[16,32],dims=[m,256],strides_bytes=[m*4],swizzle='64B',l2='128B') if method==1 else maps[-1]
 p=L.Struct([base,d['ds'],om('xn',128),outmap,saves['xn'],saves['xnout'],saves['mi'],saves['ri'],saves['mo'],saves['ro'],om('proj',128),om('gate',128),saves['proj'],saves['gate']]);k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count),1,1),(384,1,1),[p],smem);return y,saves

def forward(d,ln=1,stats=0,input_pg=False,output_pg=False,input_method=0,output_method=0):
 packed=F.pack(*d['leaves'][1:6]);d['wt'],d['w1']=packed[:5],packed[5];ab,pre=front(d,input_pg,input_method);tri=B.packed_forward(ab[:256],ab[256:],128);y,saves=output(d,tri,ln,stats,output_pg,output_method);saves['pre']=pre;return y,(ab,tri,packed),saves
