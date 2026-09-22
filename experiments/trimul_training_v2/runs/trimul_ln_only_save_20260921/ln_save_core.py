"""Fused inference forward with optional LN activations emitted from K3."""
from pathlib import Path
from functools import lru_cache
import fcntl,hashlib,json,re,subprocess,sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_selective_recompute_20260920'))
import adapter_selective as S
T,A,I,B,F=S.T,S.A,S.I,S.B,S.F
DEFAULT=(2,64,4,1,1,1)
@lru_cache(None)
def kernel(mode,method,cfg=DEFAULT):
 bi,bj,slots,acc,regs,serial=cfg;inc=T._upstream()/'csrc';src=R/'ln_only_k3.cu'
 defs=dict(MW_SAVE_IN=mode&1,MW_SAVE_OUT=(mode>>1)&1,MW_STORE_METHOD=method,MW_BI=bi,MW_BJ=bj,MW_SLOT=slots,MW_ACC=acc,TMN_K3_REGS_24_240=regs,MW_SERIAL=serial,MW_FUSED=1)
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc)]+['-D%s=%s'%v for v in defs.items()]
 key=hashlib.sha256(src.read_bytes()+str(flags).encode()+b''.join((inc/p).read_bytes() for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest()
 out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
 with out.with_suffix('.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not out.exists():
   tmp=out.with_suffix('.tmp.cubin');p=subprocess.run(['nvcc',*flags,str(src),'-o',str(tmp)],capture_output=True,text=True)
   out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError(p.stderr)
   tmp.replace(out);out.with_suffix('.json').write_text(json.dumps(dict(mode=mode,method=method,config=cfg,flags=flags),indent=2))
 compiler=out.with_suffix('.ptxas.log').read_text()
 # Original no-save K3 itself has 16 B spill stores / 32 B loads.
 # Record resources instead of excluding the matched baseline from this A/B.
 print('RESOURCE',mode,method,[v for v in compiler.splitlines() if 'spill' in v or 'Used ' in v],flush=True)
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());unit=L.Unit('ln_only_k3','sm_90a',0,drv.drv,mod,{},str(out));k=unit.kernel('infer_k3');smem=I.k3_smem(cfg);k.set_max_dynamic_smem(smem)
 return k,smem

def output(d,tri,mode,method=0,cfg=DEFAULT,bufs=None):
 n,x=d['n'],d['x'];bi,bj=cfg[:2]
 y,lin,lout=bufs if bufs is not None else (torch.empty_like(x),torch.empty_like(x) if mode&1 else None,torch.empty((n*n,256),device=x.device,dtype=x.dtype) if mode&2 else None)
 k,smem=kernel(mode,method,tuple(cfg));L=T._launch_module()
 tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
 maps=[tm(x,[64,bj,bi],[128,n,n],[256,n*256]),tm(tri,[64,1,64],[n,n,256],[n*2,n*n*2]),tm(d['leaves'][5],[64,32],[128,128],[256]),tm(d['wp'],[64,32],[256,128],[512]),tm(y,[64,16,1],[128,n,n],[256,n*256])]
 tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
 base=L.Struct([*maps,d['gi'],d['bi'],d['go'],d['bo'],x,y,None,n,n,tj,tiles,1,0,1e-5,0])
 # Valid dummy maps for compile-time-disabled save fields; no dummy allocation.
 mi=tm(lin,[64,16,1],[128,n,n],[256,n*256]) if mode&1 else maps[-1]
 mo=tm(lout,[64,16,1],[256,n,n],[512,n*512]) if mode&2 else maps[-1]
 params=L.Struct([base,d['ds'],mi,mo,lin,lout])
 k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count),1,1),(384,1,1),[params],smem)
 return y,lin,lout

def forward(d,mode,method=0,cfg=DEFAULT):
 packed=F.pack(*d['leaves'][1:6]);d['wt'],d['w1']=packed[:5],packed[5]
 ab=I.front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],True,(2,64,8,2,-1,232,2))
 tri=B.packed_forward(ab[:256],ab[256:],128)
 y,lin,lout=output(d,tri,mode,method,cfg)
 return y,(ab,tri,packed),lin,lout
