from pathlib import Path
import sys,hashlib,subprocess,json,fcntl,torch
from functools import lru_cache
R=Path(__file__).resolve().parent
OLD=R.parent/'trimul_split_bwd_20260921'
sys.path.insert(0,str(OLD))
import saved_plans as SP
T=SP.T
@lru_cache(None)
def build(count,length,part,defines=()):
 inc=T._upstream()/'csrc';src=R/'b1_fused.cu'
 defs=dict(UCOUNT=count,TRAIN_L=length,USE_SAVED_XN=1,PART_ONLY=part,**dict(defines))
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-I'+str(SP.P7),'-I'+str(R)]+['-D%s=%s'%kv for kv in defs.items()]
 deps=[src,*sorted(R.glob('*.cuh')),*sorted(R.glob('*.inc')),*[SP.P7/p for p in ('warp_primitives.cuh','front_mn_primitives.cuh','front_primitives.cuh')],*[inc/p for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh')]]
 key=hashlib.sha256(b''.join(p.read_bytes() for p in deps)+str(flags).encode()).hexdigest();out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
 with out.with_suffix('.lock').open('w') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if not out.exists():
   p=subprocess.run(['nvcc',*flags,str(src),'-o',str(out)],capture_output=True,text=True);out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError(p.stderr)
   out.with_suffix('.json').write_text(json.dumps(dict(defines=defs,flags=flags,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in deps}),indent=2))
 log=out.with_suffix('.ptxas.log').read_text()
 print('BUILD',out.name,log,flush=True)
 if 'C7507' in log:raise RuntimeError('Rejected: requested register redistribution was ignored by ptxas')
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());u=L.Unit('shared_b1','sm_90a',0,drv.drv,mod,{},str(out));k=u.kernel('b1_fused');smem=227840+512*int(dict(defines).get('B1_SPLIT_DN',0));k.set_max_dynamic_smem(smem)
 return k,u.kernel('b1_reduce'),smem
class Plan(SP.B1):
 def __init__(self,d,dy,xhat,rstd,count=132,part=2,defines=None):
  if (defines or {}).get('B1_SPLIT_DN') and (((defines or {}).get('B1_PREFETCH_DY') and not (defines or {}).get('B1_DN_REG')) or not (defines or {}).get('B1_DIRECT_DP')):raise ValueError('split dNorm requires direct dProj and disables dy prefetch')
  if (defines or {}).get('B1_REG_EARLY_RAW') and not all((defines or {}).get(k) for k in ('B1_DN_REG','B1_DN_PAIR','B1_SPLIT_DN','B1_EARLY_RAW')):raise ValueError('early raw requires register dNorm, paired GEMM, split stats, early scheduling')
  if (defines or {}).get('B1_PARAM_SHARED') and (not (defines or {}).get('B1_DN_REG') or (defines or {}).get('B1_PREFETCH_DY')):raise ValueError('shared param reduction requires register dNorm and no dy prefetch')
  if ((defines or {}).get('B1_ASYNC_WP_DN') or (defines or {}).get('B1_DEFER_DG_STORE')) and not all((defines or {}).get(k) for k in ('PAIR_WP','B1_DIRECT_DP','B1_DN_PAIR','B1_REG_EARLY_RAW','GATE_PHASE')):raise ValueError('overlap requires the verified paired/register/early-raw schedule')
  if (defines or {}).get('B1_WG_PARAM_SYNC') and not (defines or {}).get('B1_PARAM_SHARED'):raise ValueError('group-only parameter synchronization requires shared parameter path')
  if (defines or {}).get('B1_SKIP_FINAL_GROUP_SYNC') and not (defines or {}).get('B1_DTRI_ASYNC_EPI'):raise ValueError('tail-sync elision requires async epilogue followed by caller CTA barrier')
  self.dtri_store_c=(defines or {}).get("B1_DTRI_STORE_C",16)
  if self.dtri_store_c not in (16,32,64,128):raise ValueError("invalid dTri store tile")
  self.xhat,self.rstd=xhat,rstd
  if d['n'] not in (384,768) or d['x'].dtype != torch.bfloat16 or d['x'].numel()!=d['n']**2*128:raise ValueError('This development kernel supports BF16 C128, L384/768 only')
  if part!=2 or not 1<=count<=132:raise ValueError('The cooperative two-phase grid requires 1..132 CTAs and part=2')
  self.count,self.part,self.d=count,part,d;self.reduce_size=49664;self.threads=384 if (defines or {}).get("PRODUCER") else 256
  self.k,self.reduce,self.smem=build(count,d['n'],part,tuple(sorted((defines or {}).items())))
  m=d['n']**2;x=d['x'];bf=lambda shape:torch.empty(shape,device=x.device,dtype=x.dtype)
  self.outputs=(bf((m,128)),bf((128,128)),bf((256,d['n'],d['n'])),torch.empty(256,device=x.device),torch.empty(256,device=x.device),bf((128,256)))
  self.partw=torch.empty((count,49152),device=x.device);self.partln=torch.empty((count,512),device=x.device)
  self.counts=torch.zeros(2+count,device=x.device,dtype=torch.int32);self.bind(dy,xhat,rstd)

 def bind(self,dy,xhat,rstd):
  self.xhat,self.rstd=xhat,rstd;d=self.d;n=d['n'];m=n*n;L=T._launch_module();self.wpt=d['wp'].t().contiguous();self.wgt=d['wt'][4];es=xhat.element_size()
  tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
  row=lambda t,c:tm(t,[64,64],[c,m],[c*2])
  dg,dwg,dt,dgo,dbo,dwp=self.outputs
  maps=[row(dy,128),row(d['x'],128),tm(xhat,[128//es,256],[m,256],[m*es]),tm(self.wpt,[64,64],[128,256],[256]),tm(self.wgt,[64,64],[128,128],[256]),tm(dt,[64,self.dtri_store_c,1],[m,256,1],[m*2,m*512]),row(dg,128)]
  statsmap=L.tensor_map(rstd,[64,1],dims=[m,2],strides_bytes=[m*4],swizzle='none',l2='128B') if rstd.numel()==2*m else maps[2]
  maps.insert(3,statsmap)
  self.p=L.Struct([*maps,d['ds'],d['gi'],d['bi'],d['go'],d['bo'],rstd,dg,dwg,dwp,dgo,dbo,self.partw,self.partln,self.counts,m,n,m//64])
  self.inputs=(dy,self.wpt,self.wgt,xhat,rstd)
