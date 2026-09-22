from pathlib import Path
from functools import lru_cache
import torch,json,hashlib,subprocess,sys
R=Path(__file__).resolve().parent
from miniworld_engine.kernels.trimul_inproj.cuda import anthropic_training as T
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
from miniworld_engine.autotune.shape_key import both_key
@lru_cache(None)
def build(splits,group,accs,role=0,compact=0,wgroups=1,fullw=0,lnfrag=0,prefetch=0,dfast=0,wss=0):
 inc=T._upstream()/'csrc';source=R/'fused.cu';flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),f'-DSPLITS={splits}',f'-DRED_GROUP={group}',f'-DACCS={accs}',f'-DROLE={role}',f'-DCOMPACT={compact}',f'-DWGROUPS={wgroups}',f'-DFULLW={fullw}',f'-DLNFRAG={lnfrag}',f'-DPREFETCH={prefetch}',f'-DDFAST={dfast}',f'-DWSS={wss}']
 key=hashlib.sha256(source.read_bytes()+(R/'dgrad_wide.cuh').read_bytes()+(R/'wgrad_ss.cuh').read_bytes()+(R/'wgrad_ws.cuh').read_bytes()+(R/'wgrad_ws128.cuh').read_bytes()+str(flags).encode()+b''.join((inc/p).read_bytes() for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest();path=R/'build'/(key+'.cubin');path.parent.mkdir(exist_ok=True)
 if not path.exists():
  z=subprocess.run(['nvcc',*flags,str(source),'-o',str(path)],capture_output=True,text=True);path.with_suffix('.ptxas.log').write_text(z.stdout+z.stderr)
  if z.returncode:raise RuntimeError(z.stderr)
  path.with_suffix('.json').write_text(json.dumps(dict(splits=splits,group=group,accs=accs,flags=flags)))
 return path

@lru_cache(None)
def kernel(splits,group,accs,role=0,compact=0,wgroups=1,fullw=0,lnfrag=0,prefetch=0,dfast=0,wss=0):
 path=build(splits,group,accs,role,compact,wgroups,fullw,lnfrag,prefetch,dfast,wss)
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(path.read_bytes());unit=L.Unit('b1b4','sm_90a',0,drv.drv,mod,{},str(path));k=unit.kernel('fused_b1b4');k.set_max_dynamic_smem(max(98304 if dfast else (81920 if compact else 147456),(114688 if wss==4 else 65536 if wss==3 else (65536 if wss>=2 else 40960)*wgroups)));return k

class Plan:
 """Single-stream non-reentrant workspace. Allocate a separate plan per concurrent call."""
 def __init__(self,d,dy,saved,splits=64,group=32,accs=2,role=0,compact=0,wgroups=1,fullw=0,lnfrag=0,prefetch=0,dfast=0,wss=0):
  self.smem=max(98304 if dfast else (81920 if compact else 147456),(114688 if wss==4 else 65536 if wss==3 else (65536 if wss>=2 else 40960)*wgroups));self.wgroups=wgroups;self.k=kernel(splits,group,accs,role,compact,wgroups,fullw,lnfrag,prefetch,dfast,wss);ctx,_,_=saved
  (xn,wl,wlg,wr,wrg,wg,wp,go,pre,lf,rf,tri,norm,mean,rs,gate,proj)=ctx.saved_tensors
  n=d['n'];m=n*n;assert m%64==0 and xn.shape[-1]==128 and tri.shape[0]==256
  self.n=n;self.splits=splits;tiles=m//64;groups=(tiles+group-1)//group
  self.d=d;self.dy=dy;self.saved=saved;self.wpt=wp.t().contiguous()
  dg=torch.empty((m,128),device='cuda',dtype=torch.bfloat16);dt=torch.empty_like(tri);dwg=torch.empty((128,128),device='cuda',dtype=dg.dtype);dwp=torch.empty((128,256),device='cuda',dtype=dg.dtype);dgamma=torch.empty(256,device='cuda');dbeta=torch.empty_like(dgamma)
  partw=torch.empty((12,splits,4096),device='cuda');partln=torch.empty((tiles,512),device='cuda');group_ln=torch.empty((groups,512),device='cuda');counts=torch.zeros(12+groups+1,device='cuda',dtype=torch.int32)
  self.workspace=(partw,partln,group_ln,counts);self.outputs=(dg,dwg,dt,dgamma,dbeta,dwp)
  L=T._launch_module();tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
  row=lambda t,c:tm(t,[64,64],[c,m],[c*2])
  maps=[row(dy,128),row(gate,128),row(proj,128),row(xn,128),row(norm,256),tm(tri,[64,256],[m,256],[m*2]),tm(self.wpt,[64,64],[128,256],[256]),tm(dt,[64,16,1],[m,256,1],[m*2,m*512])]
  self.p=L.Struct([*maps,d['ds'],mean,rs,go,dg,dwg,dwp,dgamma,dbeta,partw,partln,group_ln,counts,m,n,tiles,groups]);self.grid=tiles+(6 if wss==4 else 4 if fullw else 12)*splits//(1 if wss>=3 else wgroups)
 def __call__(self):
  self.k.launch((self.grid,1,1),(128*self.wgroups,1,1),[self.p],self.smem);return self.outputs

def baseline(d,dy,saved):
 ctx,_,_=saved;xn,wl,wlg,wr,wrg,wg,wp,go,pre,lf,rf,tri,norm,mean,rs,gate,proj=ctx.saved_tensors;m=d['n']**2
 dp,dg=B.gate_elem_bwd_ew(dy.reshape(m,128),proj,gate,d['ds'],d['n']);dwg=torch.mm(xn.reshape(m,128).t(),dg)
 dt,dgamma,dbeta,dwp,_=B._te_backward(dp,norm,tri.reshape(256,m).t(),mean,rs,go,wp,False,shape_key=both_key(m))
 return dg,dwg,dt.t().reshape_as(tri),dgamma,dbeta,dwp

def data(n):
 sys.path.insert(0,str(R.parent/'anthropic_ln_equal_saves_20260919'));import core_saved as C
 d=C.setup(n);_,saved=C.forward(d,True,(3,64,2,2,1),(1,1));dy=torch.randn_like(d['x']);return d,dy,saved

def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
