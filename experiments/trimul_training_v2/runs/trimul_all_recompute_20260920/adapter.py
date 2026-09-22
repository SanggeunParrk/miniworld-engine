"""Full activation checkpointing, preserving the selected backward kernels.

Forward retains original input/parameters/masks only. No-save Anthropic K1/K3
compute temporary contraction operands; they are not retained for backward.
Backward rematerializes the complete saved forward once, then uses the exact
same B1-B12 kernels as save-all. This is whole-module recomputation, not fused
register-local backward recomputation.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import os
import subprocess
import sys
import torch

R=Path(__file__).resolve().parent
P=R.parent/'anthropic_b7b12_fusion_20260920'
sys.path.insert(0,str(P))
import training_forward_adapter as F
from compare_all_training import BoundB1, WarpPlan, RingPlan
C,T,B=F.C,F.T,F.B
I=C.I

@lru_cache(None)
def output_kernel(cfg):
    bi,bj,slots,acc,regs240,serial=cfg
    defs=dict(MW_BI=bi,MW_BJ=bj,MW_SLOT=slots,MW_ACC=acc,
              TMN_K3_REGS_24_240=regs240,MW_SERIAL=serial,MW_FUSED=1)
    inc=T._upstream()/'csrc'
    source=R/'no_save_k3.cu'
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
           '-I'+str(inc)]+['-D%s=%s'%v for v in defs.items()]
    key=hashlib.sha256(source.read_bytes()+str(flags).encode()+b''.join(
        (inc/p).read_bytes() for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest()
    out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
    if not out.exists():
        temporary=out.with_suffix('.%d.cubin'%os.getpid())
        proc=subprocess.run(['nvcc',*flags,str(source),'-o',str(temporary)],capture_output=True,text=True)
        out.with_suffix('.ptxas.log').write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise RuntimeError(proc.stderr)
        temporary.replace(out)
    launch=T._launch_module();driver=launch.BlockDriver(device=0)
    module=driver.load(out.read_bytes())
    unit=launch.Unit('no_save_k3','sm_90a',0,driver.drv,module,{},str(out))
    k=unit.kernel('infer_k3');smem=I.k3_smem(cfg)
    k.set_max_dynamic_smem(smem)
    return k,smem

def output_no_save(d,tri,cfg=(2,64,4,1,1,1)):
    n,x=d['n'],d['x'];y=torch.empty_like(x);bi,bj=cfg[:2]
    k,smem=output_kernel(tuple(cfg));L=T._launch_module()
    tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
    maps=[tm(x,[64,bj,bi],[128,n,n],[256,n*256]),
          tm(tri,[64,1,64],[n,n,256],[n*2,n*n*2]),
          tm(d['leaves'][5],[64,32],[128,128],[256]),
          tm(d['wp'],[64,32],[256,128],[512]),
          tm(y,[64,16,1],[128,n,n],[256,n*256])]
    tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
    base=L.Struct([*maps,d['gi'],d['bi'],d['go'],d['bo'],x,y,None,
                   n,n,tj,tiles,1,0,1e-5,0])
    params=L.Struct([base,d['ds']])
    k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count),1,1),
             (384,1,1),[params],smem)
    return y

def forward_no_save(d):
    packed=F.pack(*d['leaves'][1:6])
    d['wt'],d['w1']=packed[:5],packed[5]
    ab=I.front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],True,(2,64,8,2,-1,232,2))
    tri=B.packed_forward(ab[:256],ab[256:],128)
    return output_no_save(d,tri)

class Training:
    def __init__(self,a,k3):
        self.a=a;self.d=a['d'];self.n=self.d['n'];self.k3=k3
        self.p1=BoundB1(self.d,a['dy'],a['s'],132,2,'dual_ln_prefetch')
        self.p7=(WarpPlan(a,count=264,splits=13,source='front_prefetch_lnpair_storepipe')
                 if self.n==384 else RingPlan(a,count=264,splits=20,source='front_ring96_cache3_glu_ahead_u4_early_writer32'))

    def forward_saved(self):
        return F.forward(self.d,k3=self.k3)

    def backward(self,saved):
        a,d,n=self.a,self.d,self.n
        ctx,a['mu'],a['rs']=saved
        (a['xn'],a['wl'],a['wlg'],a['wr'],a['wrg'],a['wg'],_,_,a['pre'],lf,rf,*_)=ctx.saved_tensors
        self.p1.bind(d,a['dy'],saved)
        dg,dwg,dt,dgo,dbo,dwp=self.p1()
        dl,dr=B.packed_backward(dt,lf,rf,128)
        self.p7.bind(dl,dr,dg,a['dy'])
        dx,dwl,dwlg,dwr,dwrg,dgi,dbi=self.p7()
        return (dx.reshape_as(d['x']),dwl.t(),dwlg.t(),dwr.t(),dwrg.t(),dwg.t(),dwp,dgi,dbi,dgo,dbo)

    def backward_recompute(self):
        _,saved=self.forward_saved()
        return self.backward(saved)

    def saved(self):
        y,saved=self.forward_saved()
        return y,self.backward(saved)

    def recompute(self):
        y=forward_no_save(self.d)
        return y,self.backward_recompute()
