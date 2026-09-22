"""Retain contraction operands/results; rematerialize only LN and projections.

Retained: ab=(lf,rf), tri. Recomputed: xn/stats, input preactivations,
output-normalized tri/stats, projection and gate. No repeated contraction,
no second ab/y production, and no repeated dropout/residual forward.
"""
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import fcntl,hashlib,json,subprocess,sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_all_recompute_20260920'))
import adapter as A
C,T,B,I,F=A.C,A.T,A.B,A.I,A.F

@lru_cache(None)
def kernel(kind,cfg):
    source=R/('recompute_'+kind+'.cu');inc=T._upstream()/'csrc'
    if kind=='front':
        defs=dict(MWK1_CZ=128,MWK1_CH=256,MW_FUSED=1)
        defs.update(zip(('MWK1_BI','MWK1_BJ','MWK1_NSLOT','MWK1_SKCH','MWK1_SCHED'),cfg))
        smem=C.S.front_smem(128,256,cfg)-(cfg[0]*cfg[1]//64)*8192
    else:
        defs=dict(MWK3_CZ=128,MWK3_CH=256)
        defs.update(zip(('MWK3_BI','MWK3_BJ','MWK3_NSLOT','MWK3_NACC','MWK3_REGS','MWK3_LNSERIAL'),cfg))
        smem=T.validate_config(128,256,cfg)-8*2048
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc)]+['-D%s=%s'%v for v in defs.items()]
    key=hashlib.sha256(source.read_bytes()+str(flags).encode()+b''.join((inc/p).read_bytes() for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest()
    out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
    with out.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not out.exists():
            tmp=out.with_suffix('.tmp.cubin');p=subprocess.run(['nvcc',*flags,str(source),'-o',str(tmp)],capture_output=True,text=True)
            out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
            if p.returncode:raise RuntimeError(p.stderr)
            tmp.replace(out)
    L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes())
    unit=L.Unit('recompute_'+kind,'sm_90a',0,drv.drv,mod,{},str(out))
    k=unit.kernel('mw_recompute_'+kind);k.set_max_dynamic_smem(smem)
    return k,smem

def front(d,ab,cfg):
    n,x=d['n'],d['x'];m=n*n
    xn=torch.empty_like(x);mu=torch.empty(m,device=x.device);rs=torch.empty_like(mu)
    pre=torch.empty((1024,m),device=x.device,dtype=x.dtype)
    L=T._launch_module();k,smem=kernel('front',tuple(cfg));bi,bj=cfg[:2];ng=bi*bj//64;mb=2 if ng==1 else 1
    tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
    mz=tm(x,[64,bj,bi],[128,n,n],[256,n*256]);mw=tm(d['w1'],[64,64],[128,1024],[256])
    ma=tm(ab,[64,1,32],[n,n,512],[n*2,m*2])
    mg=tm(pre,[64,1,32],[n,n,512],[n*2,m*4]);mp=tm(pre[1],[64,1,32],[n,n,512],[n*2,m*4])
    tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
    base=L.Struct([mz,mw,ma,d['mask'],d['gi'],d['bi'],ab,mu,xn,n,n,tj,tiles,1,n,1,1e-5,n*128,128,0,0])
    p=L.Struct([base,mg,mp,rs])
    k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count*mb),1,1),(128*(ng+1),1,1),[p],smem)
    return xn,mu,rs,pre

def output(d,tri,xn,cfg):
    n=d['n'];m=n*n;bi,bj=cfg[:2]
    norm=xn.new_empty((m,256));proj=xn.new_empty((m,128));gate=torch.empty_like(proj)
    mean=xn.new_empty((m,),dtype=torch.float32);rs=torch.empty_like(mean)
    L=T._launch_module();k,smem=kernel('output',tuple(cfg))
    tm=lambda t,box,dims,strides,l2='128B':L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2=l2)
    maps=[tm(xn,[64,bj,bi],[128,n,n],[256,n*256]),tm(tri,[64,1,64],[n,n,256],[n*2,n*n*2]),
          tm(d['leaves'][5],[64,32],[128,128],[256],'256B'),tm(d['wp'],[64,32],[256,128],[512],'256B'),
          tm(proj,[64,16,1],[128,n,n],[256,n*256])]
    tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
    base=L.Struct([*maps,d['go'],d['bo'],d['go'],d['bo'],None,None,None,n,n,tj,tiles,0,0,1e-5,0])
    saves=[tm(norm,[64,16,1],[256,n,n],[512,n*512]),maps[-1],
           tm(gate,[64,16,1],[128,n,n],[256,n*256]),maps[0]]
    p=L.Struct([base,*saves,norm,proj,gate,mean,rs,None])
    assert len(p.pack())==1344
    k.launch((min(tiles,torch.cuda.get_device_properties(0).multi_processor_count),1,1),(384,1,1),[p],smem)
    return norm,mean,rs,proj,gate

def forward(d):
    packed=F.pack(*d['leaves'][1:6]);d['wt'],d['w1']=packed[:5],packed[5]
    ab=I.front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],True,(2,64,8,2,-1,232,2))
    tri=B.packed_forward(ab[:256],ab[256:],128)
    y=A.output_no_save(d,tri)
    # No xn/pre/norm/proj/gate is returned or retained by this forward.
    return y,(ab,tri,packed)

def rematerialize(d,kept,k1,k3):
    ab,tri,packed=kept;d['wt'],d['w1']=packed[:5],packed[5]
    xn,mu,rs,pre=front(d,ab,k1)
    norm,mo,ro,proj,gate=output(d,tri,xn,k3)
    ctx=SimpleNamespace(saved_tensors=(xn,*d['wt'],d['wp'],d['go'],pre,ab[:256],ab[256:],tri,norm,mo,ro,gate,proj),
                        eps=1e-5,h=128,mm=d['mask'],sm90_dual_bwd=False,dropscale=d['ds'],seq_len=d['n'])
    return ctx,mu,rs

class Training(A.Training):
    def __init__(self,a,k3):
        super().__init__(a,k3)
        self.rk1=(3,64,2,2,1) if self.n==384 else (1,128,2,2,1)
        self.rk3=tuple(k3)
        tuning=R/('tuning-L%d.json'%self.n)
        if tuning.exists():
            data=json.loads(tuning.read_text())
            if data.get('complete'):
                self.rk1=tuple(data['front']['winner'])
                self.rk3=tuple(data['output']['winner'])

    def backward_selective(self,kept):
        return self.backward(rematerialize(self.d,kept,self.rk1,self.rk3))

    def selective(self):
        y,kept=forward(self.d)
        return y,self.backward_selective(kept)
