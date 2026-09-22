"""No global forward-activation buffers in either fused backward plan."""
from pathlib import Path
from functools import lru_cache
import ctypes,fcntl,hashlib,json,re,subprocess,sys,torch
R=Path(__file__).resolve().parent
P7=R.parent/'anthropic_b7b12_fusion_20260920'
sys.path.insert(0,str(R.parent/'trimul_selective_recompute_20260920'))
import adapter_selective as S
T=S.T;B=S.B
DEFINES=()

@lru_cache(None)
def build(kind,count,splits,part,slices=2,defines=()):
    inc=T._upstream()/'csrc';source=R/('b7_warp_specialized.cu' if kind=='b7' and dict(defines).get('PIPE_DW') else kind+'_fused.cu')
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-I'+str(P7),'-I'+str(R),
           '-DUCOUNT=%d'%count,'-DDW_SPLITS=%d'%splits,'-DPART_ONLY=%d'%part,'-DWGRAD_SLICES=%d'%slices]+['-D%s=%s'%(k,v) for k,v in defines]
    deps=[source,R/'b7_pipe_dw.inc',R/'b7_packed_dx.inc',R/'b7_packed_dw.inc',R/'rs_recompute.cuh',R/'ln_recompute.cuh',R/'common_recompute.cuh',R/'b1_saved_math.inc',R/'b1_pipeline_math.inc',P7/'warp_primitives.cuh',P7/'front_mn_primitives.cuh',P7/'front_primitives.cuh']
    deps+=[inc/p for p in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh')]
    key=hashlib.sha256(b''.join(p.read_bytes() for p in deps)+str(flags).encode()).hexdigest()
    out=R/'build'/(key+'.cubin');out.parent.mkdir(exist_ok=True)
    with out.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not out.exists():
            tmp=out.with_suffix('.tmp.cubin');p=subprocess.run(['nvcc',*flags,str(source),'-o',str(tmp)],capture_output=True,text=True)
            out.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
            if p.returncode:raise RuntimeError(p.stderr)
            tmp.replace(out)
            out.with_suffix('.json').write_text(json.dumps(dict(kind=kind,count=count,splits=splits,part=part,flags=flags),indent=2))
    compiler=out.with_suffix('.ptxas.log').read_text()
    if re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)',compiler):raise RuntimeError('Spilling variant rejected: '+str(out))
    L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(out.read_bytes());u=L.Unit(kind,'sm_90a',0,drv.drv,mod,{},str(out))
    entry='b1_fused' if kind=='b1' else 'front_b7b12';k=u.kernel(entry);smem=227840 if kind=='b1' else dict(defines).get('SMEM_BYTES',131072)
    k.set_max_dynamic_smem(smem)
    print('CUBIN',kind,out,out.with_suffix('.ptxas.log').read_text(),flush=True)
    return k,u.kernel('b1_reduce' if kind=='b1' else 'front_reduce'),smem

class Base:
    def launch(self):
        if self.part==2:
            L=T._launch_module();drv=self.k.unit.drv;p=L._Packed([self.p]);stream=int(torch.cuda.current_stream().cuda_stream)
            drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.count,1,1,getattr(self,'threads',256),1,1,self.smem,drv.d.CUstream(stream),ctypes.addressof(p.array)))
        else:
            self.k.launch((self.count,1,1),(getattr(self,'threads',256),1,1),[self.p],self.smem)
            self.reduce.launch(((self.reduce_size+255)//256,1,1),(256,1,1),[self.p],0)
        return self.outputs
    __call__=launch

class B1(Base):
    def __init__(self,d,dy,tri,count=132,splits=16,part=2,defines=None):
        self.defines=tuple(sorted((dict(DEFINES) if defines is None else defines).items()))
        self.count,self.splits,self.part=count,splits,part;self.d=d;self.reduce_size=49664
        self.k,self.reduce,self.smem=build('b1',count,splits,part,defines=self.defines)
        m=d['n']**2;x=d['x'];bf=lambda shape:torch.empty(shape,device=x.device,dtype=x.dtype)
        self.outputs=(bf((m,128)),bf((128,128)),bf((256,d['n'],d['n'])),torch.empty(256,device=x.device),torch.empty(256,device=x.device),bf((128,256)))
        gates=dict(self.defines).get('GATE_SPLITS',splits)
        self.partw=torch.empty((gates+2*splits,16384),device=x.device);self.partln=torch.empty((count-gates-2*splits,512),device=x.device)
        self.counts=torch.zeros(2+count,device=x.device,dtype=torch.int32);self.bind(dy,tri)
    def bind(self,dy,tri):
        d=self.d;n=d['n'];m=n*n;L=T._launch_module();self.wpt=d['wp'].t().contiguous()
        self.wgt=d['wt'][4]
        tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
        row=lambda t,c:tm(t,[64,64],[c,m],[c*2])
        dg,dwg,dt,dgo,dbo,dwp=self.outputs
        maps=[row(dy,128),row(d['x'],128),tm(tri,[64,256],[m,256],[m*2]),
              tm(self.wpt,[64,64],[128,256],[256]),tm(self.wgt,[64,64],[128,128],[256]),
              tm(dt,[64,16,1],[m,256,1],[m*2,m*512]),row(dg,128)]
        self.p=L.Struct([*maps,d['ds'],d['gi'],d['bi'],d['go'],d['bo'],dg,dwg,dwp,dgo,dbo,self.partw,self.partln,self.counts,m,n,m//64])
        self.inputs=(dy,tri,self.wpt,self.wgt)

class B7(Base):
    def __init__(self,d,dy,dl,dr,dg,count=132,splits=6,part=2,slices=2,defines=None):
        self.defines=tuple(sorted((dict(DEFINES) if defines is None else defines).items()))
        self.d=d;self.count,self.splits,self.part=count,splits,part;self.reduce_size=131328
        self.pipe=bool(dict(self.defines).get('PIPE_DW'));self.threads=384 if self.pipe else 256
        self.k,self.reduce,self.smem=build('b7',count,splits,part,slices,defines=self.defines)
        x=d['x'];m=d['n']**2
        self.dx=torch.empty((m,128),device=x.device,dtype=x.dtype);self.dw=torch.empty((4,128,256),device=x.device,dtype=x.dtype)
        self.dgam=torch.empty(128,device=x.device);self.dbeta=torch.empty_like(self.dgam)
        groups=16 if self.pipe and not dict(self.defines).get('PIPE_HIDDEN64') else 8
        self.partw=torch.zeros((groups,splits*slices,16384),device=x.device);self.partln=torch.empty((count-groups*splits,256),device=x.device)
        self.counts=torch.zeros(2+count,device=x.device,dtype=torch.int32)
        self.outputs=(self.dx,*self.dw.unbind(),self.dgam,self.dbeta)
        self.mask=d['mask'].bfloat16().reshape(-1);self.bind(dl,dr,dg,dy)
    def bind(self,dl,dr,dg,dy):
        d=self.d;n=d['n'];m=n*n;L=T._launch_module()
        tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
        row=lambda t:tm(t,[64,64],[128,m],[256])
        # pre carries native packed W1. xn is reserved unless tiled dW output is selected.
        wl,wlg,wr,wrg,wg=d['wt'];self.weights=(wl,wlg,wr,wrg,wg)
        maps=[tm(dl,[64,32 if self.pipe else 64],[m,256],[m*2]),tm(dr,[64,32 if self.pipe else 64],[m,256],[m*2]),tm(d['w1'],[64,64],[128,1024],[256]),(tm(self.dw,[64,16],[256,512],[512]) if dict(self.defines).get('DW_REDUCE_TMA') else row(d['x'])),row(dg),
              *[tm(w,[64,64],[256,128],[512]) for w in (wlg,wl,wrg,wr)],tm(wg,[64,64],[128,128],[256]),row(d['x']),row(dy),row(self.dx)]
        self.p=L.Struct([*maps,self.mask,None,None,d['gi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.dx,self.dx,m,n,m//64,d['bi']])
        self.inputs=(dl,dr,dg,dy,d['w1'],*self.weights)
