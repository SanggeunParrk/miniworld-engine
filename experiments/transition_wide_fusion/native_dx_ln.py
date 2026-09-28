import functools
from pathlib import Path
import subprocess
import sys
import torch
from common import wide

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/transition_fused'))
import drv


@functools.lru_cache(None)
def build(d,tbk,stages,stgdx,xb):
    if d==384 and stgdx:
        raise ValueError('D384 TMA-staged output failed numerical validation; this layout is not supported')
    smem=stages*(64+d)*tbk*2 + 68*d + 4096 + (128*d if stgdx else 0) + 128
    if smem>231424:
        raise ValueError(f'configuration exceeds Hopper shared memory: {smem}')
    out=ROOT/'.bench/transition-wide-local/cache/bounded-dxln'
    out.mkdir(parents=True,exist_ok=True)
    name=f'd{d}_k{tbk}_s{stages}_st{stgdx}_xb{xb}'
    cubin=out/(name+'.cubin')
    src=Path(__file__).with_name('dx_ln_bounded.cu')
    cuda=ROOT/'src/miniworld_engine/kernels/transition/cuda'
    cmd=['nvcc','-std=c++17','-O3','-arch=sm_90a','--cubin','-Xptxas=-v',
         '-I'+str(cuda/'anthropic_v5'),'-I'+str(cuda/'wide/kernels'),
         f'-DDW={d}','-DCOLS=2',f'-DTBK={tbk}',f'-DNSTAGE={stages}',
         '-DWAIT0=1',f'-DSTGDX={stgdx}',f'-DXB={xb}',str(src),'-o',str(cubin)]
    if not cubin.exists() or cubin.stat().st_mtime<src.stat().st_mtime:
        result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(name+'.build.txt')).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout)
    return drv.Kernel(str(cubin),'dxn_lnbwd',smem)


def dx_ln(dab,wab,x,dy,gamma,rstd,c1,tbk=64,stages=2,stgdx=0,xb=1):
    m,d=x.shape
    kernel=build(d,tbk,stages,stgdx,xb)
    ctas=torch.cuda.get_device_properties(x.device).multi_processor_count
    wt=wab.t().contiguous()
    dx=torch.empty_like(x)
    pg=torch.empty((ctas,d),device=x.device,dtype=torch.float32)
    pb=torch.empty_like(pg)
    maps=[drv.TensorMap(dab,[8*d,m],16*d,[tbk,64],swizzle=2*tbk),
          drv.TensorMap(wt,[8*d,d],16*d,[tbk,d//2],swizzle=2*tbk),
          drv.TensorMap(x,[d,m],2*d,[64,64]),drv.TensorMap(dx,[d,m],2*d,[64,64])]
    kernel((ctas,1,1),(384,1,1),*maps,x,dy,dx,gamma,rstd,c1,pg,pb,m)
    return dx,pg.sum(0),pb.sum(0),kernel


def candidate(v,config):
    x,gamma,beta,wa,wb,ws,dy=v
    y,xn,rstd,c1,_=wide._fwd_launch(x,gamma,beta,wa,wb,ws,1e-5,True)
    hid,dab=wide._ext_for(x).gate(xn,dy,wide._pack(wa,wb,128),ws.t().contiguous(),True)
    dws=wide._mm_f32(dy.t(),hid)
    dwab=wide._mm_f32(dab.t(),xn)
    dx,dg,db,_=dx_ln(dab,torch.cat((wa,wb)),x,dy,gamma,rstd,c1,**config)
    return y,dx,dg,db,dwab[:4*x.shape[-1]].to(wa.dtype),dwab[4*x.shape[-1]:].to(wb.dtype),dws.to(ws.dtype)
