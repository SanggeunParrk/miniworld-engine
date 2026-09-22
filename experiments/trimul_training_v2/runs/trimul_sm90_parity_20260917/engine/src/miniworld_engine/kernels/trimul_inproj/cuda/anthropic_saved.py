"""Anthropic SM90 implementation under the existing training fusion/save policy.

F1 and F4 stay separate LayerNorms. The derived K1 consumes x_n and writes a/b
and interleaved preactivations. Derived K3 implements ONLY F567, consuming both
saved normalized inputs and saving projection/gate. Existing backward is reused.
No native-recomputation backward or changed activation checkpoint policy here.
"""
from functools import lru_cache
from hashlib import sha256
from itertools import product
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess

import torch
from miniworld_engine.kernels._compile import opaque
from .anthropic_training import _upstream, _launch_module, validate_config as _output_smem
from .anthropic_training import candidates as _output_candidates


def output_smem(c,h,cfg):
    if len(cfg)!=5:raise ValueError('F567 schedule is BI,BJ,slots,accumulators,registers')
    return _output_smem(c,h,(*cfg,1))


def output_candidates(c=128,h=256):
    # No LN here: do not expose the old no-op LNSERIAL tuning axis.
    yield from sorted(set(cfg[:5] for cfg in _output_candidates(c,h)))


def output_default(h):
    return (2,64,4,1,232)


def front_smem(c, h, cfg):
    bi,bj,slots,skch,sched=cfg
    if c != 128 or h not in (128,256):
        raise ValueError('Saved-policy SM90 front currently supports C128/H128 or H256')
    if (bi,bj) not in ((1,64),(2,64),(1,128),(3,64),(4,64),(2,128)) or slots not in (2,4,6,8) or skch not in (1,2) or sched not in (0,1):
        raise ValueError('Invalid saved front schedule')
    groups=bi*bj//64;blocks=2 if groups==1 else 1;spb=(c//64)//skch
    if slots<2*spb or (sched==0 and slots<3*spb and slots<(h//16)*spb):
        raise ValueError('Weight ring too small for requested schedule')
    smem=bi*bj*c*2+slots*skch*8192+3*groups*8192+2*c*4+(((2+2*slots)*8+127)//128)*128
    if smem*blocks>232448:raise ValueError('Saved front exceeds SM90 shared memory budget')
    return smem


def front_candidates(c=128,h=256):
    for tile,slots,skch,sched in product(((1,64),(2,64),(1,128),(3,64),(4,64),(2,128)),(2,4,6,8),(1,2),(0,1)):
        cfg=(*tile,slots,skch,sched)
        try:front_smem(c,h,cfg)
        except ValueError:continue
        yield cfg


def front_default(h):
    return (2,64,6,1,0) if h == 256 else (2,64,4,2,0)


@lru_cache(None)
def build(kind,c,h,cfg):
    smem=front_smem(c,h,cfg) if kind=='front' else output_smem(c,h,cfg)
    source=Path(__file__).with_name('anthropic_saved_'+kind+'.cu')
    inc=_upstream()/'csrc'
    inputs=(source,inc/'tmn_kernels.cuh',inc/'tmn_ptx.cuh',inc/'common/tmn_math.cuh')
    nvcc=shutil.which('nvcc')
    if nvcc is None:raise RuntimeError('nvcc required for Anthropic saved-policy kernels')
    version=subprocess.check_output([nvcc,'--version'],text=True)
    tag='MWK1' if kind=='front' else 'MWK3'
    axes=('BI','BJ','NSLOT','SKCH','SCHED') if kind=='front' else ('BI','BJ','NSLOT','NACC','REGS')
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),f'-D{tag}_CZ={c}',f'-D{tag}_CH={h}']
    flags += [f'-D{tag}_{k}={v}' for k,v in zip(axes,cfg)]
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in inputs}
    key=sha256(json.dumps([hashes,flags,version],sort_keys=True).encode()).hexdigest()
    root=Path(os.environ.get('MINIWORLD_TRIMUL_SAVED_BUILD_DIR',str(Path.home()/'.cache/miniworld-engine/anthropic-trimul-saved')))
    root.mkdir(parents=True,exist_ok=True);output=root/(key+'.cubin')
    with (root/(key+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not output.exists():
            tmp=output.with_suffix('.tmp.cubin')
            p=subprocess.run([nvcc,*flags,str(source),'-o',str(tmp)],capture_output=True,text=True)
            output.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr)
            if p.returncode:raise RuntimeError(p.stderr)
            tmp.replace(output)
            output.with_suffix('.json').write_text(json.dumps(dict(kind=kind,config=cfg,smem=smem,
                upstream_revision='f4f62fa6592ae4938d49b1757bea0cfeff9f468e',sources=hashes,flags=flags,nvcc=version,
                sha256=sha256(output.read_bytes()).hexdigest()),indent=2))
    return output


@lru_cache(None)
def _kernel(kind,c,h,cfg,device):
    L=_launch_module();path=build(kind,c,h,cfg);drv=L.BlockDriver(device=device)
    module=drv.load(path.read_bytes())
    unit=L.Unit('mw_saved_'+kind,'sm_90a',device,drv.drv,module,{},str(path))
    k=unit.kernel('mw_saved_'+kind)
    k.set_max_dynamic_smem(front_smem(c,h,cfg) if kind=='front' else output_smem(c,h,cfg))
    return k


def _check(x,*ts):
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)!=(9,0):
        raise ValueError('Saved-policy Anthropic kernels require SM90')
    if any(t.dtype!=torch.bfloat16 or t.device!=x.device or not t.is_contiguous() for t in (x,*ts)):
        raise ValueError('Saved-policy tensors must be contiguous BF16 on one device')


def _front_fake(xn,w1,mask,config):
    n=xn.shape[1];h=w1.shape[0]//4
    return xn.new_empty((2*h,n,n)),xn.new_empty((4*h,n*n))


@opaque(fake=_front_fake,name='trimul_anthropic_saved_front')
def front_training(xn:torch.Tensor,w1:torch.Tensor,mask:torch.Tensor,config:list[int])->tuple[torch.Tensor,torch.Tensor]:
    if xn.ndim!=4 or xn.shape[0]!=1 or xn.shape[1]!=xn.shape[2] or xn.shape[1]%8:
        raise ValueError('Saved-policy front requires BF16 [1,N,N,C], N multiple of 8')
    _check(xn,w1)
    n,c,h=xn.shape[1],xn.shape[-1],w1.shape[0]//4;cfg=tuple(config)
    smem=front_smem(c,h,cfg)
    if tuple(w1.shape)!=(4*h,c):raise ValueError('Packed front weights must be [4H,C]')
    if tuple(mask.shape)!=(n,n) or mask.device!=xn.device or mask.dtype!=torch.float32 or not mask.is_contiguous():
        raise ValueError('Pair mask must be contiguous FP32 [N,N] on input device')
    ab,preact=_front_fake(xn,w1,mask,config)
    bi,bj,slots,skch,sched=cfg;groups=bi*bj//64;minb=2 if groups==1 else 1
    with torch.cuda.device(xn.device):
        L=_launch_module();k=_kernel('front',c,h,cfg,xn.device.index)
        tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
        mz=tm(xn,[64,bj,bi],[c,n,n],[c*2,n*c*2])
        mw=tm(w1,[64,64],[c,4*h],[c*2])
        ma=tm(ab,[64,1,32],[n,n,2*h],[n*2,n*n*2])
        mg=tm(preact,[64,1,32],[n,n,2*h],[n*2,n*n*4])
        mp=tm(preact[1],[64,1,32],[n,n,2*h],[n*2,n*n*4])
        tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
        base=L.Struct([mz,mw,ma,mask,None,None,ab,None,None,n,n,tj,tiles,1,n,1,0.0,c*n,c,0,0])
        params=L.Struct([base,mg,mp])
        assert len(base.pack())==512 and len(params.pack())==768
        sms=torch.cuda.get_device_properties(xn.device).multi_processor_count
        k.launch((min(tiles,sms*minb),1,1),(128*(groups+1),1,1),[params],smem)
    return ab,preact


def front(xn,wl,wlg,wr,wrg,*,pair_mask=None,config=None):
    n,c,h=xn.shape[1],xn.shape[-1],wl.shape[1]
    if any(tuple(w.shape)!=(c,h) for w in (wl,wlg,wr,wrg)):
        raise ValueError('Front weights must be [C,H]')
    # Traceable packing stays outside the opaque CUDA launch, as in Triton front.
    gate=torch.cat((wlg.t(),wrg.t()),0);proj=torch.cat((wl.t(),wr.t()),0)
    w1=torch.stack((gate.reshape(-1,32,c),proj.reshape(-1,32,c)),1).reshape(4*h,c)
    mask=(pair_mask.reshape(n,n).float().contiguous() if pair_mask is not None else
          torch.ones((n,n),device=xn.device,dtype=torch.float32))
    ab,preact=front_training(xn,w1,mask,list(config or front_default(h)))
    return ab[:h].unsqueeze(0),ab[h:].unsqueeze(0),preact


def _output_fake(norm,xn,wp,wg,residual,dropscale,config):
    return torch.empty_like(xn),torch.empty_like(xn),torch.empty_like(xn)


@opaque(fake=_output_fake,name='trimul_anthropic_saved_output')
def output_training(norm:torch.Tensor,xn:torch.Tensor,wp:torch.Tensor,wg:torch.Tensor,
                    residual:torch.Tensor,dropscale:torch.Tensor,config:list[int])->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if xn.ndim!=3 or xn.shape[0]!=xn.shape[1] or xn.shape[0]%8:
        raise ValueError('Saved-policy F567 requires [N,N,C], N multiple of 8')
    n,c,h=xn.shape[0],xn.shape[-1],norm.shape[-1];cfg=tuple(config)
    smem=output_smem(c,h,cfg)
    _check(xn,norm,wp,wg,residual,dropscale)
    if tuple(norm.shape)!=(n,n,h) or tuple(wp.shape)!=(c,h) or tuple(wg.shape)!=(c,c) or residual.numel()!=n*n*c or tuple(dropscale.shape)!=(n,c):
        raise ValueError('F567 operand shape mismatch')
    y,proj,gate=_output_fake(norm,xn,wp,wg,residual,dropscale,config)
    bi,bj,slots,acc,regs=cfg
    with torch.cuda.device(xn.device):
        L=_launch_module();k=_kernel('output',c,h,cfg,xn.device.index)
        tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
        mz=tm(xn,[64,bj,bi],[c,n,n],[c*2,n*c*2])
        mx=tm(norm,[64,bj,bi],[h,n,n],[h*2,n*h*2])
        mwg=tm(wg,[64,32],[c,c],[c*2]);mwp=tm(wp,[64,32],[h,c],[h*2])
        my=tm(y,[64,16,1],[c,n,n],[c*2,n*c*2])
        mp=tm(proj,[64,16,1],[c,n,n],[c*2,n*c*2]);mg=tm(gate,[64,16,1],[c,n,n],[c*2,n*c*2])
        mr=tm(residual,[64,bj,bi],[c,n,n],[c*2,n*c*2])
        tj=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tj
        base=L.Struct([mz,mx,mwg,mwp,my,None,None,None,None,residual,y,None,n,n,tj,tiles,0,0,0.0,0])
        params=L.Struct([base,my,mp,mg,mr,None,proj,gate,None,None,dropscale])
        assert len(params.pack())==1344
        sms=torch.cuda.get_device_properties(xn.device).multi_processor_count
        k.launch((min(tiles,sms),1,1),(384,1,1),[params],smem)
    return y,proj,gate


def output(norm,xn,wp,wg,residual,dropscale,n,config=None):
    c,h=xn.shape[-1],norm.shape[-1]
    y,p,g=output_training(norm.reshape(n,n,h),xn.reshape(n,n,c),wp.contiguous(),wg.t().contiguous(),
        residual,dropscale,list(config or output_default(h)))
    return y.reshape(n*n,c),p.reshape(n*n,c),g.reshape(n*n,c)
