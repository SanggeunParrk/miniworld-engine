"""Experimental training derivative of Anthropic's native K3 (Apache-2.0).

This is F4+F567, not a complete CUDA training implementation. The existing
Triton/cuBLAS backward consumes the saved outputs. No default dispatch changes.
"""
from functools import lru_cache
from hashlib import sha256
from itertools import product
from pathlib import Path
import fcntl
import importlib
import json
import os
import shutil
import subprocess
import sys

import torch
from miniworld_engine.kernels._compile import opaque


def _upstream():
    return Path(__file__).resolve().parent / 'vendor/anthropic_v5'


@lru_cache(None)
def _launch_module():
    # Reuse the release's audited CUDA driver/TMA binding, not its inference API.
    if 'trimul_native' not in sys.modules:
        sys.path.insert(0, str(_upstream() / 'python'))
    return importlib.import_module('trimul_native.launch')


def candidates(cz=128, ch=256):
    """Explicit bounded schedule space; invalid resource combinations pruned.

    BI/BJ, ring depth, accumulator overlap, register partition and LN scheduling
    are independent tuning axes. This persistent row grid has no GROUP_M axis:
    each CTA computes every output channel of its rows.
    """
    for (bi, bj), slots, acc, regs, serial in product(
            ((1,64), (2,64), (1,128)), (4,6,8), (1,2), (232,240), (0,1)):
        cfg=(bi,bj,slots,acc,regs,serial)
        try:
            validate_config(cz,ch,cfg)
        except ValueError:
            continue
        yield cfg


def validate_config(cz, ch, cfg):
    bi,bj,slots,acc,regs,serial=cfg
    if cz != 128 or ch not in (128,256):
        raise ValueError('First training implementation covers C=128, H=128 or 256')
    if ((bi,bj) not in ((1,64),(2,64),(1,128)) or slots not in (4,6,8)
            or acc not in (1,2) or regs not in (232,240) or serial not in (0,1)
            or (bi*bj == 64 and acc != 1)):
        raise ValueError('Invalid K3 training schedule')
    bmt=bi*bj
    bars=2*(cz//64)+2+2*slots
    smem=(bmt//64)*ch*128 + (cz//64)*bmt*128 + slots*max(cz,ch)*64
    smem+=3*8*2048+(2*cz+2*ch)*4+((bars*8+127)//128)*128
    if smem > 232448:
        raise ValueError('K3 training schedule exceeds SM90 shared memory')
    return smem


def default_config(ch):
    # Starting schedules inherited from upstream, not advertised as tuned.
    return (2,64,8 if ch == 128 else 4,2 if ch == 128 else 1,232,1)


@lru_cache(None)
def build(cz, ch, cfg):
    validate_config(cz,ch,cfg)
    source=Path(__file__).with_name('anthropic_k3_training.cu')
    inc=_upstream() / 'csrc'
    inputs=[source,inc/'tmn_kernels.cuh',inc/'tmn_ptx.cuh',inc/'common/tmn_math.cuh']
    nvcc=shutil.which('nvcc')
    if nvcc is None:
        raise RuntimeError('CUDA nvcc is required to build the experimental K3')
    version=subprocess.check_output([nvcc,'--version'],text=True)
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
           '-I'+str(inc),f'-DMWK3_CZ={cz}',f'-DMWK3_CH={ch}']
    flags += [f'-DMWK3_{k}={v}' for k,v in zip(('BI','BJ','NSLOT','NACC','REGS','LNSERIAL'),cfg)]
    hashes={str(p.relative_to(_upstream())) if p.is_relative_to(_upstream()) else p.name:sha256(p.read_bytes()).hexdigest() for p in inputs}
    key=sha256(json.dumps([hashes,flags,version],sort_keys=True).encode()).hexdigest()
    root=Path(os.environ.get('MINIWORLD_TRIMUL_TRAIN_BUILD_DIR',str(Path(__file__).resolve().parent/'build/output')))
    root.mkdir(parents=True,exist_ok=True)
    output=root/(key+'.cubin')
    with (root/(key+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not output.exists():
            tmp=output.with_suffix('.tmp.cubin')
            result=subprocess.run([nvcc,*flags,str(source),'-o',str(tmp)],capture_output=True,text=True)
            output.with_suffix('.ptxas.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError(result.stderr)
            tmp.replace(output)
            output.with_suffix('.json').write_text(json.dumps(dict(
                upstream_revision='f4f62fa6592ae4938d49b1757bea0cfeff9f468e',
                sources=hashes,flags=flags,nvcc=version,config=cfg,
                sha256=sha256(output.read_bytes()).hexdigest()),indent=2))
    return output


@lru_cache(None)
def _kernel(cz,ch,cfg,device):
    L=_launch_module()
    path=build(cz,ch,cfg)
    drv=L.BlockDriver(device=device)
    module=drv.load(path.read_bytes())
    unit=L.Unit('mw_k3_training','sm_90a',device,drv.drv,module,{},str(path))
    kernel=unit.kernel('mw_k3_train')
    kernel.set_max_dynamic_smem(validate_config(cz,ch,cfg))
    return kernel


def _fake(tri,xn,wp,wg,gamma,beta,residual,dropscale,eps,config):
    n=xn.shape[0];cz=xn.shape[-1];ch=tri.shape[0]
    return (xn.new_empty((n*n,cz)),xn.new_empty((n*n,ch)),
            xn.new_empty((n*n,),dtype=torch.float32),xn.new_empty((n*n,),dtype=torch.float32),
            xn.new_empty((n*n,cz)),xn.new_empty((n*n,cz)))


@opaque(fake=_fake,name='trimul_anthropic_k3_training')
def output_training(tri:torch.Tensor,xn:torch.Tensor,wp:torch.Tensor,wg:torch.Tensor,
                    gamma:torch.Tensor,beta:torch.Tensor,residual:torch.Tensor,dropscale:torch.Tensor,
                    eps:float,config:list[int])->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Return y, normalized X, mean, rstd, projection, gate (fresh allocations).

    tri=[H,Np,Np]; xn=[N,N,C] already normalized; Wp=[C,H], Wg=[C,C]
    in nn.Linear orientation; residual=[N*N,C]; dropout-scale=[N,C]. Input
    normalization is still the engine's shared front stage in this milestone.
    Dropout is caller-generated, shared along the first token axis. No RNG here.
    """
    if xn.ndim != 3 or tri.ndim != 3 or xn.shape[0] != xn.shape[1]:
        raise ValueError('Expected square pair xn and channel-major tri')
    n,_,cz=xn.shape;ch,np_,np2=tri.shape
    cfg=tuple(config);smem=validate_config(cz,ch,cfg)
    if n <= 0 or np_ != np2 or np_ < n or np_ % 8:
        raise ValueError('Contraction plane pitch must cover N and be divisible by 8')
    shapes=((wp,(cz,ch)),(wg,(cz,cz)),(gamma,(ch,)),(beta,(ch,)),
            (residual,(n*n,cz)),(dropscale,(n,cz)))
    if any(tuple(t.shape)!=s for t,s in shapes):
        raise ValueError('K3 training operand shape mismatch')
    bf16=(tri,xn,wp,wg,residual,dropscale)
    if any(t.dtype != torch.bfloat16 for t in bf16) or any(t.dtype != torch.float32 for t in (gamma,beta)):
        raise TypeError('BF16 activations/weights and FP32 LayerNorm parameters required')
    tensors=(*bf16,gamma,beta)
    if not xn.is_cuda or any(t.device != xn.device or not t.is_contiguous() for t in tensors):
        raise ValueError('Operands must be contiguous tensors on one CUDA device')
    if torch.cuda.get_device_capability(xn.device) != (9,0):
        raise ValueError('The experimental kernel requires SM90')
    bi,bj,slots,acc,regs,serial=cfg
    y,norm,mean,rs,proj,gate=_fake(tri,xn,wp,wg,gamma,beta,residual,dropscale,eps,config)
    with torch.cuda.device(xn.device):
        L=_launch_module();k=_kernel(cz,ch,cfg,xn.device.index)
        tm=lambda t,box,dims,strides,l2='128B':L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2=l2)
        maps=[tm(xn,[64,bj,bi],[cz,n,n],[cz*2,n*cz*2]),
              tm(tri,[64,1,64],[np_,np_,ch],[np_*2,np_*np_*2]),
              tm(wg,[64,32],[cz,cz],[cz*2],'256B'),
              tm(wp,[64,32],[ch,cz],[ch*2],'256B'),
              tm(y,[64,16,1],[cz,n,n],[cz*2,n*cz*2])]
        tiles_j=(n+bj-1)//bj;tiles=((n+bi-1)//bi)*tiles_j
        # p.residual=0 releases the prenormalized Z stage eagerly. The training
        # epilogue loads the distinct original residual through p.zres.
        base=L.Struct([*maps,gamma,beta,gamma,beta,residual,y,None,n,np_,tiles_j,tiles,0,0,float(eps),0])
        saves=[tm(norm,[64,16,1],[ch,n,n],[ch*2,n*ch*2]),
               tm(proj,[64,16,1],[cz,n,n],[cz*2,n*cz*2]),
               tm(gate,[64,16,1],[cz,n,n],[cz*2,n*cz*2]),
               tm(residual,[64,bj,bi],[cz,n,n],[cz*2,n*cz*2])]
        params=L.Struct([base,*saves,norm,proj,gate,mean,rs,dropscale])
        assert len(base.pack())==768 and len(params.pack())==1344
        sms=torch.cuda.get_device_properties(xn.device).multi_processor_count
        k.launch((min(tiles,sms),1,1),(384,1,1),[params],smem)
    return y,norm,mean,rs,proj,gate


def fused_output(tri,xn,wp,wg_transposed,gamma,beta,residual,dropscale,eps,config=None):
    """Adapter for the current TriMul autograd implementations (B=1)."""
    n=xn.shape[1];cz=xn.shape[-1];ch=tri.shape[0]
    if xn.shape[0] != 1:
        raise ValueError('First TriMul training adapter requires B=1')
    return output_training(tri,xn.reshape(n,n,cz),wp.contiguous(),wg_transposed.t().contiguous(),
                           gamma,beta,residual,dropscale,float(eps),list(config or default_config(ch)))


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
