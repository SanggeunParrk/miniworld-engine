"""Explicit, independently configured CUDA streamed-K/full-K training variants.

New saved-xn contract; deliberately separate from legacy fused-LN dispatch.
"""
from functools import lru_cache
from hashlib import sha256
from itertools import product
from pathlib import Path

import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension, mathdx_includes

VARIANTS = ('streamed_k', 'full_k')
WIDTHS = (128, 256, 384, 512)


def validate(variant, width, config):
    if variant not in VARIANTS or width not in WIDTHS:
        raise ValueError('expected streamed_k/full_k and D128/256/384/512')
    if set(config) != {'bk','bn','bo','mgroups','ngroups','stages','min_blocks'}:
        raise ValueError('incomplete CUDA variant configuration')
    if (config['bk'] not in (64,128,256,384,512) or config['bn'] not in (32,64,128)
        or config['bo'] not in (64,128) or config['mgroups'] not in (1,2)
        or config['ngroups'] not in (1,2) or config['stages'] not in (1,2,3)
        or config['min_blocks'] not in (1,2)):
        raise ValueError('unsupported TMA/WGMMA layout configuration')
    if variant == 'full_k' and config['bk'] not in (width,1 << (width-1).bit_length()):
        raise ValueError('full_k requires whole K, optionally power-of-two padded')
    if variant == 'streamed_k' and config['bk'] >= width:
        raise ValueError('streamed_k requires BK < D')


def candidates(variant, width, *, backward=False):
    """Search axes; CUDA resource failures remain visible in tuning records.

    BM = 64 * mgroups. BO and ngroups split the forward output accumulator;
    backward uses one output group and has no squeeze. Its BO is canonicalized
    to 64 rather than benchmarking duplicate gate kernels. Both kernels use a
    one-dimensional grid of row tiles, so GROUP_M has no tile-order meaning.
    """
    bks=tuple(sorted({width,1 << (width-1).bit_length()})) if variant=='full_k' else tuple(k for k in (64,128,256) if k<width)
    for bk,bn,bo,mg,ng,st,mb in product(bks,(32,64,128),(64,) if backward else (64,128),(1,2),(1,) if backward else (1,2),(1,2,3),(1,2)):
        cfg=dict(bk=bk,bn=bn,bo=bo,mgroups=mg,ngroups=ng,stages=st,min_blocks=mb)
        validate(variant,width,cfg)
        yield cfg


@lru_cache(None)
def _build(variant, width, items):
    cfg=dict(items);validate(variant,width,cfg);ensure_cuda_home()
    source=Path(__file__).with_name('transition_variants_kernel.cu')
    fingerprint=sha256(source.read_bytes()).hexdigest()[:12]
    suffix='_'.join(f'{k}{v}' for k,v in items)
    flags={**cfg,'d':width,'full':int(variant=='full_k')}
    return load_extension(name=f'transition_v2_{variant}_d{width}_{suffix}_{fingerprint}',
        sources=[str(source)],extra_cflags=['-std=c++17'],
        extra_cuda_cflags=[*host_flags(),'-std=c++17','-O3','--use_fast_math',
            '--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo',
            *gencodes('90a'),*mathdx_includes(),
            *[f'-DMWV_{k.upper()}={v}' for k,v in flags.items()],
            '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__',
            '-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT162_OPERATORS__'],verbose=False)


def extension(variant, width, config):
    validate(variant,width,config)
    return _build(variant,width,tuple(sorted(config.items())))


def _forward_fake(xn,residual,wa,wb,ws,variant,config):
    return torch.empty_like(xn)


@opaque(fake=_forward_fake,name='transition_variant_cuda_fwd')
def forward(xn:torch.Tensor,residual:torch.Tensor,wa:torch.Tensor,wb:torch.Tensor,ws:torch.Tensor,
            variant:str,config:list[int])->torch.Tensor:
    return extension(variant,xn.shape[-1],dict(zip(('bk','bn','bo','mgroups','ngroups','stages','min_blocks'),config))).forward(xn,residual,wa,wb,ws)


def _gate_fake(xn,wa,wb,dh,variant,config):
    return torch.empty_like(dh),dh.new_empty((dh.shape[0],2*dh.shape[1]))


@opaque(fake=_gate_fake,name='transition_variant_cuda_gate_bwd')
def gate_backward(xn:torch.Tensor,wa:torch.Tensor,wb:torch.Tensor,dh:torch.Tensor,
                  variant:str,config:list[int])->tuple[torch.Tensor,torch.Tensor]:
    return tuple(extension(variant,xn.shape[-1],dict(zip(('bk','bn','bo','mgroups','ngroups','stages','min_blocks'),config))).gate_backward(xn,wa,wb,dh))


def _values(c):
    return [c[k] for k in ('bk','bn','bo','mgroups','ngroups','stages','min_blocks')]


@lru_cache(None)
def norm_extension():
    ensure_cuda_home();source=Path(__file__).with_name('transition_variant_norm.cu')
    fingerprint=sha256(source.read_bytes()).hexdigest()[:12]
    return load_extension(name=f'transition_variant_norm_{fingerprint}',sources=[str(source)],
        extra_cflags=['-std=c++17'],extra_cuda_cflags=[*host_flags(),'-std=c++17','-O3',
        '--use_fast_math','-lineinfo',*gencodes('90a')],verbose=False)


def _ln_fwd_fake(x,gamma,beta,eps,config):
    return torch.empty_like(x),x.new_empty((x.shape[0],),dtype=torch.float32),x.new_empty((x.shape[0],),dtype=torch.float32)


@opaque(fake=_ln_fwd_fake,name='transition_variant_cuda_ln_fwd')
def ln_forward(x:torch.Tensor,gamma:torch.Tensor,beta:torch.Tensor,eps:float,config:list[int])->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    return tuple(norm_extension().forward(x,gamma,beta,eps,config[0]))


def _ln_bwd_fake(dy,x,gamma,mean,rs,residual,config):
    return torch.empty_like(x),torch.empty_like(gamma),torch.empty_like(gamma)


@opaque(fake=_ln_bwd_fake,name='transition_variant_cuda_ln_bwd')
def ln_backward(dy:torch.Tensor,x:torch.Tensor,gamma:torch.Tensor,mean:torch.Tensor,rs:torch.Tensor,
                residual:torch.Tensor,config:list[int])->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    return tuple(norm_extension().backward(dy,x,gamma,mean,rs,residual,*config))


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gamma,beta,eps,config):
        ctx.shape=x.shape;ctx.config=config;x=x.reshape(-1,x.shape[-1]).contiguous()
        y,mu,rs=ln_forward(x,gamma,beta,eps,config)
        ctx.save_for_backward(x,gamma,mu,rs)
        return y.reshape(ctx.shape),x.view(ctx.shape)

    @staticmethod
    def backward(ctx,dy,residual):
        x,gamma,mu,rs=ctx.saved_tensors
        dx,dg,db=ln_backward(dy.reshape_as(x).contiguous(),x,gamma,mu,rs,residual.reshape_as(x).contiguous(),ctx.config)
        return dx.reshape(ctx.shape),dg,db,None,None


class _Transition(torch.autograd.Function):
    @staticmethod
    def forward(ctx,xn,wa,wb,ws,residual,variant,fconfig,bconfig):
        ctx.shape=xn.shape;ctx.variant=variant;ctx.bconfig=bconfig
        xn=xn.reshape(-1,xn.shape[-1]).contiguous()
        ctx.save_for_backward(xn,wa,wb,ws)
        return forward(xn,residual.reshape_as(xn),wa,wb,ws,variant,fconfig).reshape(ctx.shape)

    @staticmethod
    def backward(ctx,dy):
        xn,wa,wb,ws=ctx.saved_tensors;dy=dy.reshape_as(xn).contiguous()
        dh=torch.mm(dy,ws)
        h,dab=gate_backward(xn,wa,wb,dh,ctx.variant,ctx.bconfig)
        dws=torch.mm(dy.T,h);dwab=torch.mm(dab.T,xn)
        dxn=torch.mm(dab,torch.cat((wa,wb),dim=0))
        return dxn.reshape(ctx.shape),dwab[:wa.shape[0]],dwab[wa.shape[0]:],dws,dy.reshape(ctx.shape),None,None,None


def transition(x,gamma,beta,wa,wb,ws,eps=1e-5,*,variant,forward_config,backward_config,norm_config=(4,4,8,256,4)):
    """Explicit experimental SM90 entry, including residual and all gradients.

    Activations and projection weights must be contiguous BF16; gamma/beta are
    FP32. Shapes are x=[...,D], Wa/Wb=[4D,D], Ws=[D,4D]. Both schedules save xn
    for gate recomputation. h is never materialized in forward HBM. Forward and
    backward configs are independent; no unmeasured automatic dispatch is used.
    norm_config is (warps, resident waves, vector bytes, reduction threads,
    channels per reduction CTA).
    """
    validate(variant,x.shape[-1],forward_config);validate(variant,x.shape[-1],backward_config)
    xn,residual=_Norm.apply(x,gamma,beta,eps,list(norm_config))
    return _Transition.apply(xn,wa,wb,ws,residual,variant,_values(forward_config),_values(backward_config))
