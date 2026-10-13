"""E126 experimental Dense atom autograd fanout. Existing dispatcher is untouched.

Forward computation remains the original A100 attention and transition. Only fanout
gradient joins are explicit CUDA: original eager BF intermediate rounding, or the
original compiled F32 add/add with a single final BF conversion. Graph capture mode
does not select arithmetic. This module is not enabled until whole proof completes.
"""
from __future__ import annotations
import torch
from torch.autograd.function import once_differentiable
from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

def _join_fake(a,b,c,planes,compiled,cfg):
    return torch.empty_like(a)

@opaque(fake=_join_fake,name='atom_dit_sm80_gradient_join')
def _join(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,planes:int,compiled:bool,cfg:int)->torch.Tensor:
    if compiled:
        from miniworld_engine.kernels.conditioned_transition.cuda import atom_gradient_join_compiled_sm80 as k
    else:
        from miniworld_engine.kernels.conditioned_transition.cuda import atom_gradient_join_sm80 as k
    out=torch.empty_like(a)
    k.run(a.contiguous(),b.contiguous(),c.contiguous(),out,planes,cfg)
    return out

class _Two(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,compiled,cfg):
        ctx.compiled=compiled;ctx.cfg=cfg
        return x,x
    @staticmethod
    @once_differentiable
    def backward(ctx,a,b):
        if a is None:return b,None,None
        if b is None:return a,None,None
        return _join(a,b,a,2,ctx.compiled,ctx.cfg),None,None

class _Three(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,compiled,cfg):
        ctx.compiled=compiled;ctx.cfg=cfg
        # Order is transition, attention raw-conditioning, attention AdaLN.
        return x,x,x
    @staticmethod
    @once_differentiable
    def backward(ctx,a,b,c):
        fields=[v for v in (a,b,c) if v is not None]
        if not fields:return None,None,None
        if len(fields)==1:return fields[0],None,None
        if len(fields)==2:return _join(fields[0],fields[1],fields[0],2,ctx.compiled,ctx.cfg),None,None
        return _join(a,b,c,3,ctx.compiled,ctx.cfg),None,None

def block(module,single,cond,pair,mask=None):
    """Private opt-in entry for later all26-gradient proof; no primary wiring yet."""
    from miniworld_engine.integrations import augattn_sm80
    assert tuple(single.shape[:2])==(48,1) and single.shape[2] in (4096,8192)
    assert single.shape[-1]==128 and cond.shape==single.shape and single.dtype is torch.bfloat16
    assert cond.dtype is torch.bfloat16 and pair.shape==(1,single.shape[2],single.shape[2],16)
    compiled=torch.compiler.is_compiling()
    cfg2=settings.current().atom_dit_sm80_join_two_cfg
    cfg3=settings.current().atom_dit_sm80_join_three_cfg
    s_norm,s_res=_Two.apply(single,compiled,cfg2)
    c_transition,c_attention,c_norm=_Three.apply(cond,compiled,cfg3)
    normalized=module.attention.ada_ln_in(s_norm,c_norm)
    attended=augattn_sm80._fused_update(module.attention,normalized,c_attention,pair,mask,s_res)
    return module.transition(attended,c_transition)
