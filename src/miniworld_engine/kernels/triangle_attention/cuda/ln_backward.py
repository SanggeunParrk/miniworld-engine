"""Qualified H100 projection, LayerNorm input/parameter gradient and residual fusion."""
from __future__ import annotations
import functools, importlib.util, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
from miniworld_engine.kernels._compile import device_constant, opaque
from miniworld_engine.kernels.layernorm import compile_native as ln
from miniworld_engine.kernels.triangle_attention.cuda import can_use as _projection_can_use

_ROOT = Path(__file__).resolve().parent
_EXT = None

@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data = json.loads((_ROOT / 'ln_manifest.json').read_text())
        return (data['python_abi'] == sys.implementation.cache_tag
                and data['torch'] == str(torch.__version__)
                and (_ROOT / data['binary']).is_file())
    except (OSError, ValueError, KeyError):
        return False

@device_constant
def _available(device):
    return _artifact_ready()

def can_use(pair, weights, gamma, beta):
    return (_projection_can_use(pair, weights) and _available(pair.device)
            and all(t.dtype == torch.float32 and t.shape == (128,)
                    and t.is_contiguous() and t.device == pair.device for t in (gamma, beta)))

def extension():
    global _EXT
    if _EXT is None:
        data = json.loads((_ROOT / 'ln_manifest.json').read_text())
        spec = importlib.util.spec_from_file_location(data['module_name'], _ROOT / data['binary'])
        _EXT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXT)
    return _EXT

def _backward_fake(dy,w,x,mean,rstd,gamma,residual,L,ending):
    """dx like x; dgamma and dbeta like gamma."""
    return torch.empty_like(x),torch.empty_like(gamma),torch.empty_like(gamma)

@opaque(fake=_backward_fake,name='triangle_attention_projection_ln_residual_cuda')
def _backward(dy:list[torch.Tensor],w:list[torch.Tensor],x:torch.Tensor,mean:torch.Tensor,rstd:torch.Tensor,
              gamma:torch.Tensor,residual:torch.Tensor,L:int,ending:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Projection dgrad + LayerNorm backward + residual: (dx, dgamma, dbeta); ending transposes back."""
    dx,dg,db=extension().backward(dy,w,x,mean,rstd,gamma,residual.contiguous(),L,ending)
    return dx.view_as(x),dg,db

class Front(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gamma,beta,eps,ending,wq,wk,wv,wg,wb):
        xx=x.transpose(1,2).contiguous() if ending else x
        z,mean,rstd=ln._dispatch_fwd(xx,gamma,beta,eps)
        weights=(wq,wk,wv,wg,wb)
        ctx.save_for_backward(xx,z,mean,rstd,gamma,*weights)
        ctx.ending=ending
        return *(F.linear(z,w) for w in weights),x.view_as(x)
    @staticmethod
    def backward(ctx,dq,dk,dv,dg,db,residual):
        x,z,mean,rstd,gamma,*weights=ctx.saved_tensors
        dy=[g.reshape(-1,w.shape[0]).contiguous() for g,w in zip((dq,dk,dv,dg,db),weights)]
        dx,dgamma,dbeta=_backward(dy,weights,x,mean,rstd,gamma,residual,x.shape[1],ctx.ending)
        zz=z.reshape(-1,128)
        from miniworld_engine.kernels.triangle_attention.cuda import wgrad_backward
        needed=ctx.needs_input_grad[5:]
        if all(needed[:4]) and wgrad_backward.can_use(dy[:4],zz):
            dw=[*wgrad_backward.backward(dy[:4],zz),dy[4].T@zz if needed[4] else None]
        else:
            dw=[g.T@zz if n else None for g,n in zip(dy,needed)]
        return dx,dgamma,dbeta,None,None,*dw

def unfused_forward(model,pair,mask=None):
    B,L,_,C=pair.shape
    q,k,v,g,b,residual=Front.apply(pair,model.ln_pair.weight,model.ln_pair.bias,model.ln_pair.eps,not model.starting,
        model.to_query.weight,model.to_key.weight,model.to_value.weight,model.to_gate.weight,model.to_bias.weight)
    q,k,v=(t.view(B,L,L,4,32).permute(0,3,1,2,4) for t in (q,k,v))
    b=b.permute(0,3,1,2)
    if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
    from miniworld_engine.kernels.triangle_attention.cuda import gate_backward
    from miniworld_engine.kernels.bias_only_attention import dispatch as gate_dispatch
    fused_gate=(getattr(model, "_fuse_gate_backward", True)
        and getattr(model, "_fuse_bias_backward", True) and getattr(model, "_fuse_dq_backward", True)
        and gate_dispatch.gate_use_fused(C,model.to_out.weight.shape[0],B*L*L,pair.device,pair.dtype)
        and gate_backward.can_use(q,model.to_out.weight))
    if fused_gate:
        out=gate_backward.attention_gate(q,k,v,b,g,model.to_out.weight)
    else:
        out=model._kernel_triangle_attention(q,k,v,b,model._backend)
        out=out.permute(0,2,3,1,4).reshape(B,L,L,C)
        out=model._gate_out(g,out)
    if not model.starting:out=out.transpose(1,2).contiguous()
    if model.p_drop>0 and model.training:out=out*model._make_drop_scale(pair,model.p_drop)
    return residual+out


def forward(model,pair,mask=None):
    from miniworld_engine.kernels.triangle_attention.cuda import qkv_projection_attention
    if qkv_projection_attention.can_use(model,pair,mask):
        return qkv_projection_attention.forward(model,pair,mask)
    from miniworld_engine.kernels.triangle_attention.cuda import qg_projection_attention
    if qg_projection_attention.can_use(model,pair,mask):
        return qg_projection_attention.forward(model,pair,mask)
    from miniworld_engine.kernels.triangle_attention.cuda import q_projection_attention
    if q_projection_attention.can_use(model,pair,mask):
        return q_projection_attention.forward(model,pair,mask)
    return unfused_forward(model,pair,mask)
