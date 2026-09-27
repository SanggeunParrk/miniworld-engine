"""H100 Q projection plus attention; preserve native training backward.

Set MINIWORLD_TRIATTN_Q_FWD=0 to compare with separate projections.
Only supported BF16 shapes with every required native fusion enabled qualify.
"""
import torch
from torch.nn import functional as F
from miniworld_engine.kernels._compile import opaque,device_constant
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.layernorm import compile_native as ln
from miniworld_engine.kernels.triangle_attention.cuda import ln_backward,gate_backward,wgrad_backward
from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _fwd as gate_fwd

import functools,hashlib,importlib.util,json,os,sys
from pathlib import Path
_ROOT=Path(__file__).resolve().parent
_EXT=None
SUPPORTED_LENGTHS=(384, 768, 1024)

@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data=json.loads((_ROOT/'q_fwd_manifest.json').read_text())
        return (data['python_abi']==sys.implementation.cache_tag and data['torch']==str(torch.__version__)
            and all(hashlib.sha256((_ROOT/n).read_bytes()).hexdigest()==h for n,h in data['files'].items()))
    except (OSError,ValueError,KeyError):return False

@device_constant
def _available(device):
    return (_artifact_ready() and torch.cuda.get_device_capability(device)==(9,0)
            and 'H100' in torch.cuda.get_device_name(device))

def extension():
    global _EXT
    if _EXT is None:
        data=json.loads((_ROOT/'q_fwd_manifest.json').read_text())
        spec=importlib.util.spec_from_file_location(data['module_name'],_ROOT/data['binary'])
        _EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXT)
    return _EXT

def can_use(model,pair,mask=None):
    if (os.environ.get('MINIWORLD_TRIATTN_Q_FWD','1')=='0'
            or os.environ.get('MINIWORLD_TRIATTN_TRAINING_FWD','1')=='0'):
        return False
    if not (torch.is_grad_enabled() and pair.ndim==4 and pair.shape[1] in SUPPORTED_LENGTHS
            and model.use_self_attention and not model.use_qk_norm and model.n_head==4
            and all(getattr(model,n,True) for n in ('_fuse_front_backward','_fuse_projection_backward',
                '_fuse_gate_backward','_fuse_bias_backward','_fuse_dq_backward'))):
        return False
    weights=(model.to_query.weight,model.to_key.weight,model.to_value.weight,
             model.to_gate.weight,model.to_bias.weight)
    if not ln_backward.can_use(pair,weights,model.ln_pair.weight,model.ln_pair.bias):return False
    if mask is not None and not (mask.dtype==torch.bool and mask.device==pair.device
            and mask.shape==(1,pair.shape[1])):return False
    from miniworld_engine.kernels.bias_only_attention import dispatch as gate_dispatch
    return (_available(pair.device) and gate_backward.can_use(pair,model.to_out.weight)
        and gate_dispatch.gate_use_fused(128,128,pair.shape[1]**2,pair.device,pair.dtype))



def _fake(z,wq,wk,wv,b):
    L=z.shape[1]
    def value():return torch.empty_like(z).view(1,L,L,4,32).permute(0,3,1,2,4)
    return value(),torch.empty((1,4,L,L),device=z.device,dtype=torch.float32),value(),value(),value()


@opaque(fake=_fake,name='triangle_q_projection_attention_cuda')
def native(z:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,wv:torch.Tensor,b:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return tuple(extension().forward(z,wq,wk,wv,b))


def projection_view(t):
    return t.permute(0,2,3,1,4).reshape(-1,128)


class FrontAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gamma,beta,wq,wk,wv,wg,wb,wo,mask,eps,ending):
        xx=x.transpose(1,2).contiguous() if ending else x
        z,mean,rstd=ln._dispatch_fwd(xx,gamma,beta,eps)
        gate=F.linear(z,wg).reshape(-1,128)
        b=F.linear(z,wb).permute(0,3,1,2)
        if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
        b=b.contiguous()
        out,m,q,k,v=native(z,wq,wk,wv,b)
        out2=projection_view(out)
        y=gate_fwd(gate,out2,wo,shape_key=token_key(x.shape[1]))
        ctx.save_for_backward(xx,z,mean,rstd,gamma,wq,wk,wv,wg,wb,wo,q,k,v,b,m,gate,out2,mask)
        ctx.ending=ending
        return y.view_as(z),x.view_as(x)

    @staticmethod
    def backward(ctx,dy,residual):
        x,z,mean,rstd,gamma,wq,wk,wv,wg,wb,wo,q,k,v,b,m,gate,out,mask=ctx.saved_tensors
        dy=dy.reshape(-1,128).contiguous()
        dr,dg,a,delta=gate_backward.gate_backward(dy,wo,gate,out)
        dq,dk,dv,db=gate_backward.attention_backward(q,k,v,b,m,delta,dr)
        dwo=dy.T@a if ctx.needs_input_grad[8] else None
        del dr,a,delta
        if mask is not None:db=db.masked_fill(~mask[:,None,None,:],0)
        weights=(wq,wk,wv,wg,wb)
        grads=[projection_view(g).contiguous() for g in (dq,dk,dv)]
        grads.extend((dg,db.permute(0,2,3,1).reshape(-1,4).contiguous()))
        dx,dgamma,dbeta=ln_backward._backward(grads,list(weights),x,mean,rstd,gamma,residual.contiguous(),x.shape[1],ctx.ending)
        zz=z.reshape(-1,128);needed=ctx.needs_input_grad[3:8]
        if all(needed[:4]) and wgrad_backward.can_use(grads[:4],zz):
            dw=[*wgrad_backward.backward(grads[:4],zz),grads[4].T@zz if needed[4] else None]
        else:dw=[g.T@zz if n else None for g,n in zip(grads,needed)]
        return dx,dgamma,dbeta,*dw,dwo,None,None,None

def forward(model,pair,mask=None):
    out,residual=FrontAttention.apply(pair,model.ln_pair.weight,model.ln_pair.bias,
        model.to_query.weight,model.to_key.weight,model.to_value.weight,model.to_gate.weight,
        model.to_bias.weight,model.to_out.weight,mask,model.ln_pair.eps,not model.starting)
    if not model.starting:out=out.transpose(1,2).contiguous()
    if model.p_drop>0 and model.training:out=out*model._make_drop_scale(pair,model.p_drop)
    return residual+out

