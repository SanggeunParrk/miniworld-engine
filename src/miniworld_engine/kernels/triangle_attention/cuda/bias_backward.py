"""H100 native CUDA/TMA dK/dV with grouped shared-bias reduction.

Forward and delta retain the established implementation. dQ uses qualified CUDA. Native outputs are
projection-layout BF16 gradients; the large per-row bias buffer is eliminated.
"""
from __future__ import annotations
import functools,importlib.util,json,sys
from pathlib import Path
import torch
import triton
from miniworld_engine.kernels._compile import device_constant,opaque
from miniworld_engine.autotune.shape_key import token_key,pack

_ROOT=Path(__file__).resolve().parent
_EXT=None
SUPPORTED_LENGTHS=(384, 768, 1024)

@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data=json.loads((_ROOT/'bias_manifest.json').read_text())
        return data['python_abi']==sys.implementation.cache_tag and data['torch']==str(torch.__version__) and (_ROOT/data['binary']).is_file()
    except (OSError,ValueError,KeyError):return False

@device_constant
def _available(device):
    return torch.cuda.is_available() and _artifact_ready() and torch.cuda.get_device_capability(device)==(9,0) and 'H100' in torch.cuda.get_device_name(device)

def can_use(q,k,v,b):
    if not (q.is_cuda and q.dtype==torch.bfloat16 and q.ndim==5 and q.shape[0]==1
            and q.shape[1]==4 and q.shape[4]==32 and q.shape[2] in SUPPORTED_LENGTHS
            and q.shape[3]==q.shape[2]):return False
    L=q.shape[2]
    return (all(x.shape==q.shape and x.dtype==q.dtype and x.device==q.device
            and x.stride(4)==1 and x.stride(1)==32 and x.stride(3)==128 and x.stride(2)==L*128 for x in (q,k,v))
        and b.shape==(1,4,L,L) and b.dtype==q.dtype and b.device==q.device and _available(q.device))

def _extension():
    global _EXT
    if _EXT is None:
        data=json.loads((_ROOT/'bias_manifest.json').read_text())
        spec=importlib.util.spec_from_file_location(data['module_name'],_ROOT/data['binary'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.row_group=data.get('row_group',4);_EXT=module
    return _EXT

def native_backward(q,k,v,b,m,delta,dy):
    ext=_extension()
    return ext.backward(q,k,v,b,m,delta,dy,getattr(ext,'row_group',4))

def _backward_fake(q,k,v,b,m,out,dy,native_dq):
    """dq, dk, dv as [B, H, L, L, D] views of [B, L, L, H*D] storage, and db like b."""
    B,H,L,_,D=q.shape
    def grad():return torch.empty((B,L,L,H*D),device=q.device,dtype=q.dtype).view(B,L,L,H,D).permute(0,3,1,2,4)
    return grad(),grad(),grad(),torch.empty_like(b)

@opaque(fake=_backward_fake,name='triangle_attention_grouped_bwd_cuda')
def _backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,b:torch.Tensor,m:torch.Tensor,out:torch.Tensor,dy:torch.Tensor,native_dq:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Attention backward with the grouped bias gradient: (dq, dk, dv, db); native_dq selects the CUDA dq kernel."""
    from miniworld_engine.kernels.triangle_attention.triton import main as core
    B,H,L,_,D=q.shape
    if dy.dtype!=q.dtype:dy=dy.to(q.dtype)
    if dy.stride(4)!=1 or dy.stride(1)!=D or dy.stride(3)!=H*D or dy.stride(2)!=L*H*D:
        dy=dy.permute(0,2,3,1,4).contiguous().view(B,L,L,H,D).permute(0,3,1,2,4)
    delta=torch.empty((B,H,L,L),device=q.device,dtype=torch.float32)
    pre_grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),H*L,1]
    core._attn_bwd_preprocess[pre_grid](out,dy,delta,*out.stride(),*dy.stride(),H*L,B,L,D,
        shape_key=pack(token_key(L),HEAD_DIM=D),HEAD_DIM_PAD=D)
    dk,dv,db=native_backward(q,k,v,b,m,delta,dy)
    from miniworld_engine.kernels.triangle_attention.cuda import dq_backward
    if native_dq and dq_backward.available():
        dq=dq_backward.backward(q,k,v,b,m,delta,dy)
    else:
        dq=torch.empty((B,L,L,H*D),device=q.device,dtype=q.dtype).view(B,L,L,H,D).permute(0,3,1,2,4)
        q_grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),1,H*L]
        core._attn_bwd_dq[q_grid](q,k,v,b,D**-.5,dy,dq,m,delta,*q.stride(),*dq.stride(),*dy.stride(),*b.stride(),
            L,H*L,D,HEAD_DIM_PAD=D,shape_key=pack(token_key(L),HEAD_DIM=D))
    return dq,dk,dv,db

class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b,native_dq):
        from miniworld_engine.kernels.triangle_attention.triton import main as core
        b=b.contiguous()
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(q.shape[2]))
        ctx.save_for_backward(q,k,v,b,m,out)
        ctx.native_dq=native_dq
        return out
    @staticmethod
    def backward(ctx,dy):return (*_backward(*ctx.saved_tensors,dy,ctx.native_dq),None)

def attention(q,k,v,b,native_dq=True):return _Attention.apply(q,k,v,b,native_dq)
