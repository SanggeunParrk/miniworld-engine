"""Shared attention/gate autograd boundary that passes native delta directly."""
import functools,importlib.util,json,sys
from pathlib import Path
import torch
from miniworld_engine.kernels._compile import device_constant,opaque
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _fwd
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward, dq_backward, ln_backward
_ROOT=Path(__file__).resolve().parent
_EXT=None

@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data=json.loads((_ROOT/'gate_manifest.json').read_text())
        return data['torch']==str(torch.__version__) and data['python_abi']==sys.implementation.cache_tag and (_ROOT/data['binary']).is_file()
    except (OSError,ValueError,KeyError):return False

@device_constant
def _available(device):return _artifact_ready() and bias_backward._artifact_ready() and dq_backward.available()

def can_use(q,w):
    return (q.dtype==torch.bfloat16 and w.dtype==q.dtype and w.device==q.device
            and w.is_contiguous() and w.shape==(128,128) and _available(q.device))

def extension():
    global _EXT
    if _EXT is None:
        data=json.loads((_ROOT/'gate_manifest.json').read_text())
        spec=importlib.util.spec_from_file_location(data['module_name'],_ROOT/data['binary'])
        _EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXT)
    return _EXT

def _gate_backward_fake(dy,w,gate,out):
    """Three tensors like gate, and the fp32 delta [4, rows]."""
    return (torch.empty_like(gate),torch.empty_like(gate),torch.empty_like(gate),
            torch.empty((4,gate.shape[0]),device=gate.device,dtype=torch.float32))

@opaque(fake=_gate_backward_fake,name='triangle_attention_gate_delta_cuda')
def gate_backward(dy:torch.Tensor,w:torch.Tensor,gate:torch.Tensor,out:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Gate backward: three gate-shaped gradients and the fp32 softmax delta [4, rows] the attention backward consumes."""
    return tuple(extension().backward(dy,w,gate,out))

def _attention_backward_fake(q,k,v,b,m,delta,dy):
    """dq, dk, dv as [B, H, L, L, D] views of [B, L, L, H*D] storage, and db like b."""
    B,H,L,_,D=q.shape
    def grad():return torch.empty((B,L,L,H*D),device=q.device,dtype=q.dtype).view(B,L,L,H,D).permute(0,3,1,2,4)
    return grad(),grad(),grad(),torch.empty_like(b)

@opaque(fake=_attention_backward_fake,name='triangle_attention_precomputed_delta_cuda')
def attention_backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,b:torch.Tensor,m:torch.Tensor,delta:torch.Tensor,dy:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Attention backward from a precomputed softmax delta (4 heads x 32): (dq, dk, dv, db)."""
    L=q.shape[2]
    delta=delta.view(1,4,L,L)
    dy=dy.view(1,L,L,4,32).permute(0,3,1,2,4)
    dk,dv,db=bias_backward.native_backward(q,k,v,b,m,delta,dy)
    dq=dq_backward.backward(q,k,v,b,m,delta,dy)
    return dq,dk,dv,db

class AttentionGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b,gate,w):
        L=q.shape[2];b=b.contiguous()
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
        out2=out.permute(0,2,3,1,4).reshape(-1,128).contiguous()
        gate2=gate.reshape(-1,128).contiguous()
        y=_fwd(gate2,out2,w,shape_key=token_key(L))
        ctx.save_for_backward(q,k,v,b,m,gate2,w,out2)
        return y.view_as(gate)
    @staticmethod
    def backward(ctx,dy):
        q,k,v,b,m,gate,w,out=ctx.saved_tensors
        dy=dy.reshape(-1,128).contiguous()
        dr,dg,a,delta=gate_backward(dy,w,gate,out)
        dq,dk,dv,db=attention_backward(q,k,v,b,m,delta,dr)
        dw=dy.T@a if ctx.needs_input_grad[-1] else None
        return dq,dk,dv,db,dg.view(1,q.shape[2],q.shape[2],128),dw

def attention_gate(q,k,v,b,gate,w):return AttentionGate.apply(q,k,v,b,gate,w)
