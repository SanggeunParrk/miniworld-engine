"""E151 private Dense atom Function; native dispatch remains unchanged.

Original attention and transition operations keep their saved activations and
parameter gradients. The attention AdaLN backward folds the original fanout sums
into its final stores, preserving eager and compiled rounding independently of
CUDA graph capture. This entry requires the physical atom training shape.
"""
from __future__ import annotations
import os
import torch
from torch.autograd.function import once_differentiable
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.integrations import adaln_sm80 as ad
from miniworld_engine.integrations import augattn_sm80 as aa
from miniworld_engine.integrations import conditioned_transition_sm80 as ct

def _bwd_fake(dy,x,cond,lnw,ws,sb,wb,aff,cst,sbt,xst,wcat,res,dc_transition,dc_attention,masters,native,compiled):
    gd=torch.float32 if masters else ws.dtype
    return [torch.empty_like(x),torch.empty_like(cond),torch.empty_like(lnw),torch.empty(ws.shape,device=ws.device,dtype=gd),torch.empty(sb.shape,device=sb.device,dtype=gd),torch.empty(wb.shape,device=wb.device,dtype=gd)]

@opaque(fake=_bwd_fake,name='atom_dit_sm80_adaln_shared_fanout_bwd')
def _bwd(dy: torch.Tensor,x: torch.Tensor,cond: torch.Tensor,lnw: torch.Tensor,ws: torch.Tensor,sb: torch.Tensor,wb: torch.Tensor,aff: torch.Tensor,cst: torch.Tensor,sbt: torch.Tensor,xst: torch.Tensor,wcat: torch.Tensor,res: torch.Tensor,dc_transition: torch.Tensor,dc_attention: torch.Tensor,masters: bool,native: bool,compiled: bool) -> list[torch.Tensor]:
    from miniworld_engine.kernels.adaln.cuda import atom_adaln_join_shared_sm80 as fused
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_atom_cond_wgrad_fused_sm80 as wg
    if native:
        from miniworld_engine.kernels.parameter_cast.cuda.sm80 import result
        ws,sb,wb=result((ws,sb,wb))
    dx,dc,dsc,aff,psb,plw=fused.result(dy,x,cond,xst,cst,lnw,ws,wb,sb,res,dc_transition,dc_attention,not compiled)
    if os.environ.get('MINIWORLD_ADALN_SM80_ATOM_COND_WGRAD_FUSED','0')=='1':
        dws,dwb=wg.result_two(dsc,dy,aff)
    else:
        dws,dwb=ad._wgrad(dsc,aff),ad._wgrad(dy,aff)
    dsb,dlnw,dws,dwb=ad._kernels().finish([psb,plw],[ad.like(sb,masters),lnw],[dws,dwb],[ad.like(ws,masters),ad.like(wb,masters)])
    return [dx,dc,dlnw,dws,dsb,dwb]

class _Whole(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,cond,pair,kmask,kind,compiled,eps_ax,eps_ac,eps_tx,eps_tc,*params):
        a,b,n,d=x.shape
        aps=list(params[:4]);pps=list(params[4:14]);tps=list(params[14:])
        assert len(params)==23 and len(tps)==9
        masters=aps[1].dtype is torch.float32
        native=masters
        xs=x.reshape(-1,d);cs=cond.reshape(-1,d)
        normalized,*saved_a=ad.train_fwd(xs,cs,*aps,eps_ax,eps_ac,native,False)
        normalized=normalized.view(x.shape)
        attended,*saved_p=aa._fwd(normalized,cond,pair,x,kmask,None,pps,a,b,n,4,kind,True)
        result,*saved_t=ct.train_fwd(attended.reshape(-1,d),cs,*tps,eps_tx,eps_tc,True,native,False)
        ctx.save_for_backward(x,cond,pair,*params,normalized,attended,*saved_a,*saved_p,*saved_t)
        ctx.meta=(kmask,kind,compiled,a,b,n,d,masters,native)
        return result.view(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        v=ctx.saved_tensors
        x,cond,pair=v[:3];params=list(v[3:26]);normalized,attended=v[26:28]
        saved_a=list(v[28:33]);saved_p=list(v[33:40]);saved_t=list(v[40:])
        assert len(saved_a)==5 and len(saved_p)==7 and len(saved_t)==10
        kmask,kind,compiled,a,b,n,d,masters,native=ctx.meta
        aps=params[:4];pps=params[4:14];tps=params[14:]
        xs=x.reshape(-1,d);cs=cond.reshape(-1,d)
        tgrads=ct.train_bwd(dy.contiguous().reshape(-1,d),attended.reshape(-1,d),cs,*tps,*saved_t,True,masters,native)
        res,dc_transition=tgrads[:2]
        pgrads=aa._bwd(res.view(x.shape),normalized,cond,pair,kmask,pps,saved_p,a,b,n,4,kind)
        normdy,dc_attention,dpair=pgrads[:3]
        agrads=_bwd(normdy.contiguous().reshape(-1,d),xs,cs,*aps,*saved_a,res,dc_transition,dc_attention.contiguous().reshape(-1,d),masters,native,compiled)
        return (agrads[0].view(x.shape),agrads[1].view(cond.shape),dpair,None,None,None,None,None,None,None,*agrads[2:],*pgrads[3:],*tgrads[2:])

def block(module,single,cond,pair,mask=None):
    assert single.shape[:2]==(48,1) and single.shape[2] in (4096,8192)
    assert single.shape[-1]==128 and cond.shape==single.shape and cond.dtype is single.dtype is torch.bfloat16
    assert pair.shape==(1,single.shape[2],single.shape[2],16)
    assert os.environ.get('MINIWORLD_CONDTRANS_SM80_SAVE_PARAM_CAST','0')=='0'
    assert os.environ.get('MINIWORLD_ADALN_SM80_SAVE_PARAM_CAST','0')=='0'
    ada=module.attention.ada_ln_in
    aps=[ada.ln_cond.weight,ada.to_scale.weight,ada.to_scale.bias,ada.to_bias.weight]
    kind=aa._mask_kind(mask,*single.shape[:3])
    kmask=None if kind==0 else (mask[0] if mask.ndim==3 and kind==1 else mask)
    eps=(float(ada.ln_in.eps),float(ada.ln_cond.eps),float(module.transition.ada_ln_in.ln_in.eps),float(module.transition.ada_ln_in.ln_cond.eps))
    return _Whole.apply(single,cond,pair,kmask,kind,torch.compiler.is_compiling(),*eps,*aps,*aa._params(module.attention),*ct._params(module.transition))
