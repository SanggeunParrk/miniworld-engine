"""Explicit experimental module; production dispatch is unchanged.

D256 retains the existing fused CUDA tail. D384/D512 use a persistent
LayerNorm backward + residual kernel after the existing cuBLAS input GEMM.
"""
import torch
from miniworld_engine.modules import Transition
from miniworld_engine.kernels._compile import opaque
from common import wide
from ln_residual import ln_residual

CONFIGS={384:dict(bm=4,warps=4,waves=4),512:dict(bm=4,warps=4,waves=8)}


@opaque(fake=wide._bwd_launch_fake,name='transition_local_ln_residual_bwd')
def _bwd(dy: torch.Tensor,x: torch.Tensor,xn: torch.Tensor,rstd: torch.Tensor,c1: torch.Tensor,
         gamma: torch.Tensor,wa: torch.Tensor,wb: torch.Tensor,ws: torch.Tensor,hsaved: torch.Tensor,
         ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    d=x.shape[-1]
    if d not in CONFIGS:
        return wide._bwd_launch(dy,x,xn,rstd,c1,gamma,wa,wb,ws,hsaved)
    have_h=hsaved.shape[0]==x.shape[0]
    hid,dab=wide._ext_for(x).gate(xn,dy,wide._pack(wa,wb,128),ws.t().contiguous(),not have_h)
    if have_h:
        hid=hsaved
    dws=wide._mm_f32(dy.t(),hid)
    dwab=wide._mm_f32(dab.t(),xn)
    dxn=dab@torch.cat((wa,wb))
    dx,dg,db,_=ln_residual(dxn,x,dy,gamma,rstd,c1,**CONFIGS[d])
    # custom_op forbids even disjoint output views sharing one allocation.
    return dx,dg,db.clone(),dwab[:4*d],dwab[4*d:].clone(),dws


class _Experimental(wide._WideTransitionSM90A):
    @staticmethod
    def backward(ctx,dy):
        x,xn,rstd,c1,gamma,wa,wb,ws,h=ctx.saved_tensors
        grads=_bwd(dy.reshape(-1,dy.shape[-1]).contiguous(),x,xn,rstd,c1,gamma,wa,wb,ws,h)
        dx,dg,db,dwa,dwb,dws=grads
        gdt,bdt,adt,bwdt,sdt=ctx.param_dtypes
        return dx.reshape(ctx.shape),dg.to(gdt),db.to(bdt),dwa.to(adt),dwb.to(bwdt),dws.to(sdt),None


class TransitionCandidate(Transition):
    def forward(self,x):
        assert x.dtype==torch.bfloat16 and self.n==4 and x.shape[-1] in (256,384,512)
        args=(x,self.ln_in.weight,self.ln_in.bias,self.expand_a.weight,
              self.expand_b.weight,self.squeeze.weight,self.ln_in.eps)
        if not torch.is_grad_enabled():
            return wide.transition_wide_sm90a(*args)
        return _Experimental.apply(*args)
