"""Same stats, saved tensors and backward; vary only the b2b forward launch."""
import torch
from miniworld_engine.autotune.shape_key import both_key, rows_of
from miniworld_engine.kernels.transition.triton.wide_b2b import _forward
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
from miniworld_engine.kernels.transition.triton.fused import _fused_bwd
from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd, transition_b2b_fwd_saved


class MatchedB2B(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gamma,beta,wa,wb,ws,eps,backend,config,save_xn):
        shape=x.shape
        flat=x.reshape(-1,shape[-1]).contiguous()
        gamma,beta,wa,wb,ws=[t.to(x.dtype).contiguous() for t in (gamma,beta,wa,wb,ws)]
        key=both_key(rows_of(shape))
        rs,c1=stats_triton(flat,eps,shape_key=key)
        if backend=='cuda':
            if save_xn:y,xn=transition_b2b_fwd_saved(flat,rs,c1,gamma,beta,wa,wb,ws,config=config)
            else:
                y=transition_b2b_fwd(flat,rs,c1,gamma,beta,wa,wb,ws,config=config)
                xn=flat.new_empty(0)
        else:
            y,xn=_forward(flat,flat,gamma,beta,rs,c1,wa,wb,ws,config['BM'],config['BN'],
                           config['BK'],config['num_warps'],config['num_stages'],True,save_xn)
        ctx.save_for_backward(flat,rs,c1,gamma,beta,wa,wb,ws,xn)
        ctx.shape,ctx.eps,ctx.key,ctx.has_xn=shape,eps,key,save_xn
        return y.reshape(shape)

    @staticmethod
    def backward(ctx,dy):
        x,rs,c1,gamma,beta,wa,wb,ws,xn=ctx.saved_tensors
        grads=_fused_bwd(dy.contiguous(),x,rs,c1,gamma,beta,wa,wb,ws,
                          xn if ctx.has_xn else None,ctx.eps,ctx.has_xn,list(ctx.shape),ctx.key,True)
        return (*grads,None,None,None,None)


def matched(x,gamma,beta,wa,wb,ws,eps,*,backend,config):
    save_xn=torch.is_grad_enabled() and any(t.requires_grad for t in (x,gamma,beta,wa,wb,ws))
    return MatchedB2B.apply(x,gamma,beta,wa,wb,ws,eps,backend,config,save_xn)
