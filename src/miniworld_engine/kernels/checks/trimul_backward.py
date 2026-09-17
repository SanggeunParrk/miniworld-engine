"""Independent FP32 checks including saved BF16 rounding points."""
import torch
from miniworld_engine.kernels.drivers.trimul_backward import ln_operands,dual_operands


def ln_residual():
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual_bwd
    dy,x,w,mean,rs,dr,key=args=ln_operands()
    dx,dw,db=input_ln_residual_bwd(*args)
    xhat=(x.float()-mean[:,None])*rs[:,None];u=dy.float()*w.float()
    gx=rs[:,None]*(u-u.mean(1,keepdim=True)-xhat*(u*xhat).mean(1,keepdim=True))
    return dict(dx=(dx,gx.to(x.dtype).float()+dr.float()),dw=(dw,(dy.float()*xhat).sum(0)),db=(db,dy.float().sum(0)))


def dual():
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_dual_bwd
    g,f,w,v,l=args=dual_operands();y=input_dual_bwd(*args)
    ref=(g.float()@w.float()).to(g.dtype).float()+f.float()@v.float()
    return dict(dx=(y,ref))
