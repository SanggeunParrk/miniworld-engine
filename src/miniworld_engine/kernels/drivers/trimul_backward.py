"""Build inputs for the two TriMul backward fusions."""
import torch
from miniworld_engine.kernels.drivers import BF16,dev,driver_width,driver_heads,driver_length,ragged
from miniworld_engine.autotune.shape_key import both_key


def ln_operands():
    n=ragged(driver_width(128));l=driver_length(64);m=ragged(l*l);kw=dict(device=dev(),dtype=BF16)
    x=torch.randn(m,n,**kw);w=torch.randn(n,**kw)*.1+1
    mean=x.float().mean(1);rstd=torch.rsqrt(x.float().var(1,unbiased=False)+1e-5)
    return torch.randn_like(x),x,w,mean,rstd,torch.randn_like(x),both_key(m)


def dual_operands():
    n=ragged(driver_width(128));l=driver_length(64);m=ragged(l*l)
    kg=n;kp=ragged(8*driver_heads(driver_width(128)));kw=dict(device=dev(),dtype=BF16)
    return (torch.randn(m,kg,**kw),torch.randn(kp,m,**kw).t(),
            torch.randn(n,kg,**kw).t()/kg**.5,torch.randn(kp,n,**kw)/kp**.5,l)


def ln_residual():
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual_bwd
    input_ln_residual_bwd(*ln_operands())


def dual():
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_dual_bwd
    input_dual_bwd(*dual_operands())
