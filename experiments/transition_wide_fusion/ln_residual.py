"""Bounded fusion: LayerNorm backward, residual add and affine partials.

Leave the high-throughput wide GEMM with cuBLAS. Fuse its scalar consumers
instead, preserve BF16 rounding boundaries, and use deterministic partials.
"""
import torch
import triton
import triton.language as tl
from common import wide


@triton.jit
def _ln_residual(DXN,X,DY,G,RS,C1,DX,PART,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr):
    pid=tl.program_id(0)
    rr=pid*BM+tl.arange(0,BM)
    cc=tl.arange(0,BD)
    mask=(rr[:,None]<M)&(cc[None,:]<D)
    grad=tl.load(DXN+rr[:,None]*D+cc[None,:],mask,0).to(tl.float32)
    xx=tl.load(X+rr[:,None]*D+cc[None,:],mask,0).to(tl.float32)
    rs=tl.load(RS+rr,rr<M,0)
    c1=tl.load(C1+rr,rr<M,0)
    gg=tl.load(G+cc,cc<D,0)
    xhat=tl.where(cc[None,:]<D,xx*rs[:,None]-c1[:,None],0.)
    weighted=grad*gg[None,:]
    a=tl.sum(weighted*xhat,1)/D
    b=tl.sum(weighted,1)/D
    dx=(weighted-xhat*a[:,None]-b[:,None])*rs[:,None]
    if D>=384:
        dx=dx.to(tl.bfloat16).to(tl.float32)
    dy=tl.load(DY+rr[:,None]*D+cc[None,:],mask,0).to(tl.float32)
    tl.store(DX+rr[:,None]*D+cc[None,:],dx+dy,mask)
    tl.store(PART+pid*2*D+cc,tl.sum(grad*xhat,0),cc<D)
    tl.store(PART+pid*2*D+D+cc,tl.sum(grad,0),cc<D)


@triton.jit
def _ln_residual_persistent(DXN,X,DY,G,RS,C1,DX,PART,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,
                            BM:tl.constexpr,CTAS:tl.constexpr):
    pid=tl.program_id(0)
    rr0=tl.arange(0,BM)
    cc=tl.arange(0,BD)
    gg=tl.load(G+cc,cc<D,0)
    pg=tl.full((BM,BD),0,tl.float32)
    pb=tl.full((BM,BD),0,tl.float32)
    for base in range(pid*BM,M,CTAS*BM):
        rr=base+rr0
        mask=(rr[:,None]<M)&(cc[None,:]<D)
        grad=tl.load(DXN+rr[:,None]*D+cc[None,:],mask,0).to(tl.float32)
        xx=tl.load(X+rr[:,None]*D+cc[None,:],mask,0).to(tl.float32)
        rs=tl.load(RS+rr,rr<M,0)
        c1=tl.load(C1+rr,rr<M,0)
        xhat=tl.where(cc[None,:]<D,xx*rs[:,None]-c1[:,None],0.)
        pg+=grad*xhat
        pb+=grad
        weighted=grad*gg[None,:]
        a=tl.sum(weighted*xhat,1)/D
        b=tl.sum(weighted,1)/D
        dx=(weighted-xhat*a[:,None]-b[:,None])*rs[:,None]
        if D>=384:
            dx=dx.to(tl.bfloat16).to(tl.float32)
        dy=tl.load(DY+rr[:,None]*D+cc[None,:],mask,0).to(tl.float32)
        tl.store(DX+rr[:,None]*D+cc[None,:],dx+dy,mask)
    tl.store(PART+pid*2*D+cc,tl.sum(pg,0),cc<D)
    tl.store(PART+pid*2*D+D+cc,tl.sum(pb,0),cc<D)


def ln_residual(dxn,x,dy,gamma,rstd,c1,bm=16,warps=4,waves=0):
    m,d=x.shape
    dx=torch.empty_like(x)
    ctas=torch.cuda.get_device_properties(x.device).multi_processor_count*waves if waves else triton.cdiv(m,bm)
    parts=torch.empty((ctas,2,d),device=x.device,dtype=torch.float32)
    if waves:
        ker=_ln_residual_persistent[(ctas,)](dxn,x,dy,gamma,rstd,c1,dx,parts,m,d,triton.next_power_of_2(d),bm,ctas,
                                            num_warps=warps)
    else:
        ker=_ln_residual[(ctas,)](dxn,x,dy,gamma,rstd,c1,dx,parts,m,d,triton.next_power_of_2(d),bm,num_warps=warps)
    grads=parts.sum(0)
    return dx,grads[0],grads[1],ker


def candidate(v,config):
    x,gamma,beta,wa,wb,ws,dy=v
    y,xn,rstd,c1,_=wide._fwd_launch(x,gamma,beta,wa,wb,ws,1e-5,True)
    hid,dab=wide._ext_for(x).gate(xn,dy,wide._pack(wa,wb,128),ws.t().contiguous(),True)
    dws=wide._mm_f32(dy.t(),hid)
    dwab=wide._mm_f32(dab.t(),xn)
    wab=torch.cat((wa,wb))
    dxn=wide._mm_f32(dab,wab) if x.shape[-1]==256 else dab@wab
    dx,dg,db,_=ln_residual(dxn,x,dy,gamma,rstd,c1,**config)
    return y,dx,dg,db,dwab[:4*x.shape[-1]].to(wa.dtype),dwab[4*x.shape[-1]:].to(wb.dtype),dws.to(ws.dtype)
