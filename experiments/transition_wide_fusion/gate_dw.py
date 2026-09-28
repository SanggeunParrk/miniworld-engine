"""Streaming hidden-slice gate + ALL weight gradients, with bounded partials.

Each CTA owns BH hidden channels and one row partition. Its dWa/dWb/dWs
accumulators live across the partition; a,b,dh live for one BM row tile only.
No h is materialized. dAB is stored only for the remaining input-gradient
GEMM. FP32 partials have shape [partitions,3,H,D], independent of sequence
length. This is a distinct algorithm from row chunking followed by cuBLAS.
"""
import torch
import triton
import triton.language as tl
from common import wide, _transition_ln_bwd


@triton.jit
def _gate_dw(XN,DY,WA,WB,WS,DAB,PART,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,
             BM:tl.constexpr,BH:tl.constexpr,SPLIT:tl.constexpr):
    hs=tl.program_id(0)*BH+tl.arange(0,BH)
    part=tl.program_id(1)
    cols=tl.arange(0,BD)
    rr=tl.arange(0,BM)
    wa=tl.load(WA+cols[:,None]+hs[None,:]*D,cols[:,None]<D,0)
    wb=tl.load(WB+cols[:,None]+hs[None,:]*D,cols[:,None]<D,0)
    ws=tl.load(WS+cols[:,None]*(4*D)+hs[None,:],cols[:,None]<D,0)
    dwa=tl.full((BH,BD),0,tl.float32)
    dwb=tl.full((BH,BD),0,tl.float32)
    dws=tl.full((BH,BD),0,tl.float32)
    for base in range(part*BM,M,SPLIT*BM):
        rows=base+rr
        xn=tl.load(XN+rows[:,None]*D+cols[None,:],(rows[:,None]<M)&(cols[None,:]<D),0)
        dy=tl.load(DY+rows[:,None]*D+cols[None,:],(rows[:,None]<M)&(cols[None,:]<D),0)
        a=tl.dot(xn,wa)
        b=tl.dot(xn,wb)
        dh=tl.dot(dy,ws).to(tl.bfloat16).to(tl.float32)
        t=tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;',constraints='=f,f',
                                    args=[.5*a],dtype=tl.float32,is_pure=True,pack=1)
        s=.5*t+.5
        silu=a*s
        h=(silu*b).to(tl.bfloat16)
        da=((dh*b)*(s+silu*(1-s))).to(tl.bfloat16)
        db=(dh*silu).to(tl.bfloat16)
        tl.store(DAB+rows[:,None]*(8*D)+hs[None,:],da,rows[:,None]<M)
        tl.store(DAB+rows[:,None]*(8*D)+4*D+hs[None,:],db,rows[:,None]<M)
        dwa=tl.dot(tl.trans(da),xn,dwa)
        dwb=tl.dot(tl.trans(db),xn,dwb)
        dws=tl.dot(tl.trans(h),dy,dws)
    off=part*3*(4*D)*D+hs[:,None]*D+cols[None,:]
    tl.store(PART+off,dwa,cols[None,:]<D)
    tl.store(PART+off+(4*D)*D,dwb,cols[None,:]<D)
    tl.store(PART+off+2*(4*D)*D,dws,cols[None,:]<D)


def gate_dw(xn,dy,wa,wb,ws,bm=32,bh=32,warps=8,split=8):
    m,d=xn.shape
    dab=xn.new_empty((m,8*d))
    parts=torch.empty((split,3,4*d,d),device=xn.device,dtype=torch.float32)
    ker=_gate_dw[(4*d//bh,split)](xn,dy,wa,wb,ws,dab,parts,m,d,triton.next_power_of_2(d),bm,bh,split,
                                 num_warps=warps,num_stages=1)
    grads=parts.sum(0)
    return dab,grads[0],grads[1],grads[2].t().contiguous(),ker


def candidate(v,config):
    x,gamma,beta,wa,wb,ws,dy=v
    y,xn,rstd,c1,_=wide._fwd_launch(x,gamma,beta,wa,wb,ws,1e-5,True)
    dab,dwa,dwb,dws,_=gate_dw(xn,dy,wa,wb,ws,**config)
    wab=torch.cat((wa,wb))
    if x.shape[-1]==256:
        dx,dg,db=wide._ext_for(x).dxln(dab,wab.t().contiguous(),x,dy,gamma,rstd,c1)
    else:
        dx,dg,db=_transition_ln_bwd(dab@wab,x,rstd,c1,gamma)
        dx.add_(dy)
    return y,dx,dg,db,dwa.to(wa.dtype),dwb.to(wb.dtype),dws.to(ws.dtype)
