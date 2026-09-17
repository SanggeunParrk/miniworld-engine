"""Triton backward fusion: B9+B10 dual GEMM and B11+B12 LN/residual.

Input LayerNorm and its identity residual are two outputs of one autograd
Function. Their gradients meet after LN differentiation, never in dgamma/dbeta.
"""
import torch
import triton
import triton.language as tl
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key,rows_of,pack,token_key
from miniworld_engine.kernels.layernorm.triton.main import _ln_fwd


def prune_ln(configs,named_args,**meta):
    a={**named_args,**meta};n=int(a['N']);kept=[]
    for c in configs:
        bm,bk=c.kwargs['BLOCK_M1'],c.kwargs['BLOCK_K']
        if bk>triton.next_power_of_2(n):continue
        # A covering N tile has no pipelined reduction loop.
        if bk>=n and c.num_stages!=1:continue
        if c.num_warps*32>max(32,bm*bk):continue
        kept.append(c)
    return kept or list(configs)


@triton.autotune(configs=configs_for("trimul_input_ln_residual_bwd_triton"),
                 key=['shape_key', 'HAS_ROWSCALE'],
                 reset_to_zero=['DW', 'DB'], prune_configs_by={'early_config_prune': prune_ln})
@triton.jit
def _ln_bwd_residual_kernel(
    DX, DY, DW, DB, RES,
    X, W, Mean, Rstd, Rowscale,
    stride_wc, stride_bc, stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_ROWSCALE: tl.constexpr,
):
    # Map the program id to the rows of X, DX, and DY it should compute.
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M
    mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:  # bwd of y = LN(x)*rs: scale incoming grad by rs (then dx/dw/db all follow)
        rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)

    # TWO-PASS over the N tiles. dx needs c1/c2, which reduce over the WHOLE row, so once the
    # feature axis is tiled the (x, dy) tiles must be visited twice. dw/db do NOT depend on c1/c2
    # (they reduce over M, per column), so their atomics are issued in pass 1 and pass 2 only
    # re-reads x/dy and writes dx. dw/db partials are fp32 (w is cast to fp32 before use),
    # unchanged from the untiled kernel; `reset_to_zero=[DW, DB]` is kept because the autotuner
    # re-runs every candidate against these accumulators.
    #
    # COVERING TILE (BLOCK_K >= N): the two loops are single-trip but the X/DY/W loads are NOT
    # CSE'd across them (the dw/db tl.atomic_add sits between, and Triton cannot prove the raw
    # pointers do not alias), so the covering config paid 2x the read traffic instead of collapsing
    # to the untiled schedule. N and BLOCK_K are both tl.constexpr, so this guard is resolved at
    # TRACE time and only one branch is emitted.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                    mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     mask=mask, other=0).to(tl.float32)
        if HAS_ROWSCALE:
            dy = dy * rs[:, None]
        w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        wdy = tl.where(mask, w[None, :] * dy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=1) / N
        c2 = tl.sum(wdy, axis=1) / N
        # Accumulate partial sums for dw/db (column reduction over this row tile)
        tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
        tl.atomic_add(DB + cols, tl.sum(dy, axis=0), mask=col_mask)
        dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
        # Preserve the old BF16 LN gradient rounding before the residual add.
        dr = tl.load(RES + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     mask=mask, other=0).to(tl.float32)
        tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c,
                 dx.to(DX.dtype.element_ty).to(tl.float32) + dr, mask=mask)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0).to(tl.float32)
            if HAS_ROWSCALE:
                dy = dy * rs[:, None]
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            wdy = tl.where(mask, w[None, :] * dy, 0.0)
            c1 += tl.sum(xhat * wdy, axis=1)
            c2 += tl.sum(wdy, axis=1)
            # Accumulate partial sums for dw/db (column reduction over this row tile)
            tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
            tl.atomic_add(DB + cols, tl.sum(dy, axis=0), mask=col_mask)
        c1 = c1 / N
        c2 = c2 / N

        # Compute + write dx
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0).to(tl.float32)
            if HAS_ROWSCALE:
                dy = dy * rs[:, None]
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            wdy = tl.where(mask, w[None, :] * dy, 0.0)
            dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            # Preserve the old BF16 LN gradient rounding before the residual add.
            dr = tl.load(RES + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0).to(tl.float32)
            tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     dx.to(DX.dtype.element_ty).to(tl.float32) + dr, mask=mask)


def _ln_fake(dy,x,weight,mean,rstd,dr,shape_key):
    return (torch.empty_like(x),weight.new_empty(weight.shape,dtype=torch.float32),
            weight.new_empty(weight.shape,dtype=torch.float32))


@opaque(fake=_ln_fake,name='trimul_input_ln_residual_bwd')
def input_ln_residual_bwd(dy:torch.Tensor,x:torch.Tensor,weight:torch.Tensor,
                          mean:torch.Tensor,rstd:torch.Tensor,dr:torch.Tensor,
                          shape_key:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    x=x.contiguous();dy=dy.contiguous();dr=dr.contiguous()
    if x.ndim!=2 or dy.shape!=x.shape or dr.shape!=x.shape:
        raise ValueError('LN/residual backward requires equal matrix shapes')
    if x.dtype!=dy.dtype or dr.dtype!=dy.dtype:
        raise ValueError('LN/residual gradients must share the input dtype')
    m,n=x.shape
    dx=torch.empty_like(x);dw=torch.zeros(n,device=x.device,dtype=torch.float32);db=torch.zeros_like(dw)
    _ln_bwd_residual_kernel[lambda c:(triton.cdiv(m,c['BLOCK_M1']),)](
        dx,dy,dw,db,dr,x,weight,mean,rstd,rstd,
        dw.stride(0),db.stride(0),x.stride(0),x.stride(1),m,n,
        shape_key=pack(shape_key,N=n),HAS_ROWSCALE=False)
    return dx,dw,db


class InputLNResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,eps):
        flat=x.reshape(-1,x.shape[-1]).contiguous();key=both_key(rows_of(x.shape))
        y,mean,rstd=_ln_fwd(flat,weight,bias,None,eps,False,key)
        ctx.save_for_backward(flat,weight,mean,rstd);ctx.shape=x.shape;ctx.key=key
        return y.reshape(x.shape),x.view_as(x)

    @staticmethod
    def backward(ctx,dy,dr):
        x,w,mean,rstd=ctx.saved_tensors
        dx,dw,db=input_ln_residual_bwd(dy.reshape(x.shape),x,w,mean,rstd,dr.reshape(x.shape),ctx.key)
        return dx.reshape(ctx.shape),dw,db,None


def input_ln_residual(x,weight,bias,eps):
    return InputLNResidual.apply(x,weight,bias,eps)


def prune_dual(configs,named_args,**meta):
    a={**named_args,**meta};m,n=int(a['M']),int(a['N']);kept=[]
    for c in configs:
        v=c.kwargs
        if v['BLOCK_N']>triton.next_power_of_2(n):continue
        if triton.cdiv(n,v['BLOCK_N'])==1 and v['GROUP_M']!=1:continue
        if v['GROUP_M']>triton.cdiv(m,v['BLOCK_M1']):continue
        kept.append(c)
    return kept or list(configs)


@triton.autotune(configs=configs_for('trimul_input_dual_bwd_triton'),
 key=['shape_key','M','gs0','gs1','fs0','fs1','ws0','ws1','vs0','vs1'],
 prune_configs_by={'early_config_prune':prune_dual})
@triton.jit
def _input_dual_bwd_kernel(G,F,W,V,Y,M,
    KG:tl.constexpr,KP:tl.constexpr,N:tl.constexpr,
    gs0:tl.constexpr,gs1:tl.constexpr,fs0:tl.constexpr,fs1:tl.constexpr,
    ws0:tl.constexpr,ws1:tl.constexpr,vs0:tl.constexpr,vs1:tl.constexpr,
    BLOCK_M1:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr,
    GROUP_M:tl.constexpr,shape_key):
    pid=tl.program_id(0).to(tl.int64);nm=tl.cdiv(M,BLOCK_M1);nn=tl.cdiv(N,BLOCK_N)
    first=(pid//(GROUP_M*nn))*GROUP_M;gm=tl.minimum(nm-first,GROUP_M)
    local=pid%(GROUP_M*nn);pm=first+local%gm;pn=local//gm
    rows=pm*BLOCK_M1+tl.arange(0,BLOCK_M1);cols=pn*BLOCK_N+tl.arange(0,BLOCK_N)
    rk=tl.arange(0,BLOCK_K);ag=tl.zeros((BLOCK_M1,BLOCK_N),tl.float32)
    for step in range(tl.cdiv(KG,BLOCK_K)):
        k=step*BLOCK_K+rk
        a=tl.load(G+rows[:,None]*gs0+k[None,:]*gs1,(rows[:,None]<M)&(k[None,:]<KG),0)
        b=tl.load(W+k[:,None]*ws0+cols[None,:]*ws1,(k[:,None]<KG)&(cols[None,:]<N),0)
        ag=tl.dot(a,b,ag)
    gate_grad=ag.to(Y.dtype.element_ty)
    af=tl.zeros((BLOCK_M1,BLOCK_N),tl.float32)
    for step in range(tl.cdiv(KP,BLOCK_K)):
        k=step*BLOCK_K+rk
        a=tl.load(F+rows[:,None]*fs0+k[None,:]*fs1,(rows[:,None]<M)&(k[None,:]<KP),0)
        b=tl.load(V+k[:,None]*vs0+cols[None,:]*vs1,(k[:,None]<KP)&(cols[None,:]<N),0)
        af=tl.dot(a,b,af)
    tl.store(Y+rows[:,None]*N+cols[None,:],af+gate_grad.to(tl.float32),
             (rows[:,None]<M)&(cols[None,:]<N))


def _dual_fake(g,f,w,v,length):return g.new_empty((g.shape[0],w.shape[1]))


@opaque(fake=_dual_fake,name='trimul_input_dual_bwd')
def input_dual_bwd(g:torch.Tensor,f:torch.Tensor,w:torch.Tensor,v:torch.Tensor,
                   length:int)->torch.Tensor:
    if any(t.ndim!=2 for t in (g,f,w,v)):raise ValueError('dual dgrad expects matrices')
    m,kg=g.shape;kp=f.shape[1];n=w.shape[1]
    if f.shape[0]!=m or w.shape[0]!=kg or v.shape!=(kp,n):raise ValueError('dual dgrad shape mismatch')
    if any(t.dtype!=torch.bfloat16 or t.device!=g.device for t in (g,f,w,v)):
        raise ValueError('dual dgrad expects BF16 on the same device')
    out=_dual_fake(g,f,w,v,length)
    _input_dual_bwd_kernel[lambda c:(triton.cdiv(m,c['BLOCK_M1'])*triton.cdiv(n,c['BLOCK_N']),)](
        g,f,w,v,out,m,kg,kp,n,*g.stride(),*f.stride(),*w.stride(),*v.stride(),
        shape_key=token_key(length,KG=kg,KP=kp,N=n))
    return out
