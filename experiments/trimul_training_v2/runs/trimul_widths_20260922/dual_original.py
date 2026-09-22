import triton
import triton.language as tl
from miniworld_engine.kernels._tiles import tile_order
from miniworld_engine.autotune.configs import configs_for
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
    pm,pn=tile_order(pid,nm,nn,GROUP_M)
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

