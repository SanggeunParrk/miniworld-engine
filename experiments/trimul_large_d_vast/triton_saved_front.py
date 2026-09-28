"""Fused FP32-accumulator gate plus channel-major preactivation saves."""
import triton
import triton.language as tl
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

@triton.jit
def saved_front(W,X,PRE,AB,MASK,M:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BN+tl.arange(0,BN)
    cols=tl.program_id(1)*BM+tl.arange(0,BM)
    kk=tl.arange(0,BK)
    acc=tl.full((BN,BM),0,tl.float32)
    for k in range(D//BK):
        w=tl.load(W+rows[:,None]*D+(k*BK+kk[None,:]))
        x=tl.load(X+(k*BK+kk[:,None])+cols[None,:]*D)
        acc=tl.dot(w,x,acc)
    tl.store(PRE+rows[:,None]*M+cols[None,:],acc)
    both=acc.reshape((BN//64,2,32,BM)).trans(0,2,3,1)
    gate,proj=tl.split(both)
    gate=gate.reshape((BN//2,BM));proj=proj.reshape((BN//2,BM))
    tanh=tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;',constraints='=f,f',args=[gate*0.5],dtype=tl.float32,is_pure=True,pack=1)
    sig=tl.fma(tanh,0.5,0.5)
    mask=tl.load(MASK+cols)
    val=(sig*proj)*mask[None,:]
    outrows=tl.program_id(0)*(BN//2)+tl.arange(0,BN//2)
    tl.store(AB+outrows[:,None]*M+cols[None,:],val)

class TritonFront:
    def __init__(self,plan,bm=128,bn=128,bk=64,warps=4,stages=3):
        old=plan.f.front;self.__dict__.update(old.__dict__)
        assert plan.p.D==512 and not old.normalize and plan.p.n==384
        self.plan=plan;self.cfg=(bm,bn,bk,warps,stages)
    def __call__(self):
        p=self.plan.p;bm,bn,bk,warps,stages=self.cfg
        F.normalize_into(self.xn,self.x,self.gi,self.bi)
        saved_front[(8*p.D//bn,p.M//bm)](self.plan.f.w,self.xn,self.plan.pre,self.ab,self.plan.f.mask,p.M,p.D,bm,bn,bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
