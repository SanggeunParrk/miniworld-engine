import torch, triton
from miniworld_kernels.kernels.layernorm_linear import te_style as T
N=128
def mm(M,N):  # (M,N) tensor with m-major strides (1,M)
    return torch.randn(N,M,device="cuda",dtype=torch.bfloat16).t()
def cosf(a,b):
    a=a.float().flatten();b=b.float().flatten();return (a@b/(a.norm()*b.norm()+1e-30)).item()
def ref(dxn,x,w,mean,rstd):
    xf=x.float();dxnf=dxn.float();wf=w.float()
    xhat=(xf-mean[:,None])*rstd[:,None]; dxhat=dxnf*wf[None,:]
    c2=dxhat.mean(1,keepdim=True);c1=(dxhat*xhat).mean(1,keepdim=True)
    return rstd[:,None]*(dxhat-c2-xhat*c1)
def bench_uniform():
    res={}
    for L in (384,768,1024):
        M=L*L
        torch.manual_seed(0)
        x=mm(M,N); dxn=mm(M,N); w=torch.randn(N,device="cuda",dtype=torch.bfloat16)
        mean=torch.randn(M,device="cuda"); rstd=torch.rand(M,device="cuda")+0.5
        f=lambda: T._ln_bwd(dxn,x,w,mean,rstd,x.stride())
        for _ in range(5): f()
        torch.cuda.synchronize()
        t=triton.testing.do_bench(f,warmup=25,rep=100,return_mode='median')
        dx,_,_=f(); c=cosf(dx,ref(dxn,x,w,mean,rstd))
        res[L]=(t,str(T._ln_bwd_kernel.best_config),c)
    return res
print("=== BASELINE (maxnreg=None) ===",flush=True)
b=bench_uniform()
for L,(t,bc,c) in b.items(): print("L=%d %.4f ms cos=%.6f BEST=%s"%(L,t,c,bc),flush=True)

# --- extend configs with maxnreg caps, force re-tune ---
extra=[triton.Config({"BLOCK_M":bm},num_warps=nw,num_stages=ns,maxnreg=mr)
       for bm in (32,64,128) for nw in (4,8) for ns in (2,4) for mr in (96,128,168,200)]
T._ln_bwd_kernel.configs = T._ln_bwd_kernel.configs + extra
try: T._ln_bwd_kernel.cache.clear()
except Exception as e: print("cache clear warn",e)
print("=== WITH maxnreg configs (%d total) ==="%len(T._ln_bwd_kernel.configs),flush=True)
b2=bench_uniform()
for L,(t,bc,c) in b2.items():
    spd=b[L][0]/t
    print("L=%d %.4f ms cos=%.6f (%.3fx vs base) BEST=%s"%(L,t,c,spd,bc),flush=True)
