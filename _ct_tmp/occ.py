import torch, triton
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton import composed as C
g=torch.Generator('cuda').manual_seed(0); f=lambda *s: torch.randn(*s,device='cuda',dtype=torch.float32,generator=g)
M,d,n,dc=768,768,2,384; ND=n*d
x,cond,Wa,Wb,Ws,Wsc,bsc=f(M,d),f(M,dc),f(ND,d)/d**.5,f(ND,d)/d**.5,f(d,ND)/ND**.5,f(d,dc)/dc**.5,torch.full((d,),-2.,device='cuda')
h=C._expand_swiglu(x,Wa,Wb); _=C._squeeze_gate(h,cond,Ws,Wsc,bsc); torch.cuda.synchronize()
# H100 SM limits
REGS_SM=65536; SMEM_SM=232448; THR_SM=2048; WARP=32; BLK_SM=32
def occ(name, k):
    best=k.best_config; 
    # find compiled kernel metadata
    print(f"\n[{name}] best_config={best}")
    for key,ck in k.cache.items():
        pass
    # the autotuner stores compiled in .fn; introspect via k.fn? use k.kernel? Simpler: pull from k.cache
    # triton Autotuner: after run, compiled kernels live in the JITFunction cache; print regs via run metadata
for name,k in (("expand",C._expand_swiglu_kernel),("squeeze",C._squeeze_gate_kernel)):
    bc=k.best_config
    nw=bc.num_warps; ns=bc.num_stages; threads=nw*WARP
    print(f"[{name}] {bc.all_kwargs() if hasattr(bc,'all_kwargs') else bc.kwargs}  num_warps={nw} num_stages={ns} threads/block={threads}")
    # try to recover n_regs/n_spills/shared from the JIT cache
    jf=k.fn
    found=False
    for ck in list(getattr(jf,'cache',{}).values()) if hasattr(jf,'cache') else []:
        for kern in (ck.values() if isinstance(ck,dict) else []):
            r=getattr(kern,'n_regs',None); sp=getattr(kern,'n_spills',None); sh=getattr(kern,'metadata',None)
            if r is not None:
                shared=getattr(kern,'shared',getattr(getattr(kern,'metadata',None),'shared',None))
                print(f"    n_regs={r} n_spills={sp} shared={shared}")
                if r: 
                    by_reg=REGS_SM//(r*threads) if r*threads else BLK_SM
                    by_smem=SMEM_SM//shared if shared else BLK_SM
                    by_thr=THR_SM//threads
                    blocks=min(by_reg,by_smem,by_thr,BLK_SM)
                    print(f"    -> blocks/SM: reg={by_reg} smem={by_smem} thr={by_thr} => {blocks}; occupancy={100*blocks*threads/THR_SM:.0f}% ; spill={'YES' if sp else 'no'}")
                found=True
                break
        if found: break
print("\nDONE")
