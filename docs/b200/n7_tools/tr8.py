import sys, torch, statistics
from miniworld_engine.kernels.swa_dit.cuda import sm100
S=int(sys.argv[1]); N=int(sys.argv[2]); bf=int(sys.argv[3]); XF=tuple(sys.argv[4:])
H,D,C=4,32,128; dev=torch.device("cuda")
io=() if bf else (f"-DF32IO={sm100.F32IO_MODE}",)
x=torch.randn(N,H,S,D,device=dev); x=(x/x.norm(dim=-1,keepdim=True)*D**0.5)
dt=torch.bfloat16 if bf else torch.float32
kvdt=torch.bfloat16 if (bf or sm100.F32IO_MODE==2) else torch.float32
Qh=x.to(dt); Kh=x.roll(1,2).contiguous().to(kvdt); Vh=torch.randn(N,H,S,D,device=dev).to(kvdt)
seq=torch.full((N,),S,device=dev,dtype=torch.int32)
for c in (0,60,147):
    sm100.CUBINS[f"t{c}_s"]=("gattn_fwd",("-DSPLITB=1","-DTRACE=1",f"-DTRCTA={c}")+io+XF)
    TR=torch.zeros(16*128+16,dtype=torch.int64,device=dev)
    g=sm100.GattnFwd(f"t{c}_s",split_cubin=f"t{c}_s",f32io=0 if bf else sm100.F32IO_MODE,trace=TR); g.split=True
    O=torch.empty(N*S,C,device=dev,dtype=dt); L=torch.empty(N,H,S,device=dev)
    run,_,_=g.bind(Qh,Kh,Vh,seq,O=O,LSE=L,log2=True)
    for _ in range(3): TR.zero_(); run(); torch.cuda.synchronize()
    t=TR.cpu().tolist(); ent=[t[16*G] for G in range(128) if t[16*G]>0]
    per=statistics.median([ent[i+1]-ent[i] for i in range(10,len(ent)-12)]) if len(ent)>25 else 0
    ent_=ent; print(f"pre {ent_[0]-t[2040]} post {t[2041]-ent_[-1]}", end=" "); print(f"CTA {c}: blocks {len(ent)} span {ent[-1]-ent[0]} clk  steady period {per}  first gaps {[ent[i]-ent[0] for i in (1,2,3)]}")
    print("   epilogue stamps rel last enter: o_done+ld", [t[2043+k]-ent[-1] for k in range(14)], "exit", t[2041]-ent[-1])
