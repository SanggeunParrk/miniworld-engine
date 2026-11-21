import sys, torch, statistics
from miniworld_engine.kernels.swa_dit.cuda import sm100
S=int(sys.argv[1]); N=int(sys.argv[2]); bf=int(sys.argv[3]); fl=tuple(sys.argv[4:])
H,D,C=4,32,128; dev=torch.device("cuda")
io=() if bf else (f"-DF32IO={sm100.F32IO_MODE}",)
sm100.CUBINS["t_w"]=("gattn_fwd",("-DSPLITB=0","-DTRACE=1")+io+fl); sm100.CUBINS["t_s"]=("gattn_fwd",("-DSPLITB=1","-DTRACE=1")+io+fl)
TR=torch.zeros(16*128+16,dtype=torch.int64,device=dev)
g=sm100.GattnFwd("t_w",split_cubin="t_s",f32io=0 if bf else sm100.F32IO_MODE,trace=TR)
torch.manual_seed(1)
dt=torch.bfloat16 if bf else torch.float32
x=torch.randn(N,H,S,D,device=dev); x=(x/x.norm(dim=-1,keepdim=True)*D**0.5)
kvdt=torch.bfloat16 if (bf or sm100.F32IO_MODE==2) else torch.float32
Qh=x.to(dt); Kh=x.roll(1,2).contiguous().to(kvdt); Vh=torch.randn(N,H,S,D,device=dev).to(kvdt)
seq=torch.full((N,),S,device=dev,dtype=torch.int32)
O=torch.empty(N*S,C,device=dev,dtype=dt if bf else torch.float32); L=torch.empty(N,H,S,device=dev)
run,_,_=g.bind(Qh,Kh,Vh,seq,O=O,LSE=L,log2=True)
for _ in range(3): run(); torch.cuda.synchronize()
t=TR.cpu().tolist()
def col(k): return [t[16*G+k] for G in range(128)]
names={0:"enter",1:"ld done",3:"masked",4:"turn won",5:"exps done",6:"p_full arrived"}
G0,G1=20,100
def d(a,b): return statistics.median([t[16*G+b]-t[16*G+a] for G in range(G0,G1)])
print("per block (clk): ld",d(0,1),"mask",d(1,3),"turn wait",d(3,4),"exp",d(4,5),"st+arrive",d(5,6),"gap to next enter",statistics.median([t[16*(G+1)]-t[16*G+6] for G in range(G0,G1)]),"period",statistics.median([t[16*(G+1)]-t[16*G] for G in range(G0,G1)]))
print("MMA warp k8..:",[ (k,statistics.median([t[16*G+k]-t[16*G] for G in range(G0,G1)])) for k in range(8,16)])
print("first/last enter:", t[0], [t[16*G]-t[0] for G in (1,2,3,10,30,50,60,61,62,63)])
print("CTA0 clk", t[2041]-t[2040], "ns", t[2060]-t[2059], "MHz", (t[2041]-t[2040])/(t[2060]-t[2059])*1000)
