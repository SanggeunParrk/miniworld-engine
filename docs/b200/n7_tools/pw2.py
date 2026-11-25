import sys, time, torch
from foldjet_engine.kernels.swa_dit.cuda import sm100
S=int(sys.argv[1]); fl=tuple(sys.argv[2:]); N=1; H,D,C=4,32,128; dev=torch.device("cuda")
sm100.CUBINS["p_w"]=("gattn_fwd",("-DSPLITB=0",)+fl); sm100.CUBINS["p_s"]=("gattn_fwd",("-DSPLITB=1",)+fl); sm100.CUBINS["p_c"]=("gattn_fwd",("-DSPLITB=0","-DCSPLIT=1")+fl)
g=sm100.GattnFwd("p_w",split_cubin="p_s",csplit_cubin="p_c")
x=torch.randn(N,H,S,D,device=dev); x=(x/x.norm(dim=-1,keepdim=True)*D**0.5).bfloat16(); Kk=x.roll(1,2).contiguous()
V=torch.randn(N,H,S,D,device=dev).bfloat16(); seq=torch.full((N,),S,device=dev,dtype=torch.int32)
O=torch.empty(N*S,C,device=dev,dtype=torch.bfloat16); L=torch.empty(N,H,S,device=dev)
run,_,_=g.bind(x,Kk,V,seq,O=O,LSE=L,log2=True)
for _ in range(5): run()
torch.cuda.synchronize(); t=time.time(); n=0
while time.time()-t<5: run(); n+=1
torch.cuda.synchronize(); print("avg us", (time.time()-t)/n*1e6, "launches", n)
