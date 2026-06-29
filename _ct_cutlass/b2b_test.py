import torch, ct_b2b_ext as ext
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=True
def cos(a,b):
    a=a.flatten().double(); b=b.flatten().double(); return (a@b/(a.norm()*b.norm()+1e-20)).item()
M,K,ND,D,DC=2048,128,256,128,384
x=torch.randn(M,K,device="cuda"); cond=torch.randn(M,DC,device="cuda")
wa=torch.randn(ND,K,device="cuda")*0.05; wb=torch.randn(ND,K,device="cuda")*0.05
ws=torch.randn(D,ND,device="cuda")*0.05; wsc=torch.randn(D,DC,device="cuda")*0.05
bsc=torch.randn(D,device="cuda")
a=x@wa.t(); b=x@wb.t(); h=torch.nn.functional.silu(a)*b
out=h@ws.t(); scale=cond@wsc.t()+bsc; yref=torch.sigmoid(scale)*out
y=ext.b2b_forward(x,cond,wa,wb,ws,wsc,bsc)
print(f"[b2b] cos(y)={cos(y,yref):.6f} maxabs={(y.double()-yref.double()).abs().max().item():.3e}")
print("B2B OK" if cos(y,yref)>=0.999 else "B2B FAIL")
