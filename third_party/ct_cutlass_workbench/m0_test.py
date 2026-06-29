import torch, ct_m0_ext as ext
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True
def cos(a,b):
    a=a.flatten().double(); b=b.flatten().double()
    return (a@b/(a.norm()*b.norm()+1e-20)).item()
M,K,N=2048,128,128
A=torch.randn(M,K,device="cuda"); B=torch.randn(N,K,device="cuda")
C=ext.m0_gemm(A,B); ref=A@B.t()
print(f"[M0] cos={cos(C,ref):.6f}  maxabs={(C.double()-ref.double()).abs().max().item():.3e}")
print("M0 OK" if cos(C,ref)>=0.999 else "M0 FAIL")
