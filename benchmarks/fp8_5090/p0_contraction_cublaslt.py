import torch
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "_bench_util.py")).read().split("import foldjet_engine")[0])
from foldjet_engine.kernels.trimul_inproj.cuda import _h100_fp8 as F
fp8=torch.float8_e4m3fn
for L,C in ((256,128),(512,128),(1024,128),(1536,64),(2048,32)):
    a=torch.randn(C,L,L,device="cuda",dtype=torch.bfloat16); b=torch.randn(C,L,L,device="cuda",dtype=torch.bfloat16)
    aq,bq=a.to(fp8),b.to(fp8); out=torch.empty(C,L,L,device="cuda",dtype=torch.bfloat16)
    tb=bench(lambda: torch.bmm(a,b.transpose(1,2)),n=2)
    try:
        tf=bench(lambda: F.contract_e4m3(aq,bq,out),n=2)
    except Exception as e: print(L,"ERR",e); continue
    fl=2*L**3*C
    # accuracy vs fp32
    ref=torch.bmm(a[:2].float(),b[:2].float().transpose(1,2)); F.contract_e4m3(aq[:2].contiguous(),bq[:2].contiguous(),out[:2])
    print(f"L={L} C={C}: bf16 {tb:8.1f} us ({fl/tb/1e6:.0f} TF) | cuBLASLt e4m3 {tf:8.1f} us ({fl/tf/1e6:.0f} TF) | {tb/tf:.2f}x")
