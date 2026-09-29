import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/snu_hwle/psk/miniworld-engine/src")
import torch
from miniworld_engine.kernels.tm1.cute.sm100_gate_gemm_collective import gate_gemm

torch.manual_seed(0)
def check(M, N=128, K=128, mmajor=True):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    Bp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    Bg = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    # fp32 reference
    p = A.float() @ Bp.float().T
    g = A.float() @ Bg.float().T
    ref = torch.sigmoid(g) * p                      # (M, N)
    out = gate_gemm(A, Bp, Bg, mmajor=mmajor)
    if mmajor:
        y = out.t().float()   # (N,M)->(M,N)
    else:
        y = out.float()
    err = (y - ref).abs()
    cos = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0).item()
    rel = (err.mean() / ref.abs().mean()).item()
    print(f"M={M} mmajor={mmajor}: cos={cos:.6f} relmean={rel:.3e} maxabs={err.max().item():.3e} ref|mean|={ref.abs().mean().item():.3e}", flush=True)
    return cos

print("PRE", flush=True)
check(256, mmajor=True)
check(256, mmajor=False)
check(4096, mmajor=True)
c = check(1048576, mmajor=True)
# timing (isolated, two launches like the trimul does left+right; here one side)
import triton
A = torch.randn(1048576, 128, device="cuda", dtype=torch.bfloat16)*0.3
Bp = torch.randn(128,128, device="cuda", dtype=torch.bfloat16)*0.3
Bg = torch.randn(128,128, device="cuda", dtype=torch.bfloat16)*0.3
def one(): gate_gemm(A,Bp,Bg,mmajor=True)
for _ in range(5): one()
ms = triton.testing.do_bench(one, warmup=30, rep=100)
print(f"isolated gated side M=1048576: {ms:.4f} ms/side (do_bench)", flush=True)
