import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/snu_hwle/psk/miniworld-kernels/src")
import torch
if os.environ.get("MINIWORLD_TM1_SM100_IMPL")=="naive":
    from miniworld_kernels.kernels.tm1.cute.sm100_gate_gemm import gate_gemm
else:
    from miniworld_kernels.kernels.tm1.cute.sm100_gate_gemm_collective import gate_gemm
M,N,K=1048576,128,128
A=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3
BLp=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.3; BLg=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.3
BRp=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.3; BRg=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.3
def fwd():  # left+right = 2 launches, exactly what one trimul direction issues
    gate_gemm(A,BLp,BLg,mmajor=True); gate_gemm(A,BRp,BRg,mmajor=True)
for _ in range(8): fwd()
torch.cuda.synchronize()
for _ in range(3): fwd()
torch.cuda.synchronize()
