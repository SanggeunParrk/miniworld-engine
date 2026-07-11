import torch
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import layernorm_linear_cute_fused
D = torch.device("cuda"); T = torch.bfloat16
M, K, N = 32768, 128, 128  # 256 tiles > grid -> persistent reuse
torch.manual_seed(0)
x = torch.randn(M, K, device=D, dtype=T); g = torch.randn(K, device=D, dtype=T)
b = torch.randn(K, device=D, dtype=T); w = torch.randn(N, K, device=D, dtype=T)/(K**0.5); bias = torch.randn(N, device=D, dtype=T)
y = layernorm_linear_cute_fused(x, g, b, w, bias, 1e-5); torch.cuda.synchronize(); print("done", y.shape)
