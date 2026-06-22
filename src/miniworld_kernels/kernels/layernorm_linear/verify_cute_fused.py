"""Compile + verify the MILESTONE-2 fused kernel (stats inside the GEMM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def compare(a, b):
    a32, b32 = a.float(), b.float()
    return (f"max|abs|={(a32 - b32).abs().max().item():.3e}  "
            f"rel_fro={((a32 - b32).norm() / (b32.norm() + 1e-12)).item():.3e}  "
            f"cos={F.cosine_similarity(a32.flatten(), b32.flatten(), dim=0).item():.6f}")


def run(M, K, N):
    eps = 1e-5
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    gamma = torch.randn(K, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(K, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(N, K, device=DEVICE, dtype=DTYPE) / (K ** 0.5)
    bias = torch.randn(N, device=DEVICE, dtype=DTYPE)
    oracle = F.linear(F.layer_norm(x, (K,), gamma, beta, eps), weight, bias)
    y = layernorm_linear_cute_fused(x, gamma, beta, weight, bias, eps)
    print(f"=== M={M} K={K} N={N} ===")
    print(f"  fused vs true : {compare(y, oracle)}")


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}\n")
    # square grid (matches the bench)
    for d in (128, 256, 384, 512, 768):
        for M in (16384, 65536, 262144):
            run(M, d, d)
    # large / non-square N (the M1 ColVecLoad persistent bug regime)
    run(4096, 4096, 4096)
    run(4096, 4096, 8192)
    run(4096, 4096, 12288)  # QKV


if __name__ == "__main__":
    main()
