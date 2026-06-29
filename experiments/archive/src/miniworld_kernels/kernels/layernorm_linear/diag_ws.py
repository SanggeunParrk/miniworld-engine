"""Bring-up test for warp-specialized stats: the idle load-WG warps independently
replicate the STATIC tile sequence and dump per-row rstd to mDbg. Verify mDbg ==
reference rstd for ALL rows (proves the scheduler + gmem reduction on the stats warps).
Run with LNL_WS_DEBUG=1. Math WGs are untouched (output Y stays correct)."""
from __future__ import annotations
import torch

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import gemm_lnl_fused
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear import fold_for_gemm

DEVICE = torch.device("cuda")
DT = torch.bfloat16


def run(M, K, N):
    torch.manual_seed(0)
    eps = 1e-5
    x = torch.randn(M, K, device=DEVICE, dtype=DT)
    g = torch.randn(K, device=DEVICE, dtype=DT)
    b = torch.randn(K, device=DEVICE, dtype=DT)
    w = torch.randn(N, K, device=DEVICE, dtype=DT) / (K ** 0.5)
    bias = torch.randn(N, device=DEVICE, dtype=DT)
    Bw, S, B2 = fold_for_gemm(w, g, b, bias, w2_dtype=x.dtype)
    S2 = S.float().contiguous().view(1, N)
    B22 = B2.float().contiguous().view(1, N)
    Y = torch.empty(M, N, device=DEVICE, dtype=DT)
    mDbg = torch.full((M,), -123.0, device=DEVICE, dtype=torch.float32)
    gemm_lnl_fused(x, Bw, Y, S2, B22, eps, mDbg=mDbg)
    torch.cuda.synchronize()
    # reference rstd: one-pass var of bf16 x in fp32 (matches the kernel's math)
    xf = x.float()
    mean = xf.mean(-1)
    var = (xf * xf).mean(-1) - mean * mean
    ref = torch.rsqrt(var + eps)
    unfilled = (mDbg == -123.0).sum().item()
    err = (mDbg - ref).abs()
    relmax = (err / ref.abs()).max().item()
    print(f"=== M={M} K={K} N={N} ===")
    print(f"  unfilled rows: {unfilled}/{M}   max|rstd-ref|={err.max().item():.3e}   "
          f"relmax={relmax:.3e}   cos={torch.nn.functional.cosine_similarity(mDbg, ref, 0).item():.6f}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    run(16384, 128, 128)
    run(16384, 256, 256)
    run(16384, 768, 768)
    run(4096, 4096, 12288)
