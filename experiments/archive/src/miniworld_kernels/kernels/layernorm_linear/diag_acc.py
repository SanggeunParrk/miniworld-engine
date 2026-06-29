"""DEBUG=4: M2 outputs raw acc (x@W2^T). Compare to folded-acc reference, check rows 0-3."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import layernorm_linear_cute_fused
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear import fold_for_gemm

DEVICE = torch.device("cuda")
DT = torch.bfloat16


def cmp(a, b):
    a, b = a.float(), b.float()
    return (f"max={(a-b).abs().max().item():.3e} relF={((a-b).norm()/(b.norm()+1e-12)).item():.3e} "
            f"cos={F.cosine_similarity(a.flatten(), b.flatten(), 0).item():.6f}")


def run(M, K, N):
    torch.manual_seed(0)
    eps = 1e-5
    x = torch.randn(M, K, device=DEVICE, dtype=DT)
    g = torch.randn(K, device=DEVICE, dtype=DT)
    b = torch.randn(K, device=DEVICE, dtype=DT)
    w = torch.randn(N, K, device=DEVICE, dtype=DT) / (K ** 0.5)
    bias = torch.randn(N, device=DEVICE, dtype=DT)
    Bw, S, B2 = fold_for_gemm(w, g, b, bias, w2_dtype=x.dtype)
    acc_ref = (x.float() @ Bw.float().T)          # x @ W2^T (the GEMM the kernel runs)
    acc_k = layernorm_linear_cute_fused(x, g, b, w, bias, eps)   # DEBUG=4 -> raw acc
    print(f"=== M={M} K={K} N={N} (DEBUG should be 4) ===")
    print(f"  acc M2 vs ref : {cmp(acc_k, acc_ref)}")
    rowmax = (acc_k.float() - acc_ref.float()).abs().max(dim=1).values
    bad = (rowmax > 0.05 * acc_ref.float().abs().mean()).nonzero().flatten()
    print(f"  bad rows: {bad.numel()}/{M}; local%128={sorted(set((bad%128).tolist()))[:8]}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    run(16384, 512, 512)
    run(4096, 4096, 4096)
