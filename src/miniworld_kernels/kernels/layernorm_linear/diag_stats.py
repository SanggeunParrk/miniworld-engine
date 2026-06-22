"""Pin down the M2 fused cos gap: compare M1, M2, oracle elementwise + stats."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear import layernorm_linear_cute
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import layernorm_linear_cute_fused

DEVICE = torch.device("cuda")
DT = torch.bfloat16


def cmp(a, b):
    a, b = a.float(), b.float()
    return (f"max={ (a-b).abs().max().item():.3e} relF={((a-b).norm()/(b.norm()+1e-12)).item():.3e} "
            f"cos={F.cosine_similarity(a.flatten(), b.flatten(), 0).item():.6f}")


def run(M, K, N):
    torch.manual_seed(0)
    eps = 1e-5
    x = torch.randn(M, K, device=DEVICE, dtype=DT)
    g = torch.randn(K, device=DEVICE, dtype=DT)
    b = torch.randn(K, device=DEVICE, dtype=DT)
    w = torch.randn(N, K, device=DEVICE, dtype=DT) / (K ** 0.5)
    bias = torch.randn(N, device=DEVICE, dtype=DT)
    oracle = F.linear(F.layer_norm(x, (K,), g, b, eps), w, bias)
    y1 = layernorm_linear_cute(x, g, b, w, bias, eps)
    y2 = layernorm_linear_cute_fused(x, g, b, w, bias, eps)
    print(f"=== M={M} K={K} N={N} ===")
    print(f"  M1 vs oracle : {cmp(y1, oracle)}")
    print(f"  M2 vs oracle : {cmp(y2, oracle)}")
    print(f"  M2 vs M1     : {cmp(y2, y1)}")
    # stats: how does in-kernel one-pass (bf16 x, fp32 acc) compare to layer_norm?
    xf = x.float()
    mean = xf.mean(-1)
    var_tp = ((xf - mean[:, None]) ** 2).mean(-1)            # two-pass (layer_norm)
    var_op = (xf * xf).mean(-1) - mean * mean                # one-pass (kernel)
    rstd_tp = torch.rsqrt(var_tp + eps)
    rstd_op = torch.rsqrt(var_op + eps)
    print(f"  rstd one-pass vs two-pass: {cmp(rstd_op, rstd_tp)}")
    # bad-row pattern (M2 vs M1 — isolates the kernel bug from oracle conditioning)
    rowmax = (y2.float() - y1.float()).abs().max(dim=1).values
    bad = (rowmax > 0.05).nonzero().flatten()
    print(f"  bad rows(M2 vs M1, >0.05): {bad.numel()}/{M}")
    if bad.numel():
        print(f"    first 8 idx       : {bad[:8].tolist()}")
        print(f"    idx % 128 (local) : {sorted(set((bad % 128).tolist()))[:16]}")
        print(f"    idx //128 (M-tile): {sorted(set((bad // 128).tolist()))[:16]}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    for K in (128, 256, 384, 512, 640, 768):
        run(16384, K, K)
