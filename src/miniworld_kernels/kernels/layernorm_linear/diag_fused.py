"""Debug the fused kernel: with LNL_DEBUG=1 the kernel outputs rstd[m] broadcast
over n; compare to torch rstd to isolate the reduction/coord bug."""
import os
import torch
import torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import layernorm_linear_cute_fused

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
MODE = int(os.environ.get("LNL_DEBUG", "0"))


def run(M, K, N):
    eps = 1e-5
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    gamma = torch.randn(K, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(K, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(N, K, device=DEVICE, dtype=DTYPE) / (K ** 0.5)
    bias = torch.randn(N, device=DEVICE, dtype=DTYPE)
    y = layernorm_linear_cute_fused(x, gamma, beta, weight, bias, eps).float()  # (M,N)

    print(f"=== M={M} K={K} N={N}  LNL_DEBUG={MODE} ===")
    rows = torch.arange(M, device=DEVICE).float()[:, None].expand(M, N)
    cols = torch.arange(N, device=DEVICE).float()[None, :].expand(M, N)
    if MODE == 3:  # s_rstd[m]=tile_local_m broadcast → expect y[r,c] == r % 128
        ref = (torch.arange(M, device=DEVICE) % 128).float()
        col0 = y[:, 0]
        print(f"  row-constant? max(col0-colL)={(y[:,0]-y[:,N-1]).abs().max().item():.3e}")
        print(f"  y[:,0] vs (row%128): max|abs|={(col0 - ref).abs().max().item():.3e}")
        bad = ((col0 - ref).abs() > 0.5).nonzero(as_tuple=True)[0]
        print(f"  bad rows(>0.5): {bad.numel()}/{M}")
        if bad.numel():
            br = bad.tolist()
            print(f"    tile-local (row%128): {sorted(set(b % 128 for b in br))[:20]}")
            print(f"    M-tile idx (row//128): {sorted(set(b // 128 for b in br))[:12]}")
            print(f"    sample out: {[round(y[b,0].item(),1) for b in br[:6]]} vs ref {[int(ref[b].item()) for b in br[:6]]}")
    elif MODE == 4:  # expect y[r,c] == c
        print(f"  y vs col-index: max|abs|={(y - cols).abs().max().item():.3e}")
        print(f"  out[:4,:4]=\n{y[:4,:4]}")
    else:
        xf = x.float()
        mean = xf.mean(1)
        var = (xf * xf).mean(1) - mean * mean
        rstd = torch.rsqrt(var + eps)
        ref = rstd if MODE == 1 else mean * rstd
        col0 = y[:, 0]
        print(f"  row-constant? max(col0-colL)={(y[:,0]-y[:,N-1]).abs().max().item():.3e}")
        print(f"  vs torch: max|abs|={(col0-ref).abs().max().item():.3e}  cos={F.cosine_similarity(col0, ref, dim=0).item():.6f}")
        bad = ((col0 - ref).abs() > 0.05).nonzero(as_tuple=True)[0]
        print(f"  bad rows(>0.05): {bad.numel()}/{M}")
        if bad.numel():
            br = bad.tolist()
            print(f"    first 16: {br[:16]}")
            print(f"    bad row % 128 (tile-local): {sorted(set(b % 128 for b in br))[:20]}")
            print(f"    bad row // 128 (M-tile idx): {sorted(set(b // 128 for b in br))[:20]}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    run(65536, 128, 128)  # a shape that's wrong under persistent reuse
