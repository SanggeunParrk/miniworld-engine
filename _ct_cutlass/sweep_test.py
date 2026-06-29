"""Per-shape config sweep: find best CUTLASS config vs cuBLAS for each training GEMM shape."""
import time
import torch
import ct_sweep_ext as ext

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


# All distinct training GEMM shapes (M, K, N) for D = A@B^T, A:(M,K), B:(N,K).
# atom: d=128 ND=256 ND2=512 dc=384 ; token: d=768 ND=1536 ND2=3072 dc=384
shapes = {}
for L in (2048, 4096, 8192):  # atom
    d, ND, ND2, dc = 128, 256, 512, 384
    shapes[f"atom L={L} expand(x@Wa)"] = (L, d, ND)
    shapes[f"atom L={L} squeeze(h@Ws)"] = (L, ND, d)
    shapes[f"atom L={L} scale(cond@Wsc)"] = (L, dc, d)
    shapes[f"atom L={L} dh(dout@Ws)"] = (L, d, ND)
    shapes[f"atom L={L} dx(dab@Wcat)"] = (L, ND2, d)
    shapes[f"atom L={L} dcond(dscale@Wsc)"] = (L, d, dc)
for L in (384, 512, 768, 1024):  # token
    d, ND, ND2, dc = 768, 1536, 3072, 384
    shapes[f"token L={L} expand(x@Wa)"] = (L, d, ND)
    shapes[f"token L={L} squeeze(h@Ws)"] = (L, ND, d)
    shapes[f"token L={L} scale(cond@Wsc)"] = (L, dc, d)
    shapes[f"token L={L} dh(dout@Ws)"] = (L, d, ND)
    shapes[f"token L={L} dx(dab@Wcat)"] = (L, ND2, d)
    shapes[f"token L={L} dcond(dscale@Wsc)"] = (L, d, dc)

NCFG = ext.num_cfg()
print(f"num configs = {NCFG}")
print(f"{'shape':42s} {'best_cfg':>8s} {'cutlass_us':>11s} {'cublas_us':>10s} {'ratio':>7s} {'cos':>8s}")
summary = []
for name, (M, K, N) in shapes.items():
    A = torch.randn(M, K, device=dev)
    B = torch.randn(N, K, device=dev)
    ref = A @ B.t()
    best_us, best_cfg, best_cos = 1e18, -1, 0.0
    for c in range(NCFG):
        try:
            D = ext.gemm_cfg(A, B, c)
        except Exception:
            continue
        cc = cos(D, ref)
        if cc < 0.99:
            continue
        t = bench(lambda: ext.gemm_cfg(A, B, c))
        if t < best_us:
            best_us, best_cfg, best_cos = t, c, cc
    t_cublas = bench(lambda: A @ B.t())
    ratio = best_us / t_cublas if best_cfg >= 0 else float("nan")
    print(f"{name:42s} {best_cfg:>8d} {best_us:>11.2f} {t_cublas:>10.2f} {ratio:>7.3f} {best_cos:>8.5f}")
    summary.append((name, best_cfg, ratio))

worst = max((r for _, _, r in summary if r == r), default=0)
print(f"\nworst ratio (best-cfg vs cuBLAS): {worst:.3f}x")
print("PARITY OK (<=1.2x all shapes)" if worst <= 1.2 else "SOME SHAPES >1.2x")
