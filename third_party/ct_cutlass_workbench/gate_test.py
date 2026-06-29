"""Gate: build the minimal CUTLASS Hopper TF32 GEMM, verify cos vs torch fp32, bench vs cuBLAS."""
import time
import torch
import ct_gate_ext

torch.manual_seed(0)
dev = "cuda"


def cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


# token expand shape: M=1024, K=768, N=1536  (a = x @ Wa^T)
M, K, N = 1024, 768, 1536
A = torch.randn(M, K, device=dev, dtype=torch.float32)
B = torch.randn(N, K, device=dev, dtype=torch.float32)  # weight (N,K), D = A @ B^T

D_cutlass = ct_gate_ext.gate_gemm(A, B)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
D_torch_tf32 = A @ B.t()
D_ref_fp32 = A.double() @ B.t().double()

c_cutlass = cos(D_cutlass, D_ref_fp32)
c_torch = cos(D_torch_tf32, D_ref_fp32)
maxdiff = (D_cutlass.double() - D_torch_tf32.double()).abs().max().item()

print(f"[GATE] shape M={M} K={K} N={N}")
print(f"[GATE] cos(cutlass, fp32_ref) = {c_cutlass:.6f}")
print(f"[GATE] cos(torch_tf32, fp32_ref) = {c_torch:.6f}")
print(f"[GATE] maxabsdiff(cutlass, torch_tf32) = {maxdiff:.3e}")

t_cutlass = bench(lambda: ct_gate_ext.gate_gemm(A, B))
t_torch = bench(lambda: A @ B.t())
print(f"[GATE] cutlass = {t_cutlass*1000:.2f} us")
print(f"[GATE] torch(cuBLAS tf32) = {t_torch*1000:.2f} us")
print(f"[GATE] cutlass/cublas ratio = {t_cutlass/t_torch:.3f}x")

ok = c_cutlass >= 0.999 and (t_cutlass / t_torch) <= 1.5
print(f"[GATE] RESULT: {'PASS' if ok else 'CHECK'} (cos>=0.999 and <=1.5x cuBLAS)")
