"""De-risk: JIT-build the minimal CUTLASS SM90 TF32 GEMM and validate vs torch.matmul."""
from __future__ import annotations
import os
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/psk6950/miniworld-kernels/_ct_cutlass/cutlass"
BUILD = "/home/psk6950/.cache/adaln_cutlass_build"
os.makedirs(BUILD, exist_ok=True)

print("building (nvcc, CUTLASS templates — slow)...", flush=True)
mod = load(
    name="adaln_cutlass_tf32",
    sources=[os.path.join(HERE, "gemm_tf32.cu")],
    extra_include_paths=[
        os.path.join(CUTLASS, "include"),
        os.path.join(CUTLASS, "tools", "util", "include"),
    ],
    extra_cuda_cflags=[
        "-O3", "-std=c++17", "-arch=sm_90a",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-DNDEBUG",
    ],
    build_directory=BUILD,
    verbose=True,
)
print("build OK", flush=True)

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
for (M, K, N) in [(4096, 768, 768), (32768, 768, 1536), (8192, 128, 256)]:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    W = torch.randn(N, K, device="cuda", dtype=torch.float32) * K ** -0.5
    D = mod.gemm_tf32(A, W)
    ref = (A @ W.t())
    c = torch.nn.functional.cosine_similarity(D.float().flatten(), ref.float().flatten(), dim=0).item()
    maxe = (D - ref).abs().max().item()
    print(f"  M={M} K={K} N={N}: cos={c:.6f} maxabs={maxe:.3e}", flush=True)
print("DONE", flush=True)
