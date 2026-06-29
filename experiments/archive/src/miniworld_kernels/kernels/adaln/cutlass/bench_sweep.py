"""Sweep CUTLASS TF32 GEMM configs vs cuBLAS at adaln token/atom shapes (find best config)."""
from __future__ import annotations
import os, torch, triton
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/psk6950/miniworld-kernels/_ct_cutlass/cutlass"
BUILD = "/home/psk6950/.cache/adaln_cutlass_sweep"
os.makedirs(BUILD, exist_ok=True)
print("building sweep (8 kernels — slow)...", flush=True)
mod = load(name="adaln_cutlass_sweep", sources=[os.path.join(HERE, "gemm_sweep.cu")],
           extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include")],
           extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90a", "--expt-relaxed-constexpr",
                              "--expt-extended-lambda", "-DNDEBUG"],
           build_directory=BUILD, verbose=False)
print("build OK", flush=True)
torch.backends.cuda.matmul.allow_tf32 = True


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return m * 1000


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def t_cub(A, W):  # kernel-only-ish cuBLAS via cuda events
    import torch.cuda as tc
    for _ in range(20):
        A @ W.t()
    tc.synchronize()
    b = tc.Event(enable_timing=True); e = tc.Event(enable_timing=True)
    b.record()
    for _ in range(100):
        A @ W.t()
    e.record(); tc.synchronize()
    return b.elapsed_time(e) / 100 * 1000  # us


for (M, K, N) in [(32768, 768, 768), (32768, 768, 1536), (262144, 128, 128), (262144, 128, 256)]:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    W = torch.randn(N, K, device="cuda", dtype=torch.float32) * K ** -0.5
    ref = A @ W.t()
    tcub = t_cub(A, W)
    gflop = 2 * M * K * N / 1e9
    print(f"\n## M={M} K={K} N={N}  ({gflop:.0f} GF)   cuBLAS={tcub:.1f}us ({gflop/ (tcub/1e6)/1e3:.0f} TF/s)  [kernel-only events]", flush=True)
    best = (1e9, -1)
    for cfg in range(mod.n_cfg()):
        try:
            c = cos(ref, mod.gemm(A, W, cfg))             # correctness
            tm = mod.bench(A, W, cfg, 20, 100)            # kernel-only (C++ events, setup excluded)
            mark = " <-- best" if tm < best[0] else ""
            if tm < best[0]:
                best = (tm, cfg)
            print(f"   cfg{cfg}: {tm:7.1f}us  ({gflop/(tm/1e6)/1e3:3.0f} TF/s)  cuBLAS/ct={tcub/tm:.2f}x  cos={c:.5f}{mark}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   cfg{cfg}: FAIL {str(e)[:80]}", flush=True)
    print(f"   => BEST cfg{best[1]} {best[0]:.1f}us (cuBLAS {tcub:.1f}us, ratio {tcub/best[0]:.2f}x)", flush=True)
print("DONE", flush=True)
