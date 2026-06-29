"""Build + validate the CUTLASS TF32 fused adaLN forward (fwd1 EVT + fwd2)."""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/psk6950/miniworld-kernels/_ct_cutlass/cutlass"
BUILD = "/home/psk6950/.cache/adaln_cutlass_fwd"
os.makedirs(BUILD, exist_ok=True)

print("building fused fwd (EVT — slow)...", flush=True)
mod = load(
    name="adaln_cutlass_fwd",
    sources=[os.path.join(HERE, "adaln_fwd.cu")],
    extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include")],
    extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90a", "--expt-relaxed-constexpr",
                       "--expt-extended-lambda", "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-DNDEBUG"],
    build_directory=BUILD, verbose=True,
)
print("build OK", flush=True)

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
eps = 1e-5


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


for (M, d) in [(4096, 768), (32768, 768), (8192, 128)]:
    cond = torch.randn(M, d, device="cuda", dtype=torch.float32)
    x = torch.randn(M, d, device="cuda", dtype=torch.float32)
    lnw = torch.randn(d, device="cuda", dtype=torch.float32)
    Ws = torch.randn(d, d, device="cuda", dtype=torch.float32) * d ** -0.5
    Wb = torch.randn(d, d, device="cuda", dtype=torch.float32) * d ** -0.5
    scale_b = torch.randn(d, device="cuda", dtype=torch.float32) * 0.1

    cond_aff = F.layer_norm(cond, (d,), lnw, None, eps)
    x_hat = F.layer_norm(x, (d,), None, None, eps)
    # reference
    scale = F.linear(cond_aff, Ws, scale_b)
    bias = F.linear(cond_aff, Wb, None)
    ref = torch.sigmoid(scale) * x_hat + bias
    # cutlass fused
    y1 = mod.adaln_fwd1(cond_aff, Ws, scale_b, x_hat)
    y = mod.adaln_fwd2(cond_aff, Wb, y1)
    print(f"  M={M} d={d}: fwd1cos={cos(torch.sigmoid(scale)*x_hat, y1):.6f}  "
          f"fullcos={cos(ref, y):.6f}  maxabs={(ref-y).abs().max().item():.3e}", flush=True)
print("DONE", flush=True)
