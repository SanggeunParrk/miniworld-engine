"""A/B: fused-1+4 cute backward vs the unfused compose (cuBLAS dgrad + Triton LN-bwd).

Backward-only median ms (triton.testing.do_bench, L2-flushed). Both produce the SAME 5 grads;
the only difference is the dx path:
  fused   = dgrad_lnbwd_cute (dY@W + LN-norm-bwd in one epilogue) + T-decomposition for dW/dγ/dβ
  unfused = quack gemm(dY@W) → Triton layer_norm_bwd_dx_fused + cuBLAS wgrad on x_normed
Decision gate (FUSED_DGRAD_LNBWD_DESIGN.md): keep the fused path only where it WINS.
"""
import torch, triton
from miniworld_kernels.kernels.layernorm_linear.autograd import (
    _compose_backward, _compose_backward_fused,
)

D = torch.device("cuda"); dt = torch.bfloat16
def bench(fn): return triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])[0]

print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
print(f"{'M':>8} {'d':>5} | {'fused ms':>9} {'unfused ms':>11} {'speedup':>8}")
for d in (128, 256):
    for M in (16384, 65536, 262144):
        torch.manual_seed(0)
        x = torch.randn(M, d, device=D, dtype=dt)
        g = torch.randn(d, device=D, dtype=dt); b = torch.randn(d, device=D, dtype=dt)
        W = (torch.randn(d, d, device=D, dtype=dt) * d**-0.5)
        dY = torch.randn(M, d, device=D, dtype=dt)
        mean = x.float().mean(-1); rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
        args = (dY, x, mean, rstd, g, b, W, True)
        # correctness sanity (cos of dx between the two paths)
        dxf = _compose_backward_fused(*args)[0]
        dxu = _compose_backward(*args, dx_via_quack=True)[0]
        cos = torch.nn.functional.cosine_similarity(dxf.float().flatten(), dxu.float().flatten(), 0).item()
        tf = bench(lambda: _compose_backward_fused(*args))
        tu = bench(lambda: _compose_backward(*args, dx_via_quack=True))
        print(f"{M:>8} {d:>5} | {tf:>9.4f} {tu:>11.4f} {tu/tf:>7.2f}x   dx-cos(f,u)={cos:.5f}")
