"""The token-DiT attention core at the training shape, standalone: engine op fwd and fwd+bwd, fp32 (TF32 inside) and
bf16, against measured GEMM roofs. A = 48 augments, 16 heads x 48, bias [1, L, L, 16] shared by the augments.

FLOPs: one unit = 2 A H L^2 D. Forward = 2 units (QK^T, PV). Backward = 5 (recomputed QK^T, dV, dP, dK, dQ).
Timing: triton do_bench (L2 evicted), as the engine benchmarks kernels.
"""
import sys
import torch
import triton.testing as tt
from miniworld_engine.kernels.augmented_attention import triton_augmented_attention_pair_bias as attn

A, H, Dh, dev = 48, 16, 48, "cuda"
us = lambda f: tt.do_bench(f, warmup=10, rep=60, return_mode="median") * 1e3


def roof(dt):
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=dt); b = torch.randn(n, n, device=dev, dtype=dt)
    torch.backends.cuda.matmul.allow_tf32 = True
    t = us(lambda: a @ b)
    return 2 * n ** 3 / t / 1e6


torch.set_float32_matmul_precision("high")                           # fp32 GEMMs in TF32, as "medium" does
r32, r16 = roof(torch.float32), roof(torch.bfloat16)
print(f"GEMM roofs: TF32 {r32:.0f} TF/s, bf16 {r16:.0f} TF/s", flush=True)
for L in (384, 768):
    unit = 2 * A * H * L * L * Dh
    for dt in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        q, k, v = (torch.randn(A, 1, L, H, Dh, device=dev, dtype=dt) * 0.5 for _ in range(3))
        bias = torch.randn(1, L, L, H, device=dev, dtype=dt)
        for t_ in (q, k, v, bias): t_.requires_grad_(True)
        do = torch.randn(A, 1, L, H, Dh, device=dev, dtype=dt)
        with torch.no_grad():
            tf = us(lambda: attn(q, k, v, bias, None))
        def fb():
            o = attn(q, k, v, bias, None)
            torch.autograd.backward(o, do)
        tfb = us(fb)
        rf = r32 if dt == torch.float32 else r16
        name = "fp32/TF32" if dt == torch.float32 else "bf16"
        print(f"L{L} {name:9s}: fwd {tf:8.1f} us ({2 * unit / tf / 1e6:5.0f} TF/s, {100 * 2 * unit / tf / 1e6 / rf:4.1f}% of roof)"
              f"   fwd+bwd {tfb:8.1f} us ({7 * unit / tfb / 1e6:5.0f} TF/s, {100 * 7 * unit / tfb / 1e6 / rf:4.1f}%)"
              f"   floor fwd+bwd {7 * unit / rf / 1e6:6.1f} us", flush=True)
