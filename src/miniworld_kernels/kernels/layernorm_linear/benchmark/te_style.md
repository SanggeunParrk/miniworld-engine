# TE-style trainable LayerNormLinear — `layernorm_linear_te_fn`

Full **forward+backward** median latency, H100 80GB bf16, `torch 2.10+cu128`, CUDA-warmed.
Same algorithm as Transformer Engine (materialize `x_normed=LN(x)` → plain cuBLAS GEMM), but
**custom for stride coverage**: consumes a strided/m-major input (trimul BDLL view, strides
`(1, L*L)`) with NO `.contiguous()` copy and returns `dx` in the same layout.

> ⛔ All numbers are full fwd+bwd via `triton.testing.do_bench` (warmup 25, rep 100), grads
> zeroed each iter. Bench: `te_style_bench.py`; plot: `te_style_plot.py` (both via `srun`).

![fwd+bwd vs TE / torch.compile](te_style_fwd_bwd.png)

## contiguous input (fair head-to-head)

| M | d | ours (ms) | TE (ms) | torch.compile (ms) | **ours vs TE** |
|---:|---:|---:|---:|---:|:---:|
| 16384 | 128 | 0.2176 | 0.2034 | 0.3279 | 0.93x |
| 65536 | 128 | 0.2371 | 0.2199 | 0.2957 | 0.93x |
| 262144 | 128 | 0.3727 | 0.5554 | 0.4180 | **1.49x** |
| 16384 | 256 | 0.2083 | 0.1932 | 0.2288 | 0.93x |
| 65536 | 256 | 0.3174 | 0.2913 | 0.4196 | 0.92x |
| 262144 | 256 | 0.7600 | 0.7884 | 0.6782 | **1.04x** |
| 16384 | 384 | 0.2156 | 0.2061 | 0.2641 | 0.96x |
| 65536 | 384 | 0.3272 | 0.3521 | 0.4383 | 1.08x |
| 262144 | 384 | 1.0688 | 1.1180 | 1.1579 | **1.05x** |
| 16384 | 512 | 0.3127 | 0.2703 | 0.4167 | 0.86x |
| 65536 | 512 | 0.4174 | 0.4165 | 0.4208 | 1.00x |
| 262144 | 512 | 1.4546 | 1.3922 | 1.4594 | 0.96x |

Large-M (262144) beats TE through d≤384 (1.04–1.49x) and ties at d=512 (0.96x); small-M within
4–14% (residual fixed overhead, widest at d=512). d=512 large-M is the one place ours dips slightly
(0.96x) — the LN-materialize round-trip cost grows with d.

## m-major input (trimul BDLL view; TE copies to contiguous internally)

| M | d | ours (ms) | TE (ms) | torch.compile (ms) | **ours vs TE** |
|---:|---:|---:|---:|---:|:---:|
| 16384 | 128 | 0.3158 | 0.3223 | 0.3739 | 1.02x |
| 65536 | 128 | 0.3120 | 0.3389 | 0.3894 | 1.09x |
| 262144 | 128 | 0.4950 | 1.1258 | 1.4780 | **2.27x** |
| 16384 | 256 | 0.2161 | 0.2277 | 0.2560 | 1.05x |
| 65536 | 256 | 0.3217 | 0.5076 | 0.9642 | 1.58x |
| 262144 | 256 | 0.7816 | 2.0176 | 2.8608 | **2.58x** |
| 16384 | 384 | 0.2174 | 0.2387 | 0.3464 | 1.10x |
| 65536 | 384 | 0.3475 | 0.7380 | 1.2504 | 2.12x |
| 262144 | 384 | 1.1920 | 3.0665 | 4.1724 | **2.57x** |
| 16384 | 512 | 0.2227 | 0.2826 | 0.4134 | 1.27x |
| 65536 | 512 | 0.4485 | 0.9003 | 1.4375 | 2.01x |
| 262144 | 512 | 1.5767 | 3.9320 | 5.2750 | **2.49x** |

Wins everywhere (1.02–2.58x); the lead grows with M because TE pays the strided→contiguous copy
regardless (probed: `TE(raw-strided)` == `TE(.contiguous())` timing), while we absorb the stride.

All grads verified `cos=1.0` vs fp32 autograd (`te_style_verify.py`); `dx.stride() == x.stride()`.
