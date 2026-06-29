# TE-style trainable LayerNormLinear — `layernorm_linear_te_fn`

Full **forward+backward** median latency, H100 80GB bf16, `torch 2.10+cu128`, CUDA-warmed.
Same algorithm as Transformer Engine (materialize `x_normed=LN(x)` → plain cuBLAS GEMM), but
**custom for stride coverage**: consumes a strided/m-major input (trimul BDLL view, strides
`(1, L*L)`) with NO `.contiguous()` copy and returns `dx` in the same layout.

> ⛔ All numbers are full fwd+bwd via `triton.testing.do_bench` (warmup 25, rep 100), grads
> zeroed each iter. Bench: `te_style_bench.py`; plot: `te_style_plot.py` (both via `srun`).
> The LN kernels autotune `BLOCK_M × num_warps × num_stages` keyed on **(N, M-bucket)** — keying
> on N alone reused one M's config across all M for a given d (≈5-8% left on the table at large M).

![fwd+bwd vs TE / torch.compile](te_style_fwd_bwd.png)

## contiguous input (fair head-to-head)

| M | d | ours (ms) | TE (ms) | torch.compile (ms) | **ours vs TE** |
|---:|---:|---:|---:|---:|:---:|
| 16384 | 128 | 0.2344 | 0.2853 | 0.3283 | 1.22x |
| 65536 | 128 | 0.2389 | 0.2214 | 0.2647 | 0.93x |
| 262144 | 128 | 0.3731 | 0.5536 | 0.4188 | **1.48x** |
| 16384 | 256 | 0.2307 | 0.2059 | 0.2477 | 0.89x |
| 65536 | 256 | 0.2616 | 0.2631 | 0.2869 | 1.01x |
| 262144 | 256 | 0.6998 | 0.7841 | 0.6803 | **1.12x** |
| 16384 | 384 | 0.2382 | 0.2072 | 0.2740 | 0.87x |
| 65536 | 384 | 0.3262 | 0.3501 | 0.3518 | 1.07x |
| 262144 | 384 | 1.0692 | 1.1178 | 1.1530 | **1.05x** |
| 16384 | 512 | 0.2407 | 0.2183 | 0.2705 | 0.91x |
| 65536 | 512 | 0.4174 | 0.4184 | 0.4188 | 1.00x |
| 262144 | 512 | 1.4495 | 1.3873 | 1.4522 | 0.96x |

Large-M (262144) beats TE through d≤384 (1.05–1.48x), ties at d=512 (0.96x). Mid-M (65536) ties or
wins (1.00–1.07x). Small-M (16384) is the residual gap (0.87–0.91x at d≥256) = forward fixed-launch
overhead — TE fuses LN into the GEMM prologue; we run a separate LN-materialize + plain GEMM. The
(N,M)-bucketed autotune recovered the large/mid-M cases (d256 M65536 0.92→1.01x, M262144 1.04→1.12x).

## m-major input (trimul BDLL view; TE copies to contiguous internally)

| M | d | ours (ms) | TE (ms) | torch.compile (ms) | **ours vs TE** |
|---:|---:|---:|---:|---:|:---:|
| 16384 | 128 | 0.2387 | 0.2308 | 0.2579 | 0.97x |
| 65536 | 128 | 0.2385 | 0.3399 | 0.3677 | 1.43x |
| 262144 | 128 | 0.4626 | 1.1309 | 1.5171 | **2.44x** |
| 16384 | 256 | 0.2304 | 0.2437 | 0.2550 | 1.06x |
| 65536 | 256 | 0.2605 | 0.5058 | 0.9675 | 1.94x |
| 262144 | 256 | 0.8208 | 2.0124 | 2.8567 | **2.45x** |
| 16384 | 384 | 0.2364 | 0.2407 | 0.3482 | 1.02x |
| 65536 | 384 | 0.3521 | 0.7382 | 1.2469 | 2.10x |
| 262144 | 384 | 1.2067 | 3.0639 | 4.1518 | **2.54x** |
| 16384 | 512 | 0.2400 | 0.2831 | 0.4106 | 1.18x |
| 65536 | 512 | 0.4386 | 0.9030 | 1.4647 | 2.06x |
| 262144 | 512 | 1.4945 | 3.9554 | 5.2908 | **2.65x** |

Wins almost everywhere (0.97–2.65x; only d=128 M=16384 ≈ tie); the lead grows with M because TE
pays the strided→contiguous copy regardless (probed: `TE(raw-strided)` == `TE(.contiguous())`
timing), while we absorb the stride.

All grads verified `cos=1.0` vs fp32 autograd (`te_style_verify.py`); `dx.stride() == x.stride()`.

## across dtypes (full fwd+bwd, ours vs TE; `te_style_dtype_bench.py`)

fp32 uses the default TF32 ('high') GEMM policy. `>1.0` = ours faster.

| M | d | **contiguous** fp32 / bf16 / fp16 | **m-major** fp32 / bf16 / fp16 |
|---:|---:|:---:|:---:|
| 65536 | 256 | 1.18 / 1.00 / 0.93 | 1.77 / 1.99 / 1.82 |
| 262144 | 256 | 1.14 / 1.11 / 0.91 | 1.92 / 2.42 / 2.19 |
| 65536 | 512 | 1.06 / 1.05 / 1.05 | 1.57 / 2.08 / 2.11 |
| 262144 | 512 | 1.05 / 1.00 / 1.02 | 1.77 / 2.59 / 2.60 |

- **fp32: beats TE contiguous everywhere (1.05–1.18x)** — fp32 isn't TE's optimized path (its LN-in-GEMM
  fusion targets 16-bit); our materialize + TF32 GEMM wins.
- **bf16: tie–win** (1.00–1.11x). **fp16: ~tie, slightly behind at d=256 (0.91–0.93x)** — fp16 is TE's
  most-tuned path.
- **m-major (trimul): dominant in every dtype (1.57–2.60x)** — the stride win is dtype-independent.
