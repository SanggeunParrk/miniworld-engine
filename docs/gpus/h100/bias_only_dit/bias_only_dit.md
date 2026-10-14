# Bias-only token DiT on H100 (sm90)

Kernel-level status of the bias-only token DiT block (`modules/bias_only_dit`, `BiasOnlyDiTBlock`) on H100: AdaLN +
**bias-only attention** (the logits are the pair bias, `softmax(to_bias(LN(pair))) v`, gated by `sigmoid(g)` and the conditioning
scale) + the conditioned SwiGLU transition. d_single 768, d_cond 384, d_pair 128, 16 heads x 48, transition n = 2; the
module-level summary is in [h100.md](../h100.md), the same block on B200 is
[../../b200/bias_only_dit/bias_only_dit.md](../../b200/bias_only_dit/bias_only_dit.md). Columns are (Length, Dimension, dtype)
with the token width 768 (`token_single`).

With no query and no key, `P = softmax(bias)` depends on neither the sample nor the solver step, only on the pair: one `P` per
block serves every sample. The fast path is therefore a different algorithm from the token DiT's: `P` is made once per pair and
the per-sample attention is a GEMM per head (`P_h [L, L] x v_h [L, 48]`) with the gate in its epilogue -- no online softmax.

**No other implementation of this block exists** (Anthropic's kit has no bias-only block -- the bench row is
`UnsupportedBenchmark` -- and cuEquivariance has only the triangle ops), so every measurement is against the PyTorch module.

## Scope and dispatch

Hand-written CUDA and cuBLAS only (no Triton), bf16 and fp32 (TF32 tensor cores, fp32 softmax / norm statistics / residual),
16 x 48 heads, B == 1, L a multiple of 128 up to 768, a key mask [1, L] or none. Everything else keeps the module's PyTorch
composition. `settings.bias_only_dit_h100` (default on) switches the Hopper paths.

- **Inference** (`integrations/bias_only_dit.py` -> `kernels/bias_only_dit/cuda/runner.py`, `FusedBiasOnlyDiT`): one conditioning
  per sample or one shared by the samples. Per call: the conditioning tables (LN rows + two cuBLAS GEMMs), then per block
  `adaln_in_rows` -> cuBLAS v | g -> **`pvx`** (bf16) / **`pvx32`** (fp32) -> cuBLAS out -> `resgate_adaln_rows` -> expand
  (bf16: `gemm_swiglu_sm90`, SwiGLU in the epilogue; fp32: cuBLAS + `swiglu_cuda`) -> cuBLAS squeeze -> `resgate_out_rows`.
  `P` is made once per pair and mask (`runner.hoist`: pair LayerNorm + projection + softmax rows) and served from the
  capture-scoped cache.
- **Training** (`integrations/bias_only_dit_train.py`): one autograd Function per block, forward and backward each one opaque op.
  bf16: pair bias + `softmax_t` (P and P^T), v | g GEMM, `pvx` (a = sigmoid(g) P v), out GEMM, the AdaLN / residual rows, expand
  GEMM + `swiglu_rows` (a | b saved), squeeze GEMM; backward `res_c_bwd`, dh GEMM + `swiglu_bwd`, dxt, `res_adaln_b_bwd`, dog,
  `gate_bwd_rows`, **`pvx` ungated** (dV = P^T dO), **`dpb2`** (dbias = P o (sum_a dO v^T - D)), dxa, `adaln_a_bwd`, the conditioning
  backward, `pair_bias_bwd`, the weight gradients as one cuBLAS GEMM each (no row split on Hopper). fp32: the same steps on TF32
  GEMMs; the attention core is `pvx32`, dV and dbias are two batched cuBLAS GEMMs over the heads with the samples folded into
  the columns (`[H, L, A 48]` is one regular matrix per head).

## Kernels (`kernels/bias_only_dit/cuda/pv_gate_inf.cu`, namespace `bo90`)

### `pvx` · sigmoid(g) (P v) per head and sample, bf16

CTA = (head, 64 x WGS query rows, a group of up to 5 samples). One or two consumer warpgroups keep the accumulators of the whole
group and a producer thread streams `[P rows | v of every sample]` k-blocks (64 keys) through a TMA ring, so P is read once per
(head, rows, group) and v once per (head, rows). The samples of a group form the N dimension of ONE wgmma per k step
(`m64n192` + `m64n128` for 5 samples; 25 % of each 64-column v tile is padding). The gate rows arrive by TMA into shared memory
before the epilogue, the bf16 result is staged and leaves as one TMA store per sample: no per-thread global access in the
epilogue. `(WGS, samples per CTA)` come from a cost model so that small L still fills the SMs. The same kernel without the gate
is dV = P^T dO of the backward.

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Core alone (CUDA events, S = 5 / S = 48), previous kernel `pv_grouped` -> `pvx`: L768 96.8 -> 23.2 us / 704 -> 152 us; L384
29.2 -> 10.6 us / 187 -> 55 us.

### `pvx32` · the same in fp32 on TF32 wgmma

TF32 wgmma takes K-major operands only, so v arrives transposed (`v^T [768, S L]`, a second cuBLAS GEMM `Wv xa^T`; g is a third).
Sample j of a group is rows `48 j .. 48 j + 47` of the B tile (a multiple of 8 rows, 128-B-swizzle aligned), so the whole group is
one `m64n(48 x samples)k8` wgmma with no padding. L768 S5 36.8 us, S48 264 us; error against fp64 7.7e-4.

| (Length, Dimension, dtype) | (128, 768, fp32) | (256, 768, fp32) | (384, 768, fp32) | (512, 768, fp32) | (640, 768, fp32) | (768, 768, fp32) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### `dpb2` · dbias = P o (sum_a dO_a v_a^T - D), bf16

A 128 x TN (TN = 256 at L % 256 == 0, else 128) output tile per CTA, two consumer warpgroups, both operands K-major, three
m64nTNk16 steps per sample over the A samples; P is read and the result written in 64-column chunks through TMA with a
128-B-swizzled staging tile. The 64 x 64-tile kernel it replaces re-read ~1.8 GB of operands from L2 at A = 48. L768 A48
269 -> 128 us, L384 87 -> 58 us (L256 keeps the old kernel: 64 CTAs).

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Row kernels and the rest

The row passes (`adaln_in_rows`, `resgate_adaln_rows`, `resgate_out_rows`, `swiglu_rows`, the training rows) are shared with the
token DiT (`conditioned_transition/cuda/token_dit_rows.cu`, `bias_only_dit/cuda/bias_only_dit_train_rows.cu`,
`bias_only_dit_f32_rows.cu`); every GEMM is cuBLAS (fp32: TF32), except the bf16 inference expand GEMM (`gemm_swiglu_sm90`).

## Measurements (2026-10-14)

H100 80GB HBM3 (cssb node01), torch 2.13.0+cu129, `benchmarks/runners/bench.py target=bias_only_dit level=module`, CUDA graph,
key mask 20 % (inference) / 12.5 % (training), ms, median. Accuracy columns are against an fp32 IEEE PyTorch block.

```
python benchmarks/runners/bench.py target=bias_only_dit level=module mode=inference precision=bf16-mixed cudagraph=manual \
    min_seq_len=384 max_seq_len=768 seq_len_step=384 'implementations=[pytorch,miniworld]' +shared_cond=true mask_prob=0.2
# fp32: precision=32; training: mode=training cudagraph=manual mask_prob=0.125
```

### Inference (S = 5, one conditioning shared by the samples)

| (Length, Dimension, dtype) | PyTorch compiled | ours | × | output rel. error: PyTorch / ours |
|---|---|---|---|---|
| (384, 768, bf16) | 0.208 | **0.090** | 2.3 | 3.9e-3 / 3.1e-3 |
| (768, 768, bf16) | 0.425 | **0.145** | 2.9 | 3.9e-3 / 3.1e-3 |
| (384, 768, fp32) | 0.367 | **0.163** | 2.3 | -- / 2.9e-4 |
| (768, 768, fp32) | 0.777 | **0.299** | 2.6 | -- / 2.9e-4 |

One conditioning per sample (bf16): L384 0.104, L768 0.179 ms (PyTorch 0.212 / 0.432).

### Training (A = 48)

| (Length, Dimension, dtype) | PyTorch compiled | ours | × |
|---|---|---|---|
| (384, 768, bf16) | 2.516 | **2.134** | 1.18 |
| (768, 768, bf16) | 4.949 | **4.314** | 1.15 |
| (384, 768, fp32) | 5.655 | **5.102** | 1.11 |
| (768, 768, fp32) | 11.073 | **10.284** | 1.08 |

Gradients (output, input, every parameter) against the fp32 IEEE block are within the PyTorch TF32 / bf16 error of the same
recipe (`tests/gpu/sm90/test_h100_bias_only_dit_gpu.py`), and a training step is bitwise repeatable.

### Where the time goes (bf16 training L768 A48, kernel sum 4.3 ms; `ncu`, % of the roofline = max(DRAM, tensor))

| | share | roofline | |
|---|---|---|---|
| cuBLAS GEMMs (forward, data and weight gradients) | 52 % | 64-91 % tensor | at its practical rate |
| row passes (LN / AdaLN / residual / gate / SwiGLU, fwd + bwd) | 39 % | 82-90 % DRAM | one read and one write of each tensor |
| `pvx` x 2, `dpb2`, `softmax_t`, pair bias fwd + bwd | 9 % | 45-80 % | `pvx` ~46 % |

The step is at ~75 % of the time-weighted roofline. What is left is the cuBLAS GEMMs and one HBM round trip per row pass;
removing those needs fusions with the GEMM epilogues, which were measured and lost on this card (below).

## What was tried and not kept

| attempt | result |
|---|---|
| TMA multicast of the v tiles across a cluster of row tiles (`pvx`) | no change at L768 / S5 (45.3 vs 43.4 us): the kernel was latency bound, not L2-byte bound |
| `pvx` with the samples as 64-column wgmma per sample | 25 % of peak tensor rate; the wide-N form removed the instruction-shape limit but the epilogue was the real cost |
| dV and dbias as two batched cuBLAS GEMMs (samples folded into columns) for bf16 | +25 us a step: the permute copies cost what the GEMMs save (kept for fp32, where it is the only core) |
| dh GEMM with the SwiGLU backward in its epilogue (`dswi`, 128 x 128 tiles, a / b fragments in registers) | 509 vs 318 us at M = 36864 (re-reads ~1.4 GB from L2) |
| expand GEMM + SwiGLU epilogue storing a | b (`expand_pipe`) in training | 813 us vs 380 for cuBLAS + `swiglu_rows` |
| the pair part (pair bias, softmax) on a side stream beside the GEMM chain | +30 us: GEMM CTAs hold the shared memory of every SM |
| weight-gradient GEMMs split over the rows (B200's default split 4) | +750 us a step on H100 (TF32 `nt` kernels: 157-210 TFLOP/s one GEMM, 135-190 split) |
| K-major (explicitly transposed) operands for the TF32 weight gradients | GEMMs 784 us faster, transposes cost 800+ us |
| fp32 expand + SwiGLU on `gemm_swiglu_sm90`'s TF32 build | -16 us but output error 2.9e-4 -> 4.6e-4 (its TMA-fed operands are truncated, not rounded) |
| a custom TF32 weight-gradient kernel | net 1-3 % after the transposes and partial-sum reduce (computed, not built) |

## Limits and next

- bf16 training is at ~75 % of the roofline and 1.15x PyTorch compiled; B200's best (fused backward tails) is 1.26x, and its
  fused tail measured neutral against separate launches. The remaining headroom is `pvx` (an epilogue that overlaps the next
  item's MMAs) and the GEMM epilogue fusions above.
- fp32 training is bound by cuBLAS's TF32 `nt` weight-gradient kernels (3.3 ms of 10.3).
- Only 16 x 48 heads are served on H100 (the 24 x 32, 12 x 64 and 16 x 64 layouts of B200 fall back to PyTorch).
