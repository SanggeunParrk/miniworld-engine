# Token DiT on B200 (sm100)

Kernel-level status of the token DiT block (AF3 Alg. 23: AdaLN + AugmentedAttention with pair bias + conditioned SwiGLU
transition; d_single 768, d_cond 384, d_pair 128, 16 heads x 48, transition n = 2) on B200; the module-level summary is in
[b200.md](../b200.md). Columns are (Length, Dimension, dtype) from the shape registry (`token_single`, d_hidden 768).

On B200 the token DiT runs **hand-written CUDA and cuBLAS only** -- no Triton, no quack -- on two paths that
`modules/dit` dispatches to:

- **Inference** (`integrations/token_dit.py` -> `kernels/conditioned_transition/triton/token_dit_runner.py`, the fused
  step runner shared with H100): serves `DiTBlock` calls with no autograd, the engine's kernel backend (implementation
  TRITON or MINIWORLD), one conditioning shared by the samples. On capability 10.0 the runner takes the CUDA row kernels
  (`kernels/conditioned_transition/cuda/token_dit_rows.cu`), the sm_100a gated attention core
  (`kernels/augmented_attention/cuda/sm100/attn_inf.cu`) in bf16, and from M = S L >= 3840 the expand GEMM with the
  SwiGLU in its epilogue (`kernels/conditioned_transition/cuda/gemm_swiglu2_sm100.cu`); the other GEMMs are cuBLAS.
  Inference only (the token DiT is not recycled), so the block's weight pack and its pair bias are made once and reused
  while the weights / the pair tensor are unchanged (keyed on pointer and version), CUDA-graph replays included: a replay
  reads the pack and pair bias it was captured with and the live single / cond.
- **Training** (`integrations/token_dit_train.py`, new): serves `DiTBlock` calls under autograd, engine kernel backend
  (implementation TRITON or MINIWORLD), bf16 (inputs, or `compute_dtype`), B == 1, even A, L % 128 == 0, key mask [B, L]
  or none. One autograd Function per block whose forward and backward are each one opaque op (torch.compile keeps them
  as nodes): cuBLAS GEMMs (bf16 operands, fp32 accumulation, fp32 weight gradients), the sm_100a attention forward /
  backward (`attn_fwd2`, `attn_dqb`, `attn_dkv`), and 16 CUDA row kernels
  (`kernels/conditioned_transition/cuda/token_dit_train_rows.cu`). The key mask folds into the pair bias as -inf.
- The attention module alone (`AugmentedAttentionPairBias`, `compute_dtype=bf16`, no mask, even A) also takes the sm_100a
  forward / backward (`settings.augmented_attention_bf16_sm100`).

"미검증" = the kernels accept the shape but no test or measurement has run there. cache build ✓: nothing on these paths
autotunes -- the CUDA row kernels have fixed launch shapes, the sm_100a kernels are cubins built on first use into
`MINIWORLD_ENGINE_JIT_ROOT` (keyed by source and flags), the SwiGLU GEMM choice is a fixed row threshold. fp32 inference
on B200 keeps the Triton gated core (the sm_100a cores are bf16 only).

Switches (all default on): `MINIWORLD_TOKEN_DIT_ROWS_CUDA`, `MINIWORLD_AUGATTN_BF16_SM100`,
`MINIWORLD_TOKEN_DIT_GEMM_SWIGLU` (`auto` = M >= 3840, `1` forces, `0` off), `MINIWORLD_TOKEN_DIT_TRAIN`.

## Inference

### I1 · attention core `attn_inf` (softmax(qk^T + bias) v, gated by sigmoid(g), written over q)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I2 · row kernels `adaln_rows`, `resgate_adaln_rows`, `swiglu_rows`, pair LayerNorm (+ one cuBLAS projection)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I3 · expand GEMM + SwiGLU `gemm_swiglu2_sm100` (M >= 3840; below it cuBLAS + `swiglu_rows`)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

(at S = 5; the choice follows M = S L, so L = 768 is the first registry length above the threshold.)

## Training

### T1 · attention forward `attn_fwd2` · T2 · backward dQ + dbias `attn_dqb` · T3 · backward dK + dV `attn_dkv`

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T4 · row kernels (`token_dit_train_rows.cu`, 16 kernels)

Forward `cond_prep`, `adaln_a`, `qknorm`, `gate_o`, `res_adaln_b`, `res_c`, `pair_ln` (+ `swiglu_rows`); backward
`res_c_bwd`, `swiglu_bwd`, `res_adaln_b_bwd`, `gate_o_bwd` (also the core backward's dO and D), `qknorm_bwd`,
`adaln_a_bwd`, `cond_bwd`, `unfold_lnw`, `pair_ln_bwd`. Per-column gradient sums leave each block of rows as one row of a
partial buffer, summed once on the host side.

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Measurements (2026-09-29)

B200 (148 SMs, 1000 W power cap), the v2.2.0 pixi env (torch 2.13.0+cu129), GPU 6 through `gpuq` with nothing else on
it. ms per block, median. Run-to-run spread on this card is +-3-5 % (power cap), so differences below that are not
resolved.

All rows from `benchmarks/runners/bench.py target=dit level=module` (one DiTBlock, bf16, key mask 12.5 % masked), in one
run. "ours" is the bench's `miniworld` row (implementation TRITON: the engine's kernels). cuEquivariance has no DiT block
(no column). × = ours vs the fastest other row.

### Training (A = 48, compiled; CUDA graph off / on)

| (Length, Dimension, dtype) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (384, 768, bf16) | 3.087 / 2.856 | — | no backward | **1.934 / 1.767** | 1.60 / 1.62 |
| (768, 768, bf16) | 8.540 / 8.315 | — | no backward | **4.096 / 4.046** | 2.08 / 2.06 |

Before this path (engine module path on B200): 2.502 / 2.289 and 6.036 / 5.806 ms. Accuracy
(`tests/integrations/test_b200_token_dit_train_gpu.py`, against the fp32 PyTorch block): output and every input /
parameter gradient within 1.09x of the engine module path's own error in the same bf16 regime (qk-norm on / off, key mask
on / off); the compiled block matches eager.

### Inference, one conditioning shared by the S = 5 samples (a sampling step; `+shared_cond=true`)

Both fused rows compute a layer's pair bias once per pair (their samplers do, and so does ours) and the conditioning once
per token. Anthropic runs `compile=false` (it has no compilable graph); PyTorch and ours are given compile=false /
compiled, both with CUDA graphs.

| (Length, Dimension, dtype) | PyTorch | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (384, 768, bf16) | 0.500 eager / 0.165 compiled | — | 0.123 | **0.090** | 1.36 |
| (768, 768, bf16) | 1.570 eager / 0.373 compiled | — | 0.208 | **0.137** | 1.51 |

Before (per-call weight pack and pair bias, Triton rows and core): 0.258 / 0.403 ms.

### Inference, a different conditioning per sample (the bench default)

The fused step does not serve per-sample conditioning; ours is the module composition (Triton kernels, bf16 attention
core not requested) -- no B200 CUDA path yet.

| (Length, Dimension, dtype) | PyTorch | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (384, 768, bf16) | 0.479 eager / 0.164 compiled | — | **0.145** | 0.188 / 0.170 | 0.85 |
| (768, 768, bf16) | 1.539 eager / 0.369 compiled | — | **0.285** | 0.362 / 0.330 | 0.86 |

### Inference step inside the runner (24 blocks, S = 5, CUDA graph; probe, per block)

| (Length, Dimension, dtype) | before (Triton rows + Triton core) | ours | × |
|---|---|---|---|
| (384, 768, bf16) | 62.9 us | **60.0 us** | 1.05 |
| (768, 768, bf16) | 115.9 us | **101.8 us** | 1.14 |

Per block at L768: attention core 33 us, cuBLAS GEMMs ~45 us, `gemm_swiglu2` 18.4 us (cuBLAS expand + row pass: 22.3),
`resgate_adaln_rows` x2 20.5 us.

## What was tried and not kept

| attempt | result |
|---|---|
| row kernels with one warp per row | 2x slower than Triton (13 warps per SM at M = 1920: latency-bound); one block per row fixed it |
| pair bias as one fused WMMA kernel (LayerNorm + projection) | 3.4-3.6x slower than LayerNorm + cuBLAS; dropped |
| 1-CTA SwiGLU GEMM (`gemm_swiglu_sm100.cu`, kept for the record) | 540 TFLOPS: a 128 x 256 tile per SM is TMA-intake-bound; the 2-CTA kernel replaced it |
| `gemm_swiglu2` at L384 (M = 1920) | 14.0 us vs 13.7 for cuBLAS + row pass: 96 items on 74 SM pairs leave most pairs idle in round 2 |
| training column sums by atomicAdd per block | every block onto the same 768 addresses serialised in L2 (`qknorm_bwd` 736 us, `res_c_bwd` 105 us at L384); partial buffers instead |

## Limits and next

- A DiTBlock call sees one block, so the pair bias and the cond LayerNorm are recomputed per block; the research stack
  runner (branch `b200/token-dit`) hoisted them for the whole stack (training 1.62 / 3.56 ms per block there, with Triton
  and quack).
- Training SwiGLU is cuBLAS + CUDA row kernels; `gemm_swiglu2`'s `-DSAVE_AB` variant (writes a, b) is the next step for the
  forward.
- Per-sample conditioning at inference has no B200 CUDA path (Anthropic 1.17x faster there): the training block's forward
  without the saves would serve it.
- Not done: QK-norm in the inference path.
