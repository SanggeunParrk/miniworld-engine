# Token DiT on B200 (sm100)

Kernel-level status of the token DiT block (AF3 Alg. 23: AdaLN + AugmentedAttention with pair bias + conditioned SwiGLU
transition; d_single 768, d_cond 384, d_pair 128, 16 heads x 48, transition n = 2) on B200; the module-level summary is in
[b200.md](../b200.md). Columns are (Length, Dimension, dtype) from the shape registry (`token_single`, d_hidden 768). The
same paths also serve three other head layouts in bf16 -- 24 x 32 and 12 x 64 at d_single 768, 16 x 64 at d_single 1024 --
described in "Head layouts" below; everything else on this page is the 16 x 48 registry row.

Summary (2026-09-30). Inference and training, bf16 and fp32 (fp32 = TF32 tensor cores with fp32 softmax, residual and
LayerNorm), QK-norm on or off; inference at any L (L128-768 tested and measured), training at L % 128 == 0 (measured at
L384 / L768, A = 48). Against the fastest other implementation (`bench.py`): inference over Anthropic bf16 1.37-1.72x, fp32
1.02-1.29x (with QK-norm, which Anthropic lacks, 1.7-2.9x over PyTorch compiled); training over PyTorch compiled bf16
1.65-2.17x, fp32 1.53-1.83x. Time-roofline SoL of the step: inference 16-39 % bf16 / 19-43 % fp32 (small L is latency),
training 51-52 % bf16 / 53-57 % fp32 (cuBLAS at its power-capped 58-61 %, the attention at 14-47 %).

Head layouts (2026-10-01). 24 x 32, 12 x 64 (d 768) and 16 x 64 (d 1024) run the same two paths in bf16, inference and
training, built from the same kernels with the head count and width as compile-time flags; 16 x 48 is unchanged. Inference
leads Anthropic by 1.30-1.56x and training (CUDA graph on) PyTorch compiled by 1.20-2.73x at L256-768. fp32 stays 16 x 48
only.

On B200 the token DiT runs **hand-written CUDA and cuBLAS only** -- no Triton, no quack -- on two paths that
`modules/dit` dispatches to:

- **Inference** (`integrations/token_dit.py` -> `kernels/conditioned_transition/triton/token_dit_runner.py`, the fused
  step runner shared with H100): serves `DiTBlock` calls with no autograd, the engine's kernel backend (implementation
  TRITON or MINIWORLD), bf16 or fp32, any L >= 8, the conditioning shared by the samples (L table rows) or one per sample
  (S L rows). The core's TMA maps are 3-D per sample (tile tails load as zeros, stores clip, keys past L are masked), so it
  takes any multiple of 8; `integrations/token_dit.py` pads other lengths to the next multiple of 8 (zero single / cond /
  pair rows, the key mask extended with False) and returns the first L rows. QK-norm blocks (`use_qk_norm`) take the
  same step: one in-place CUDA pass (`qknorm_rows`, RMSNorm of every 48-wide q / k head, eps as the module's
  `effective_eps`) right after the projection GEMM, with the sm_scale log2 e fold moved from Wq / bq into the q norm's
  weight (the norm would cancel it). Off B200 QK-norm keeps the module path.
  On capability 10.0 the runner takes the CUDA row kernels (`kernels/conditioned_transition/cuda/token_dit_rows.cu`, both
  dtypes; the first pass reads the step's input and the last writes its output, so the step has no PyTorch kernels), the
  sm_100a gated attention core -- `kernels/augmented_attention/cuda/sm100/attn_inf.cu` in bf16, `attn_inf_tf32.cu` (TF32
  tensor cores, fp32 softmax) in fp32 -- and, in bf16 from M = S L >= 3840, the expand GEMM with the SwiGLU in its epilogue
  (`kernels/conditioned_transition/cuda/gemm_swiglu2_sm100.cu`); the other GEMMs are cuBLAS (fp32: TF32, forced by the
  runner whatever the caller's `allow_tf32`). fp32 packs the projection as q | k | g | v: one GEMM writes q | k | g, a
  second writes v^T [768, S L] into the same buffer (see "What was tried" for why v goes K-major there).
  Inference only (the token DiT is not recycled), so the block's weight pack and its pair bias are made once and reused
  while the weights / the pair tensor are unchanged (keyed on pointer and version), CUDA-graph replays included: a replay
  reads the pack and pair bias it was captured with and the live single / cond.
- **Training** (`integrations/token_dit_train.py`): serves `DiTBlock` calls under autograd, engine kernel backend
  (implementation TRITON or MINIWORLD), bf16 (inputs, or `compute_dtype=bf16`) or fp32 (fp32 inputs, no `compute_dtype`),
  B == 1, even A, L % 128 == 0, key mask [B, L] or none. One autograd Function per block whose forward and backward are
  each one opaque op (torch.compile keeps them as nodes): cuBLAS GEMMs (operands in the path's dtype -- fp32 ones forced to
  TF32 -- fp32 accumulation, fp32 weight gradients; in bf16 the expand GEMM is `gemm_swiglu2_sm100 -DSAVE_AB`, the SwiGLU in
  its epilogue and [a | b] saved for the backward), the sm_100a attention forward / backward (bf16: `attn_fwd2`,
  `attn_dkv` + `attn_dqb`; fp32: `attn_fwd_tf32`, `attn_dkv_tf32` + `attn_dqb_tf32`, TF32 tensor cores with an fp32
  softmax; without QK-norm they read q / k / v as column views of the projection output), the CUDA `glue.cu` passes (bf16:
  the bias transpose `attn_dkv` reads), and the CUDA row kernels of
  `kernels/conditioned_transition/cuda/token_dit_train_rows.cu`, templated on the operand dtype. The key mask folds into
  the pair bias as -inf.
- The attention module alone (`AugmentedAttentionPairBias`, `compute_dtype=bf16`, no mask, even A) also takes the sm_100a
  bf16 forward / backward (`settings.augmented_attention_bf16_sm100`), with the dO prep in `glue.cu` (it was Triton).

"미검증" = the kernels accept the shape but no test or measurement has run there. 성능 확인: see "Hardware limit (SoL) per
kernel" (△ = fastest measured, a kernel of the group below 70 % SoL; — = not measured). cache build ✓: nothing on these paths
autotunes -- the CUDA row kernels have fixed launch shapes, the sm_100a kernels are cubins built on first use into
`MINIWORLD_ENGINE_JIT_ROOT` (keyed by source and flags), the SwiGLU GEMM choice is a fixed row threshold.

Switches (all default on): `MINIWORLD_TOKEN_DIT_ROWS_CUDA`, `MINIWORLD_AUGATTN_BF16_SM100`,
`MINIWORLD_TOKEN_DIT_GEMM_SWIGLU` (`auto` = M >= 3840, `1` forces, `0` off), `MINIWORLD_TOKEN_DIT_TRAIN`.

## Inference

### I1 · attention core `attn_inf` (softmax(qk^T + bias) v, gated by sigmoid(g), written over q)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I1' · attention core `attn_inf_tf32` (fp32: TF32 QK and PV, fp32 softmax; q | k | g + v^T)

| (Length, Dimension, dtype) | (128, 768, fp32) | (256, 768, fp32) | (384, 768, fp32) | (512, 768, fp32) | (640, 768, fp32) | (768, 768, fp32) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Both cores against fp64 (`tests/numerics/test_augmented_attention_bf16_sm100_gpu.py::test_gated_inference_core_matches_fp64`,
S L = 5 x 384, 2 x 256, 3 x 128, 3 x 200, 2 x 520, 5 x 136): bf16 within 1.2e-2, fp32 within 2e-3 relative. The step at lengths
that are not multiples of 128 (`tests/integrations/test_b200_token_dit_gpu.py::test_token_dit_any_length`, L = 136, 200,
333, 517, 700, both dtypes, shared / per-sample conditioning, key mask, CUDA-graph replay) matches the PyTorch block. Every
length between the table's columns runs the same code.

### I2 · row kernels `adaln_in_rows`, `resgate_adaln_rows`, `resgate_out_rows`, `layernorm_rows`, `swiglu_rows`, `qknorm_rows`, pair LayerNorm (+ one cuBLAS projection)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The same row kernels serve fp32 at every length above.

### I3 · expand GEMM + SwiGLU `gemm_swiglu2_sm100` (M >= 3840; below it cuBLAS + `swiglu_rows`)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

(at S = 5; the choice follows M = S L, so L = 768 is the first registry length above the threshold.)

## Training

### T1 · attention forward `attn_fwd2` · T2 · backward dQ + dbias `attn_dqb` · T3 · backward dK + dV `attn_dkv`

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | — | — | △ | — | — | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T1' · fp32: `attn_fwd_tf32` · T2' · `attn_dqb_tf32` · T3' · `attn_dkv_tf32` (TF32 tensor cores, fp32 softmax)

| (Length, Dimension, dtype) | (128, 768, fp32) | (256, 768, fp32) | (384, 768, fp32) | (512, 768, fp32) | (640, 768, fp32) | (768, 768, fp32) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | — | — | △ | — | — | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Against fp64 (`tests/numerics/test_augmented_attention_tf32_sm100_gpu.py`, A L = 2 x 128, 4 x 256, 2 x 384, key mask on /
off): O within 2e-3, dq / dk / dv / dbias within 3e-3 relative; LSE within 5e-3 absolute (TF32 drops the operands' low
mantissa bits, so the logits shift by ~1e-3; the backward recomputes P from the same operands). Kernel times at A = 48
(CUDA events, one process): forward 121 / 385 us, backward (dkv + dqb) 452 / 1554 us at L384 / L768 (472 / 1706 with the first
dkv configuration: two K / V slots, 2 stages), against 71 / 211 and 214 / 712 us for the bf16 kernels.

How the fp32 tiles differ from the bf16 ones: a 48-wide fp32 row is 192 B, so every K-major tile is a 32-column box in the
128-B swizzle plus a 16-column box in the 64-B swizzle. A tf32 MMA reads an MN-major operand (v in PV, q / dO in dK / dV, K
in dQ) only in the 128-B swizzle with 32-B atoms (TMA `SWIZZLE_128B_ATOM_32B`, UMMA layout type 1): those tiles are loaded
a second time as two 32-column boxes. To fit, the blocks are 32 keys (forward) / 32 queries (dkv) / a 64-key chunk (dqb),
dkv reads the bias as given (no transposed copy), keeps one K / V slot with 3 q / dO / bias stages and stages dK / dV in
the finished item's K / V slot, and dqb reduces dQ with per-thread v4 reductions.

### T4 · row kernels (`token_dit_train_rows.cu`, 16 kernels)

Forward `cond_prep`, `adaln_a` (reads the block's input, writes the fp32 residual), `qknorm` (QK-norm on only), `gate_o`,
`res_adaln_b`, `res_c`, `pair_ln` (+ `swiglu_rows` in fp32; bf16 has the SwiGLU in the expand GEMM); backward
`res_c_bwd`, `swiglu_bwd`, `res_adaln_b_bwd`, `gate_o_bwd` (also the core backward's dO and D), `qknorm_bwd`,
`adaln_a_bwd`, `cond_bwd`, `unfold_lnw`, `pair_ln_bwd`. Per-column gradient sums leave each block of rows as one row of a
partial buffer, summed once on the host side.

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | — | — | △ | — | — | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The same kernels serve fp32 (every GEMM operand and saved activation fp32): CUDA at (256, 768, fp32), (384, 768, fp32) and
(768, 768, fp32); 128 / 512 / 640 미검증.

## Measurements (2026-09-29 / 30)

B200 (148 SMs, 1000 W power cap), the v2.2.0 pixi env (torch 2.13.0+cu129), one GPU through `gpuq` with nothing else on
it. ms per block, median. Run-to-run spread on this card is +-3-5 % (power cap), so differences below that are not
resolved.

All rows from `benchmarks/runners/bench.py target=dit level=module` (one DiTBlock); each section states its settings. "ours"
is the bench's `miniworld` row (implementation TRITON: the engine's kernels). cuEquivariance has no DiT block (no column).
× = ours vs the fastest other row.

### Training (2026-09-30; A = 48, compiled; CUDA graph off / on)

`bench.py target=dit level=module mode=training`, bf16 = `precision=bf16-mixed`, fp32 = `precision=32` (TF32 allowed, the
bench default; the fused path runs its fp32 GEMMs on TF32 either way), QK-norm = `+dit_qk_norm=true`. Anthropic has no
backward; cuEquivariance has no DiT block.

| (Length, Dimension, dtype) | QK-norm | PyTorch compiled | ours | × |
|---|---|---|---|---|
| (384, 768, bf16) | off | 3.085 / 2.850 | **1.833 / 1.702** | 1.68 / 1.67 |
| (768, 768, bf16) | off | 8.524 / 8.292 | **3.924 / 3.848** | 2.17 / 2.15 |
| (384, 768, bf16) | on | 3.204 / 2.956 | **1.943 / 1.771** | 1.65 / 1.67 |
| (768, 768, bf16) | on | 8.724 / 8.481 | **4.074 / 3.900** | 2.14 / 2.17 |
| (384, 768, fp32) | off | 4.682 / 4.553 | **2.993 / 2.958** | 1.56 / 1.54 |
| (768, 768, fp32) | off | 12.359 / 12.150 | **6.755 / 6.695** | 1.83 / 1.81 |
| (384, 768, fp32) | on | 4.810 / 4.636 | **3.125 / 3.037** | 1.54 / 1.53 |
| (768, 768, fp32) | on | 12.550 / 12.389 | **7.080 / 6.959** | 1.77 / 1.78 |

Earlier the same day: bf16 1.934 / 1.767 and 4.096 / 4.046 ms, fp32 3.144 / 3.022 and 7.139 / 6.801 ms (QK-norm off). Before
this path (engine module path on B200, bf16): 2.502 / 2.289 and 6.036 / 5.806 ms. What moved it since:
- the input's fp32 copy folded into `adaln_a` (it reads the input and writes the residual on the way);
- the 23 parameter-gradient copies (custom-op outputs may not alias the shared GEMM buffers) as one multi-tensor copy;
- without QK-norm, the attention reads q / k / v as column views of the q|k|v|g GEMM output (no copy pass; the TMA maps take
  the row stride);
- bf16: the expand GEMM carries the SwiGLU and saves [a | b] (`gemm_swiglu2_sm100 -DSAVE_AB`) instead of cuBLAS + `swiglu_rows`;
- fp32: `attn_dkv_tf32` with one K / V slot and 3 stages (see T1'-T3').

Accuracy (`tests/integrations/test_token_dit_train_gpu.py`, against the fp32 IEEE PyTorch block; bf16 and fp32, QK-norm
on / off, key mask on / off): output and every input / parameter gradient (the q / k norm weights included) at most 1.09x
the engine module path's own error in the same regime -- bf16 worst ~7e-3 (expand weights), fp32 worst 1.8e-3
(`norm_key.weight`); the compiled block matches eager.

### Inference over L = 128-768 (2026-09-30)

`bench.py target=dit level=module mode=inference`, S = 5, key mask 20 %, one GPU through `gpuq`. Lengths 128-768 in steps
of 128 plus 200 / 450 / 700 (not multiples of 128; 450 and 700 not of 8 either: the pad-to-8 path). bf16 =
`precision=bf16-mixed`, fp32 = `precision=32` (TF32 allowed). Both fused rows compute a layer's pair bias once per pair
tensor. Anthropic runs `compile=false` (it has no compilable graph); PyTorch eager and compiled, with CUDA graphs. Ours and
Anthropic from one run (after the glue folds below); PyTorch from the earlier run of the same day. × = ours against the
faster of Anthropic and PyTorch compiled. Output error against the fp32 IEEE PyTorch block, every length: bf16 ours
3.3-3.6e-3 (Anthropic 3.4e-3), fp32 ours 2.9e-4 (Anthropic 3.1-3.4e-4).

One conditioning shared by the samples (a sampling step; `+shared_cond=true`), bf16:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16) | 0.186 / 0.094 | — | 0.084 | **0.053** | 1.58 |
| (200, 768, bf16) | 0.247 / 0.113 | — | 0.102 | **0.061** | 1.67 |
| (256, 768, bf16) | 0.303 / 0.125 | — | 0.106 | **0.065** | 1.63 |
| (384, 768, bf16) | 0.500 / 0.164 | — | 0.123 | **0.076** | 1.62 |
| (450, 768, bf16) | 0.678 / 0.219 | — | 0.151 | **0.108** | 1.40 |
| (512, 768, bf16) | 0.772 / 0.223 | — | 0.155 | **0.100** | 1.55 |
| (640, 768, bf16) | 1.149 / 0.299 | — | 0.174 | **0.113** | 1.55 |
| (700, 768, bf16) | 1.440 / 0.373 | — | 0.203 | **0.137** | 1.48 |
| (768, 768, bf16) | 1.570 / 0.371 | — | 0.207 | **0.121** | 1.71 |

A different conditioning per sample (S L table rows; Anthropic's conditioning dedup off), bf16:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16) | 0.170 / 0.094 | — | 0.086 | **0.055** | 1.56 |
| (200, 768, bf16) | 0.229 / 0.113 | — | 0.104 | **0.063** | 1.65 |
| (256, 768, bf16) | 0.285 / 0.125 | — | 0.110 | **0.069** | 1.59 |
| (384, 768, bf16) | 0.477 / 0.166 | — | 0.131 | **0.082** | 1.60 |
| (450, 768, bf16) | 0.654 / 0.219 | — | 0.160 | **0.117** | 1.37 |
| (512, 768, bf16) | 0.750 / 0.221 | — | 0.168 | **0.108** | 1.55 |
| (640, 768, bf16) | 1.123 / 0.297 | — | 0.194 | **0.123** | 1.58 |
| (700, 768, bf16) | 1.409 / 0.371 | — | 0.223 | **0.153** | 1.46 |
| (768, 768, bf16) | 1.538 / 0.368 | — | 0.229 | **0.133** | 1.72 |

Shared conditioning, fp32:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, fp32) | 0.211 / 0.139 | — | 0.104 | **0.082** | 1.28 |
| (200, 768, fp32) | 0.281 / 0.174 | — | 0.119 | **0.092** | 1.29 |
| (256, 768, fp32) | 0.334 / 0.190 | — | 0.121 | **0.100** | 1.20 |
| (384, 768, fp32) | 0.544 / 0.281 | — | 0.156 | **0.123** | 1.27 |
| (450, 768, fp32) | 0.741 / 0.348 | — | 0.174 | **0.166** | 1.05 |
| (512, 768, fp32) | 0.807 / 0.362 | — | 0.178 | **0.160** | 1.12 |
| (640, 768, fp32) | 1.169 / 0.508 | — | 0.215 | **0.192** | 1.12 |
| (700, 768, fp32) | 1.412 / 0.586 | — | 0.226 | **0.221** | 1.02 |
| (768, 768, fp32) | 1.547 / 0.625 | — | 0.254 | **0.209** | 1.22 |

Per-sample conditioning, fp32:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, fp32) | 0.199 / 0.139 | — | 0.100 | **0.086** | 1.17 |
| (200, 768, fp32) | 0.266 / 0.174 | — | 0.121 | **0.098** | 1.23 |
| (256, 768, fp32) | 0.317 / 0.188 | — | 0.127 | **0.108** | 1.17 |
| (384, 768, fp32) | 0.524 / 0.277 | — | 0.170 | **0.137** | 1.24 |
| (450, 768, fp32) | 0.719 / 0.346 | — | 0.190 | **0.184** | 1.03 |
| (512, 768, fp32) | 0.782 / 0.360 | — | 0.196 | **0.176** | 1.12 |
| (640, 768, fp32) | 1.142 / 0.506 | — | 0.246 | **0.215** | 1.14 |
| (700, 768, fp32) | 1.383 / 0.584 | — | 0.260 | **0.252** | 1.03 |
| (768, 768, fp32) | 1.516 / 0.624 | — | 0.289 | **0.235** | 1.23 |

bf16 leads at every length (1.37-1.72x), fp32 too, by less (1.02-1.29x; 1.02-1.05x at 450 / 700, where the last 128-query
tile is mostly padding: L = 456 / 704 after the pad to 8, 4 and 6 tiles for 3.6 and 5.5 tiles of rows). What changed on
2026-09-30 afternoon: the step's PyTorch glue (17-21 % of the bf16 step: the residual copied in and out, the conditioning
LayerNorm and its casts) moved into the CUDA row passes -- `adaln_in_rows` reads the input and writes the fp32 residual
with the first AdaLN, `resgate_out_rows` writes the last residual straight into the output dtype, `layernorm_rows` takes
the conditioning in any dtype -- bf16 L384 0.090 -> 0.076 ms, L768 0.139 -> 0.121 ms; fp32 gained 2-5 %. Before per-sample
conditioning took the fused step (the module composition, bf16): 0.170-0.188 / 0.330-0.362 ms at L384 / L768.

The fp32 step on B200 runs: CUDA row kernels, cuBLAS TF32 GEMMs (q | k | g, v^T, out, expand, squeeze; forced to TF32 by the
runner, whatever the caller's `allow_tf32`: with IEEE fp32 GEMMs the step ran 5-8x slower), CUDA `swiglu_rows`, the
`attn_inf_tf32` core; the pair bias once per pair tensor (CUDA LayerNorm + cuBLAS projection).

### Inference with QK-norm (2026-09-30; `+dit_qk_norm=true`)

Same bench and setup, blocks with `use_qk_norm=True` (`dit_qk_norm`, a bench switch added for this). Anthropic's composition
has no q / k RMSNorm and refuses, so × is against PyTorch compiled (PyTorch from the earlier run of the day, ours after the
glue folds). The QK-norm pass costs ours 2-6 us a block. Output error against the fp32 IEEE PyTorch block: bf16 ours
3.3-3.5e-3 (PyTorch bf16 4.2e-3), fp32 ours 2.9e-4.

Shared conditioning, bf16:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16) | 0.203 / 0.096 | — | no QK-norm | **0.055** | 1.74 |
| (256, 768, bf16) | 0.334 / 0.129 | — | no QK-norm | **0.070** | 1.85 |
| (384, 768, bf16) | 0.542 / 0.168 | — | no QK-norm | **0.080** | 2.10 |
| (512, 768, bf16) | 0.831 / 0.227 | — | no QK-norm | **0.104** | 2.18 |
| (640, 768, bf16) | 1.222 / 0.302 | — | no QK-norm | **0.119** | 2.54 |
| (768, 768, bf16) | 1.656 / 0.375 | — | no QK-norm | **0.127** | 2.95 |

Per-sample conditioning, bf16:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16) | 0.188 / 0.096 | — | no QK-norm | **0.057** | 1.68 |
| (256, 768, bf16) | 0.315 / 0.129 | — | no QK-norm | **0.074** | 1.75 |
| (384, 768, bf16) | 0.520 / 0.168 | — | no QK-norm | **0.086** | 1.95 |
| (512, 768, bf16) | 0.807 / 0.227 | — | no QK-norm | **0.115** | 1.98 |
| (640, 768, bf16) | 1.195 / 0.299 | — | no QK-norm | **0.129** | 2.32 |
| (768, 768, bf16) | 1.625 / 0.375 | — | no QK-norm | **0.139** | 2.69 |

Shared conditioning, fp32:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, fp32) | 0.227 / 0.141 | — | no QK-norm | **0.082** | 1.73 |
| (256, 768, fp32) | 0.363 / 0.194 | — | no QK-norm | **0.102** | 1.90 |
| (384, 768, fp32) | 0.589 / 0.285 | — | no QK-norm | **0.129** | 2.21 |
| (512, 768, fp32) | 0.866 / 0.369 | — | no QK-norm | **0.164** | 2.25 |
| (640, 768, fp32) | 1.243 / 0.512 | — | no QK-norm | **0.199** | 2.58 |
| (768, 768, fp32) | 1.634 / 0.631 | — | no QK-norm | **0.215** | 2.93 |

Per-sample conditioning, fp32:

| (Length, Dimension, dtype) | PyTorch eager / compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, fp32) | 0.215 / 0.143 | — | no QK-norm | **0.086** | 1.67 |
| (256, 768, fp32) | 0.346 / 0.194 | — | no QK-norm | **0.112** | 1.73 |
| (384, 768, fp32) | 0.567 / 0.282 | — | no QK-norm | **0.143** | 1.97 |
| (512, 768, fp32) | 0.840 / 0.367 | — | no QK-norm | **0.180** | 2.04 |
| (640, 768, fp32) | 1.212 / 0.511 | — | no QK-norm | **0.221** | 2.31 |
| (768, 768, fp32) | 1.603 / 0.635 | — | no QK-norm | **0.242** | 2.63 |

Tests: `tests/integrations/test_b200_token_dit_gpu.py::test_token_dit_qk_norm` (bf16 / fp32, L = 384, 200, 333, shared /
per-sample conditioning, key mask; the pass runs and a changed norm weight changes the output).

### Inference step inside the runner (2026-09-29 probe, 24 blocks, S = 5, CUDA graph, per block; before the glue folds)

| (Length, Dimension, dtype) | before (Triton rows + Triton core) | ours | × |
|---|---|---|---|
| (384, 768, bf16) | 62.9 us | **60.0 us** | 1.05 |
| (768, 768, bf16) | 115.9 us | **101.8 us** | 1.14 |

Per block at L768: attention core 33 us, cuBLAS GEMMs ~45 us, `gemm_swiglu2` 18.4 us (cuBLAS expand + row pass: 22.3),
`resgate_adaln_rows` x2 20.5 us.

### Hardware limit (SoL) per kernel (2026-09-30)

Time roofline, the TriMul doc's method (no power readings). floor = max(minimum HBM bytes / BW, FLOPs / tensor rate, exp2
count / MUFU rate) from each kernel's inputs, outputs, GEMM work and softmax exponentials; SoL = floor / measured. Ceilings:
HBM 6.96 TB/s (the fastest effective bandwidth of a large, >= 256 MB, elementwise kernel in these runs; an fp32 copy
reached 6.31 TB/s next to them), tensor 2.23 PF/s bf16 (resident-operand MMA) and 1.115 PF/s TF32 (half, the spec
ratio), MUFU 31 ex2 / ns / SM (measured: 16 / clk). Measured: kernel durations from torch.profiler over back-to-back steps
(median per call site; a kernel called twice per step, as `resgate_adaln_rows`, sums both), inference S = 5 with one shared
conditioning and a 20 % key mask, training A = 48, forward + backward, TF32 allowed as in `bench.py`. The weight-side
`unfold_lnw` (13 us, fixed) has no data floor worth the name. "*" = above the HBM floor: its input was written just before
and is L2-resident. "step" = the kernels' SoL weighted by their time (PyTorch glue counted at 0). Script:
`sol_tdit.py` + `tables.py` (B200 scratch `scratch/int/sol`).

Inference, bf16:

| kernel | bound | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|---|
| attn_inf | HBM / MUFU | 10 % | 17 % | 22 % | 18 % | 23 % | 30 % |
| adaln_in_rows | HBM | 21 % | 32 % | 35 % | 38 % | 40 % | 44 % |
| layernorm_rows | HBM | 1 % | 2 % | 4 % | 5 % | 6 % | 7 % |
| qknorm_rows | HBM | 20 % | 29 % | 33 % | 36 % | 39 % | 40 % |
| resgate_adaln_rows | HBM | 26 % | 39 % | 43 % | 48 % | 51 % | 53 % |
| resgate_out_rows | HBM | 25 % | 37 % | 46 % | 48 % | 57 % | 62 % |
| swiglu_rows | HBM | 29 % | 42 % | 50 % | 54 % | 57 % | — |
| cuBLAS GEMMs | tensor | 14 % | 23 % | 30 % | 32 % | 38 % | 39 % |
| gemm_swiglu2 | tensor | — | — | — | — | — | 43 % |
| **step (time-weighted)** | | 16 % | 25 % | 32 % | 32 % | 37 % | 39 % |
| step kernel time, us | | 49.1 | 63.1 | 75.0 | 102.2 | 114.3 | 120.9 |

Inference, fp32:

| kernel | bound | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|---|
| attn_inf_tf32 | HBM | 15 % | 23 % | 30 % | 20 % | 23 % | 24 % |
| adaln_in_rows | HBM | 32 % | 43 % | 48 % | 56 % | 59 % | 63 % |
| layernorm_rows | HBM | 3 % | 5 % | 7 % | 9 % | 11 % | 13 % |
| qknorm_rows | HBM | 39 % | 53 % | 62 % | 68 % | 73 % | 76 % |
| resgate_adaln_rows | HBM | 35 % | 52 % | 57 % | 61 % | 62 % | 70 % |
| resgate_out_rows | HBM | 36 % | 58 % | 62 % | 62 % | 70 % | 75 % |
| swiglu_rows | HBM | 61 % | 85 % | 97 % | 100 %* | 100 %* | 100 %* |
| cuBLAS GEMMs | tensor | 16 % | 24 % | 29 % | 34 % | 35 % | 40 % |
| **step (time-weighted)** | | 19 % | 29 % | 35 % | 37 % | 38 % | 43 % |
| step kernel time, us | | 74.3 | 99.5 | 125.1 | 161.4 | 196.1 | 212.0 |

Training, bf16 (QK-norm off; the QK-norm kernels' own rows from the QK-norm-on runs):

| kernel | bound | L384 | L768 |
|---|---|---|---|
| attn_dkv | HBM / MUFU | 38 % | 26 % |
| attn_dqb | HBM / MUFU | 31 % | 31 % |
| attn_fwd2 | MUFU | 38 % | 47 % |
| adaln_a | HBM | 41 % | 43 % |
| adaln_a_bwd | HBM | 57 % | 56 % |
| bias_transpose | HBM | 29 % | 42 % |
| cond_bwd | HBM | 54 % | 54 % |
| cond_prep | HBM | 50 % | 56 % |
| gate_o | HBM | 60 % | 62 % |
| gate_o_bwd | HBM | 62 % | 66 % |
| pair_ln | HBM | 47 % | 51 % |
| pair_ln_bwd | HBM | 79 % | 85 % |
| qknorm, QK-norm on | HBM | 49 % | 52 % |
| qknorm_bwd, QK-norm on | HBM | 44 % | 50 % |
| qknorm_bwd | HBM | 65 % | 72 % |
| res_adaln_b | HBM | 56 % | 58 % |
| res_adaln_b_bwd | HBM | 51 % | 55 % |
| res_c | HBM | 68 % | 69 % |
| res_c_bwd | HBM | 64 % | 65 % |
| swiglu_bwd | HBM | 78 % | 80 % |
| unfold_lnw | HBM | 16 % | 15 % |
| PyTorch glue | — | 5 % of step | 3 % of step |
| cuBLAS GEMMs | tensor | 58 % | 60 % |
| gemm_swiglu2 | tensor | 49 % | 53 % |
| **step (time-weighted)** | | 51 % | 52 % |
| step kernel time, us | | 1695.1 | 3699.0 |

Training, fp32:

| kernel | bound | L384 | L768 |
|---|---|---|---|
| attn_dkv_tf32 | HBM / tensor | 26 % | 21 % |
| attn_dqb_tf32 | HBM / tensor | 21 % | 14 % |
| attn_fwd_tf32 | HBM / MUFU | 30 % | 25 % |
| adaln_a | HBM | 54 % | 58 % |
| adaln_a_bwd | HBM | 69 % | 69 % |
| cond_bwd | HBM | 88 % | 87 % |
| cond_prep | HBM | 76 % | 81 % |
| gate_o | HBM | 77 % | 80 % |
| gate_o_bwd | HBM | 100 % | 99 % |
| pair_ln | HBM | 76 % | 84 % |
| pair_ln_bwd | HBM | 93 % | 99 % |
| qknorm, QK-norm on | HBM | 74 % | 77 % |
| qknorm_bwd, QK-norm on | HBM | 58 % | 63 % |
| qknorm_bwd | HBM | 78 % | 83 % |
| res_adaln_b | HBM | 73 % | 76 % |
| res_adaln_b_bwd | HBM | 71 % | 76 % |
| res_c | HBM | 81 % | 85 % |
| res_c_bwd | HBM | 92 % | 92 % |
| swiglu_bwd | HBM | 97 % | 99 % |
| swiglu_rows | HBM | 89 % | 94 % |
| unfold_lnw | HBM | 16 % | 15 % |
| PyTorch glue | — | 2 % of step | 1 % of step |
| cuBLAS GEMMs | tensor | 60 % | 61 % |
| **step (time-weighted)** | | 57 % | 53 % |
| step kernel time, us | | 2814.4 | 6432.2 |

What bounds the step:
- Inference at L128-256 is latency, not bandwidth: every kernel moves a few MB, and at S = 5 the attention core has 3 sample
  pairs x (L / 128) x 16 heads = 48-96 work items for 148 SMs. The cores reach 24-30 % at L768, where bf16 is MUFU-bound (the
  exponentials) and fp32 HBM-bound. `layernorm_rows` (the conditioning, L rows) is ~2 us of launch latency at any L.
- No PyTorch glue is left in the inference step (it was 17-21 % of the bf16 step: see the inference section). The bench's
  CUDA-graph time is within ~5 us of the kernels' sum, so launch gaps (what PDL would hide) are not where the time goes.
- cuBLAS GEMMs run at 58-61 % in training (`gemm_swiglu2` 49-53 %), the power-capped ceiling of cuBLAS on this card
  (1.3-1.45 PF/s sustained of 2.23), and 14-40 % in inference (M = 640-3840 rows: tile underfill).
- Training row kernels: bf16 41-84 %, fp32 58-100 %. The training attention: bf16 26-47 %, fp32 14-30 % (the fp32 backward
  loads q / dO or K twice for the MN-major operands; see T1'-T3'). Training glue left (1-5 % of the step): the partial-sum
  reduction (~20 us) and one multi-tensor copy of the parameter gradients (~10-25 us).

성능 확인 in the tables above (2026-09-30): △ = the fastest measured (every table column: see the comparisons) and a kernel
of the group below 70 % SoL; ✓ would need every kernel of the group at 70 % or more; ✗ = slower than another implementation
(none); — = not measured (training: L384 / L768 only).

## Head layouts (2026-10-01)

| heads x head dim | d_single | inference bf16 | inference fp32 | training bf16 | training fp32 |
|---|---|---|---|---|---|
| 16 x 48 (registry row) | 768 | ✓ | ✓ | ✓ | ✓ |
| 24 x 32 | 768 | ✓ | module path | ✓ | module path |
| 12 x 64 | 768 | ✓ | module path | ✓ | module path |
| 16 x 64 | 1024 | ✓ | module path | ✓ | module path |

d_cond 384, d_pair 128 and transition n = 2 in every layout; the transition and the projections scale with d_single.
`integrations/token_dit.py` and `integrations/token_dit_train.py` list the layouts as `LAYOUTS` ((heads, d_single) pairs);
`serves()` takes a layout from the block's `n_head` and the input width, and the fp32 path stays 16 x 48 (the TF32 kernels'
tiles are built around 48-wide rows: see T1'-T3').

How each piece takes a layout:
- **Attention cores** (`attn_inf`, `attn_fwd2`, `attn_dkv`, `attn_dqb`): one cubin per layout, built with `-DNHEAD=<heads>
  -DDHP=<head dim> -DRSQDV=<1 / sqrt(head dim)>` (`sm100._tdit_defs`; no flags for 16 x 48, so its cubins are the same as
  before). The head width sets the K steps of every MMA (DH / 16), the q / k / v / dO boxes (DH columns, 128-B rows padded
  for the swizzle) and the fp32 output staging: 48 = a 32-column box in the 128-B swizzle + a 16-column box in the 64-B
  swizzle, 32 = the 32-column box alone, 64 = two 32-column halves through the same 128-B staging buffer in turn. TMEM:
  dkv holds dK at 384 and dV at 384 + DH (DH 64 fills the 512 columns); dqb keeps 3 dQ buffers at 320 + DH b (DH 64: exactly
  512) and, at DH 64, 2 K slots instead of 3 (the wider dQ staging). `sm100.forward / backward(..., heads, dh)` and
  `GatedInferenceCore(idx, dtype, heads, dh)` take the layout.
- **Inference rows** (`token_dit_rows.cu`): the row kernels take the width at run time (192 threads at 768, 256 at 1024);
  `qknorm_rows(qk, wq, wk, eq, ek, d)` is templated on (d, head dim).
- **Training rows** (`token_dit_train_rows.cu`): one extension per layout (`train.ext(d, hd)`, `-DTD_D -DTD_HD`): the per-head
  sums of `qknorm`, `gate_o_bwd` and `qknorm_bwd` run over HD / 4 threads, and the saved q / k RMS statistics are
  [M, 2 x heads].

Fixed on the way: `attn_dkv` and `attn_dqb` cleared their TMEM accumulators at start for the 48-wide case only (96 dK / dV
columns, 3 x 48 dQ columns). TMEM is not cleared between kernels, so at DH 64 the last 32 dV columns (and part of the dQ
buffers) started each CTA's first work item from what the previous kernel left there. The attention tests alone passed;
the block's gradients depended on which kernel ran before (to_value.weight 0.13 relative error against 0.011 for the bf16
PyTorch block, and not with a synchronize after every call). The clears now cover 2 DH and 3 DH columns; the other widths
were already covered.

Tests: `tests/numerics/test_tdit_heads_sm100_gpu.py` (every layout's training forward / backward at L128 / 384 and gated
inference core at L128 / 200 / 384 against fp64) and `tests/integrations/test_b200_token_dit_layouts_gpu.py` (the fused
inference step against the PyTorch block at S L = 5 x 384, 4 x 200, 3 x 333, shared / per-sample conditioning, QK-norm,
key mask, CUDA-graph replay; the fused training block against the fp32 block at L128 / 384, QK-norm on / off, every
gradient within 1.3x the bf16 PyTorch block's error).

Measurements (2026-10-01, `benchmarks/runners/bench.py target=dit level=module +tdit_n_head=<heads>
d_single_token=<d>`, one DiTBlock, no key mask, compiled; one GPU through `gpuq`, every row of the table from one run;
script `scratch/apb/tdit_layouts.sh` on the B200 host). Inference: S = 5, one conditioning shared by the samples, CUDA
graph; Anthropic `compile=false`. Output error against the fp32 IEEE PyTorch block: ours 3.3-3.6e-3, Anthropic 3.4e-3,
bf16 PyTorch 4.2e-3, in every layout.

Inference, bf16, ms (× = ours against the faster of Anthropic and PyTorch compiled):

| layout | (Length, Dimension, dtype) | PyTorch compiled | Anthropic | ours | × |
|---|---|---|---|---|---|
| 16 x 48 | (256, 768, bf16) | 0.125 | 0.105 | **0.065** | 1.61 |
| 16 x 48 | (512, 768, bf16) | 0.223 | 0.156 | **0.100** | 1.55 |
| 16 x 48 | (768, 768, bf16) | 0.372 | 0.209 | **0.121** | 1.73 |
| 24 x 32 | (256, 768, bf16) | 0.135 | 0.102 | **0.065** | 1.56 |
| 24 x 32 | (512, 768, bf16) | 0.262 | 0.151 | **0.100** | 1.51 |
| 24 x 32 | (768, 768, bf16) | 0.448 | 0.203 | **0.133** | 1.52 |
| 12 x 64 | (256, 768, bf16) | 0.121 | 0.098 | **0.066** | 1.50 |
| 12 x 64 | (512, 768, bf16) | 0.211 | 0.139 | **0.090** | 1.55 |
| 12 x 64 | (768, 768, bf16) | 0.334 | 0.170 | **0.123** | 1.38 |
| 16 x 64 | (256, 1024, bf16) | 0.135 | 0.108 | **0.076** | 1.43 |
| 16 x 64 | (512, 1024, bf16) | 0.248 | 0.160 | **0.123** | 1.30 |
| 16 x 64 | (768, 1024, bf16) | 0.416 | 0.225 | **0.158** | 1.43 |

Training, bf16, A = 48, CUDA graph off / on, ms (× = graph on; Anthropic has no backward):

| layout | (Length, Dimension, dtype) | PyTorch compiled | ours | × |
|---|---|---|---|---|
| 16 x 48 | (256, 768, bf16) | 1.832 / 1.605 | 1.941 / **1.210** | 1.33 |
| 16 x 48 | (512, 768, bf16) | 4.383 / 4.170 | 2.487 / **2.381** | 1.75 |
| 16 x 48 | (768, 768, bf16) | 8.532 / 8.310 | 3.891 / **3.789** | 2.19 |
| 24 x 32 | (256, 768, bf16) | 2.137 / 1.914 | 1.732 / **1.243** | 1.54 |
| 24 x 32 | (512, 768, bf16) | 5.538 / 5.319 | 2.676 / **2.504** | 2.12 |
| 24 x 32 | (768, 768, bf16) | 11.320 / 11.093 | 4.219 / **4.062** | 2.73 |
| 12 x 64 | (256, 768, bf16) | 1.691 / 1.458 | 1.736 / **1.217** | 1.20 |
| 12 x 64 | (512, 768, bf16) | 3.832 / 3.612 | 2.477 / **2.345** | 1.54 |
| 12 x 64 | (768, 768, bf16) | 7.164 / 6.934 | 3.686 / **3.603** | 1.92 |
| 16 x 64 | (256, 1024, bf16) | 2.180 / 1.957 | 1.774 / **1.624** | 1.20 |
| 16 x 64 | (512, 1024, bf16) | 5.052 / 4.833 | 3.288 / **3.267** | 1.48 |
| 16 x 64 | (768, 1024, bf16) | 9.503 / 9.289 | 5.139 / **4.997** | 1.86 |

The 16 x 48 rows reproduce the tables above (training L768 3.79 against 3.85 ms). At d 768, 24 x 32 costs ours 10 %
(inference) and 7 % (training) more than 16 x 48 at L768 (inference the same, training 3-5 % more at L256-512), while
PyTorch compiled slows down by 20 / 33 % at L768, hence the larger ×. 12 x 64 is ours' fastest d-768 layout in training at
L512-768 (2.35 / 3.60 ms). Where the 24 x 32 difference goes is not profiled per kernel: the SoL
tables above are 16 x 48 only. With CUDA graphs off, ours trails PyTorch compiled at L256 in 16 x 48 (0.94x) and 12 x 64
(0.97x); host time, as in the 16 x 48 tables.

## What was tried and not kept

| attempt | result |
|---|---|
| row kernels with one warp per row | 2x slower than Triton (13 warps per SM at M = 1920: latency-bound); one block per row fixed it |
| pair bias as one fused WMMA kernel (LayerNorm + projection) | 3.4-3.6x slower than LayerNorm + cuBLAS; dropped |
| 1-CTA SwiGLU GEMM (`gemm_swiglu_sm100.cu`, kept for the record) | 540 TFLOPS: a 128 x 256 tile per SM is TMA-intake-bound; the 2-CTA kernel replaced it |
| `gemm_swiglu2` at L384 (M = 1920) | 14.0 us vs 13.7 for cuBLAS + row pass: 96 items on 74 SM pairs leave most pairs idle in round 2 |
| tf32 MMA with an MN-major operand in the plain 128-B swizzle (the bf16 kernels' layout for v, q / dO, K) | returns zeros on B200, A or B, no error (isolated probe `tf32_probe2.cu`); an MN-major tf32 operand must be in the 128-B swizzle with 32-B atoms (UMMA layout type 1, SBO 512), which is exact (`tf32_probe3.cu`, also TMA-loaded); a K-major operand in that layout faults. The inference twin reads v^T from a second GEMM instead (a 32-B-atom v tile would not fit next to the g / o staging) |
| a 2-CTA key tile for the fp32 dkv (halving the q / dO intake per CTA) | not built: the kernel was not intake-bound -- its TMA intake floor is ~69 us at L384 against 240 measured -- but latency-bound on 2 stages; one K / V slot with 3 stages took the backward (dkv + dqb) 472 -> 452 us (L384) and 1706 -> 1554 us (L768). dqb with one K slot (STK=1) was slower (558 / 1968 us) |
| fusing the training row passes into custom GEMM epilogues | measured against what it saves, most of them write tensors the backward reads anyway (z, O, a / b), so an epilogue saves one read; the LayerNorm passes need whole rows. Done where it pays: the SwiGLU into the expand GEMM (bf16, `-DSAVE_AB`) |
| `setmaxnreg` dec 56 / inc 224 in the fp32 backward kernels | the 384-thread kernels compile to 168 registers, so inc gains nothing, and dec 56 spilled the producer / MMA warps (172 B in dkv, 108 B in dqb); removed, no spills |
| training column sums by atomicAdd per block | every block onto the same 768 addresses serialised in L2 (`qknorm_bwd` 736 us, `res_c_bwd` 105 us at L384); partial buffers instead |

## Limits and next

- Head layouts other than 16 x 48: bf16 only (fp32 keeps the module path), and no per-kernel SoL yet.
- A DiTBlock call sees one block, so the pair bias and the cond LayerNorm are recomputed per block; the research stack
  runner (branch `b200/token-dit`) hoisted them for the whole stack (training 1.62 / 3.56 ms per block there, with Triton
  and quack).
- fp32 training's SwiGLU is cuBLAS + `swiglu_rows` (`gemm_swiglu2` is bf16).
- fp32 inference at lengths whose last query tile is mostly padding (450, 700) leads Anthropic by 2-5 % only; a 64-row last
  tile or in-kernel handling of L % 8 (no pad copies) are the levers.
- The fp32 backward attention (dkv + dqb) is 2.1-2.2x the bf16 kernels. dqb is the larger part at L768: with 64-key chunks
  (the fp32 tiles of a 128-key chunk do not fit twice) it reloads a query tile's q / dO once per chunk. A 128-key item that
  streams two 64-key sub-chunks against one q / dO load is the next lever there.
- Training row passes that write what the backward reads (z, O, the LayerNorm outputs) gain little from GEMM epilogues (see
  "What was tried"); what is left of the block is cuBLAS at its power-capped rate (58-61 %) and the attention.
