# Token DiT on H100 (sm90)

Kernel-level status of the token DiT block (AF3 Alg. 23: AdaLN + AugmentedAttention with pair bias + conditioned SwiGLU
transition; d_single 768, d_cond 384, d_pair 128, 16 heads x 48, transition n = 2) on H100; the module-level summary is in
[h100.md](../h100.md). Columns are (Length, Dimension, dtype) from the shape registry (`token_single`, d_hidden 768).

On H100 the token DiT runs **hand-written CUDA and cuBLAS only** -- no Triton -- in bf16 and fp32, QK-norm on or off, on
two paths that `modules/dit` dispatches to (the same two integrations as on B200):

- **Inference** (`integrations/token_dit.py` -> `kernels/conditioned_transition/triton/token_dit_runner.py`): `DiTBlock`
  calls with no autograd and the engine's kernel backend, shared or per-sample conditioning. Weight pack and pair bias
  use the upstream capture-scoped caches: eager entries stay outside captures; each capture records their construction
  once and each replay recomputes them from the current weights / pair. A shared conditioning needs L rows, per-sample S L.
  - row passes: `kernels/conditioned_transition/cuda/token_dit_rows.cu` (the first AdaLN pass reads the step's input and
    writes the fp32 residual; the last residual pass writes the output in its dtype -- no conversion kernels; `cond_rows`:
    the step's conditioning LayerNorm and its GEMM-dtype copy in one warp-per-row pass);
  - QK-norm: `qknorm_rows`, in place on the q | k columns of the q|k|v|g GEMM output (sm_scale log2 e folds into the query
    norm's weight instead of Wq);
  - attention core, **bf16**: `kernels/augmented_attention/cuda/attn_fwd.cu` built GATED (`attn_inf`): q | k | v | g read
    as column views of the GEMM output through TMA, sigmoid(g) o written over q;
  - attention core, **fp32**: `kernels/augmented_attention/cuda/attn_tf32.cu` (`attn_tf32_inf`, TF32 wgmma): q | k | g from
    one GEMM, v^T from a second (TF32 wgmma takes v K-major), the hoisted pair bias written key-permuted (below);
  - expand + SwiGLU, **bf16**: `kernels/conditioned_transition/cuda/gemm_swiglu_sm90.cu`, one sm_90a GEMM (128 x 256 tiles
    whose B rows are 128 of Wa then the same 128 of Wb, wgmma m64n256, SwiGLU in the epilogue): [a | b] never reaches HBM.
    Its TF32 build is opt-in (`MINIWORLD_TOKEN_DIT_GEMM_SWIGLU_FP32=1`; see below);
  - the other GEMMs cuBLAS (fp32: TF32 under the caller's `allow_tf32`).
- **Training** (`integrations/token_dit_train.py`): `DiTBlock` calls under autograd, B == 1, L % 128 == 0, key mask [B, L]
  or none. One autograd Function per block whose forward and backward are each one opaque op (torch.compile keeps them
  as nodes). Row kernels `kernels/conditioned_transition/cuda/token_dit_train_rows.cu` (16 kernels, the B200 set; built
  a second time with fp32 GEMM operands, `-DTDT_OP_F32`, for the fp32 block), cuBLAS GEMMs, and the attention core:
  - **bf16**: `attn_fwd.cu`, `attn_dqb.cu` + `attn_dkv.cu` (or `attn_dq.cu` + `attn_dkv.cu` when A is not a multiple of 3);
    the qk-norm row pass writes q pre-scaled by log2 e / sqrt 48, the bias projection carries log2 e, the key mask is an
    additive row. The forward writes og = sigmoid(g) o itself (MODE 2; its g tile TMA-loaded up front), so the block keeps
    no O (113 MB a block at L768, A48) and runs no gate pass: the backward takes D = rowsum(dO O) = rowsum(d og * og).
    `attn_dqb` sums dbias over its three samples in shared memory, then adds the tile with one TMA bulk reduce;
  - **fp32**: `attn_tf32.cu` forward (og, fp32), `attn_tf32_dqb` (dQ; dbias as one TMA bulk reduce-add per 64 x 64 dS tile) and
    `attn_tf32_dkv` (dK, dV); q, k, dO transposed by a tiled CUDA transpose for the products that contract over tokens.

Switches (all default on): `MINIWORLD_TOKEN_DIT_ROWS_CUDA` (inference rows), `MINIWORLD_AUGATTN_BF16_SM90` (bf16 cores),
`MINIWORLD_AUGATTN_TF32_SM90` (fp32 inference core), `MINIWORLD_TOKEN_DIT_TRAIN` (training block),
`MINIWORLD_TOKEN_DIT_TRAIN_FP32` (the fp32 training block). `MINIWORLD_TOKEN_DIT_GEMM_SWIGLU=0` keeps cuBLAS + the row pass.

Every attention kernel here is bitwise deterministic run to run (6 runs, `torch.equal`, every output without atomics);
the bf16 kernels now issue `fence.proxy.async` before releasing a TMA stage (or the parked q tile) they read with
ldmatrix: a generic-proxy read of a stage the async proxy refills is not ordered by the mbarrier release alone.

### The TF32 core's layouts

TF32 wgmma reads both operands K-major only (no transpose bit), which fixes three things:

- P V needs v K-major, i.e. v^T [768, tokens]; so do the backward's products over tokens (dV = P^T dO, dK = dS^T q,
  dQ = dS k), which take q^T, dO^T, k^T.
- P goes to the A operand from registers. The TF32 A fragment holds k-indices (t, t + 4) of an 8-wide k-step, the
  accumulator columns (2t, 2t + 1). Loading the key rows permuted within each group of 8 -- shared-memory row 2j + i is
  key 4i + j, a 4-D TMA box (head columns, i with a 4-row stride, j with a 1-row stride, 8-row groups) -- makes S column
  2t + i key t + 4i, so the accumulator IS the A fragment, no shuffle. The bias and key mask are read in that key order:
  the inference hoist writes the pair LayerNorm's rows permuted (free), training gathers the bias once per call.
- The head dim 48 is two 128-B-swizzled chunks, 32 + 16 columns.

## Inference

bf16 and fp32, QK-norm on / off, shared conditioning, S = 5 (the sampler's diffusion batch).

### I1 · attention core (bf16 `attn_inf` / fp32 `attn_tf32_inf`)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| (Length, Dimension, dtype) | (128, 768, fp32) | (256, 768, fp32) | (384, 768, fp32) | (512, 768, fp32) | (640, 768, fp32) | (768, 768, fp32) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I2 · row kernels (`adaln_rows`, `resgate_adaln_rows`, `swiglu_rows`, `qknorm_rows`, pair LayerNorm) + cuBLAS

| (Length, Dimension, dtype) | (128, 768, bf16/fp32) | (256, 768, bf16/fp32) | (384, 768, bf16/fp32) | (512, 768, bf16/fp32) | (640, 768, bf16/fp32) | (768, 768, bf16/fp32) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Training

### T1 · attention core, bf16 (`attn_fwd`, `attn_dqb` + `attn_dkv`) and fp32 (`attn_tf32_fwd`, `attn_tf32_dqb` + `attn_tf32_dkv`)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| (Length, Dimension, dtype) | (128, 768, fp32) | (256, 768, fp32) | (384, 768, fp32) | (512, 768, fp32) | (640, 768, fp32) | (768, 768, fp32) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T2 · row kernels (`token_dit_train_rows.cu`, 16 kernels + transpose; bf16 and fp32 builds)

| (Length, Dimension, dtype) | (128, 768, bf16/fp32) | (256, 768, bf16/fp32) | (384, 768, bf16/fp32) | (512, 768, bf16/fp32) | (640, 768, bf16/fp32) | (768, 768, bf16/fp32) |
|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

"미검증" = the kernels accept the shape (L % 128 == 0) but no test or measurement has run there. cache build ✓: nothing on
these paths autotunes (fixed launch shapes; the CUDA extensions build on first use into `MINIWORLD_ENGINE_JIT_ROOT`).

Tests: `tests/integrations/test_token_dit_train_gpu.py` (bf16 and fp32 blocks against the fp32 PyTorch block, qk-norm and
mask on / off, A = 4 and 6, torch.compile), `tests/integrations/test_h100_module_wiring_gpu.py` (inference: live inputs /
weights / mask, CUDA graphs, the bf16 and fp32 cores against the Triton cores they replace, qk-norm on / off),
`tests/numerics/test_token_dit_rows_cuda_gpu.py`, `tests/numerics/test_augmented_attention_bf16_sm90_gpu.py`.

## Measurements (2026-10-01)

H100 80GB HBM3 (cssb node02), the v2.2.0 pixi env (torch 2.13.0+cu129), `benchmarks/runners/bench.py target=dit
level=module` (one DiTBlock, key mask 12.5 % masked), ms per block, median. "ours" is the bench's `miniworld` row; Anthropic
is its release at f4f62fa (`MINIWORLD_ANTHROPIC_ROOT` = an external checkout; inference only -- no backward). × = PyTorch / ours
unless the column says otherwise. "before" = the engine on `main`
(2719096e): training through the general module path (Triton), inference through the fused runner with Triton rows and
the Triton gated core.

### Training (A = 48, compiled; CUDA graph off / on)

| (Length, Dimension, dtype) | PyTorch compiled | before | ours | × |
|---|---|---|---|---|
| (384, 768, bf16) | 4.819 / 4.688 | 4.021 / — | **3.341 / 3.205** | 1.44 / 1.46 |
| (768, 768, bf16) | 13.264 / 13.269 | 9.920 / — | **7.344 / 7.278** | 1.81 / 1.82 |
| (384, 768, fp32) | 9.152 / 9.089 | 8.351 / — | **7.264 / 7.126** | 1.26 / 1.28 |
| (768, 768, fp32) | 24.277 / 24.140 | 19.655 / — | **15.992 / 15.798** | 1.52 / 1.53 |

(fp32: TF32 GEMMs, `allow_tf32` as the bench sets it. "before" with a CUDA graph was not measured. The first version of this
path, 2026-09-30, measured 7.459 / 7.229 ms bf16 and 15.869 / 15.769 ms fp32 at L768: within the run-to-run spread.)

Accuracy against the fp32 PyTorch block (test above, L256, A 4 / 6): bf16 within 1.5x of the engine module path's own
error in the same regime; fp32 output and every gradient 5e-4 (1.2e-3 with qk-norm), below the fp32 module path's 8e-4
(1.4e-3).

### Inference, one conditioning shared by the S = 5 samples (`+shared_cond=true`; compiled, CUDA graph)

| (Length, Dimension, dtype) | PyTorch compiled | before | ours | × |
|---|---|---|---|---|
| (384, 768, bf16) | 0.244 | 0.121 | **0.101** | 2.41 |
| (768, 768, bf16) | 0.620 | 0.221 | **0.178** | 3.49 |
| (384, 768, fp32) | 0.473 | — | **0.196** | 2.41 |
| (768, 768, fp32) | 1.247 | — | **0.370** | 3.37 |

(bf16 was 0.111 / 0.200 before the SwiGLU GEMM and `cond_rows`, 2026-09-30.)

Against Anthropic, which runs `compile=false` only (no compilable graph); PyTorch eager here, ours unchanged by compile
(one opaque op). Both fused rows compute a layer's pair bias once per pair and the conditioning once per token.

| (Length, Dimension, dtype) | PyTorch eager | Anthropic | ours | × vs Anthropic |
|---|---|---|---|---|
| (384, 768, bf16) | 0.709 | 0.140 | **0.102** | 1.38 |
| (768, 768, bf16) | 2.285 | 0.251 | **0.177** | 1.41 |
| (384, 768, fp32) | 0.846 | 0.226 | **0.196** | 1.15 |
| (768, 768, fp32) | 2.535 | 0.418 | **0.372** | 1.12 |

### Inference, a different conditioning per sample (the bench default; `compile=false`, CUDA graph)

Historical measurements before the upstream reconciliation: these rows used the module composition (Triton kernels).
The merged fused step now serves per-sample conditioning on H100 and B200; its latency has not been remeasured here.

| (Length, Dimension, dtype) | PyTorch eager | Anthropic | ours | × vs Anthropic |
|---|---|---|---|---|
| (384, 768, bf16) | 0.685 | **0.204** | 0.231 | 0.88 |
| (768, 768, bf16) | 2.256 | **0.429** | 0.519 | 0.83 |
| (384, 768, fp32) | 0.821 | **0.327** | 0.477 | 0.68 |
| (768, 768, fp32) | 2.501 | **0.750** | 1.197 | 0.63 |

### Where the time goes (L768, torch.profiler, per block)

| | inference bf16 (177 us) | training bf16 (7.3 ms) | training fp32 (15.8 ms) |
|---|---|---|---|
| GEMMs | ~101 us (cuBLAS 72 + the SwiGLU GEMM 29) | 2.9 ms (~630 TFLOP/s) | 7.5 ms (~240 TFLOP/s, TF32) |
| attention core | 42 us | 2.0 ms (fwd 0.47 with og, dqb 1.0, dkv 0.57) | 4.3 ms (fwd 0.75, dqb 1.47, dkv 1.58, transposes / gathers 0.5) |
| row kernels | ~36 us | 2.2 ms (each at the HBM roof, ~2.8 TB/s) | 3.5 ms |

Attention cores alone (do_bench, L2 evicted): fp32 inference core 89.0 us at L768 / 36.1 us at L384 against 114.2 / 41.4
for the Triton TF32 gated core it replaces; fp32 training forward 702 us at L768, A48.

## What was tried and not kept

| attempt | result |
|---|---|
| fp32 core on mma.sync m16n8k8 TF32 (FA2 layout, every fragment loaded by the threads) | numerics fine (5.6e-4) but 170 us at L768 against Triton's 114: K / V re-read per 64-row CTA in fp32 is ~465 MB of L2 traffic a call; the wgmma + TMA version reads each tile once per 128 rows and wins (89 us) |
| rounding k / v to TF32 at load (cvt.rna) in that kernel | not the bottleneck (166 vs 170 us) |
| one-pass pair bias (LayerNorm + the 16 projections, a warp per pair row, a 5-step reduce-scatter; and its backward with dWf in registers), training | 620 / 675 us at L768 against 113 + 63 (LN + cuBLAS) / ~300: a latency-bound shuffle chain per row (B200 found the same); LN + cuBLAS kept |
| bf16 dbias: every warpgroup TMA-reduce-adds its own sample's dS tile (no exchange) | 1185 us at L768 against 1019: three times the reduce traffic into L2; the exchange + one reduce per three samples (kept) is 998 |
| og epilogue with the g rows read from global (+ an L2 prefetch) | 496 us at L768 against 472 for O + the gate pass; the g tile TMA-loaded up front made it 474 (151 vs 169 at L384) |
| SwiGLU GEMM also storing [a \| b] for the training backward (through the same staging tile) | 731 us at M = 36864 against 380 for cuBLAS + the row pass: the epilogue serialises; the training block keeps cuBLAS + `swiglu_rows` |
| SwiGLU GEMM in TF32 for the fp32 block | relative error 1.6e-3 against 4.2e-4 (its TMA-fed operands are truncated where cuBLAS rounds) and faster only at M = 3840 (70.9 vs 85.5 us; 47.6 vs 45.9 at 1920, 759 vs 725 at 36864): opt-in |

## 2026-10-05 FP32 backward layout experiment

The explicit candidate in `experiments/token_dit_layout_20261005` fuses the bias's two gathers and transpose into one
CUDA read/write pass, restores dbias with an arithmetic eight-key permutation, and permutes LSE / D in one kernel.
The model and attention math are unchanged. Production dispatch is unchanged.

Paired whole `DiTBlock` forward + backward on H100, FP32, QK-norm on, key mask on, CUDA Graph, A samples; baseline is the
reviewed upstream + local integration at `135b7966`. Each row has 60 alternating paired samples, five replays per sample.

| L | D | A | baseline ms | candidate ms | speedup |
|---|---|---|---|---|---|
| 384 | 768 | 4 | 1.456 | 1.383 | 1.053x |
| 384 | 768 | 6 | 1.688 | 1.608 | 1.050x |
| 768 | 768 | 4 | 2.970 | 2.776 | 1.070x |
| 768 | 768 | 6 | 3.533 | 3.338 | 1.059x |

Validation: 20 existing FP32 numerical / compile / live-weight graph checks passed; standalone layout maps are bit-exact
at L128/256/384/768. Whole-block replay with updated mask, inputs and weights passed, and profiler traces show all three
new CUDA kernels. New-kernel memcheck, racecheck and synccheck each reported zero errors; whole-block racecheck and
synccheck also passed. Whole-block memcheck reports one `cuKernelGetFunction` invalid-handle API error in the unchanged
SwiGLU row extension on both baseline and candidate, with no invalid-memory access reported. This API error remains
unresolved; do not describe the full block as passing default memcheck. Evidence is in `paired.json`, `manifest.json`,
`whole-block-dispatch.json` and the job / sanitizer logs in the experiment directory (Slurm jobs 21411–21413,
`normal_h100`). No new BF16 or B200 speedup is claimed.

## Limits and next

- Training bf16: GEMMs run at ~630 TFLOP/s and every row kernel at the HBM roof; what is left is `attn_dqb` (1.0 ms at
  L768: dbias is ~400 us of it, bound by the reduce traffic into L2 -- A / 3 x 16 x L^2 fp32 -- with shared memory too
  full to sum more than three samples on chip) and fusions across GEMM boundaries, which need GEMM epilogues that keep
  cuBLAS's mainloop speed and do not serialise (the [a | b]-storing SwiGLU GEMM did).
- fp32: the TF32 cores truncate their TMA-fed operands to TF32 (~1.4e-3 at the core against ~4e-4 for a rounding TF32
  GEMM); the inference v^T GEMM costs ~16 us a block at L768; the training bias transposes / gathers (~0.5 ms) are torch
  ops.
- Inference bf16: 177 us a block at L768, of which ~16 us is the conditioning (LayerNorm + two small GEMMs) a DiTBlock call
  cannot share with the other blocks of a step; the research stack runner, which hoisted it, measured 155 us.
- Per-sample conditioning at inference is now served by the merged fused step. The older module-path timings above
  do not describe its current performance; a paired benchmark remains to be run.
- B200 shares the row kernels, the integrations and the qk-norm / I/O conventions changed here; its token DiT tests were
  not re-run on B200.
