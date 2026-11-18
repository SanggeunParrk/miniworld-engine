# FP8 training (backward) kernels, H100 -- experimental, opt-in (target 3.1.0)

Training FP8 routes are separate from the inference `safe` preset (`docs/fp8.md`) and default OFF. With every switch off the training
outputs and all gradients are the 2.5.14 default's bit for bit. FP8 is outside the judge (no registry rows, tolerances or
`backend_versions` entries).

## OuterProductMean (`kernels/integrations/opm_fp8_train.py`, branch `fp8t/opm`, 2026-11-18)

### Switch

| env | effect |
|---|---|
| `MINIWORLD_OPM_FP8_TRAIN=e4m3` | FP8 forward + backward products, dO quantised to e4m3 (**recommended form**: more accurate than e5m2 here, see the error table) |
| `MINIWORLD_OPM_FP8_TRAIN=e5m2` | same, dO in e5m2 (e4m3 x e5m2 products) |
| `MINIWORLD_FP8_TRAIN=1` | turns the route on as `e4m3` unless `MINIWORLD_OPM_FP8_TRAIN` is set |
| `MINIWORLD_OPM_FP8_TRAIN_MINS=<s>` | padded-depth gate (default 1024; 0 = none) |

Module globals `opm_fp8_train.MODE` / `MIN_S` may be set at run time. Eager autograd and CUDA-graph capture work (all scales are device
tensors, no host sync); `torch.compile` tracing is not served (the route declines while compiling and the default runs). H100 sm_90a,
bf16 activations, batch 1, the `opm_train` widths (64/128, 128/256, 128/384, 256/512), both normalisation orders, `count_offset`, the
fused pair residual and RoseTTAFold3's projection biases.

### What runs

Default 2.5.14 training step: prologue (LN + both projections + mask -> A2 / BT bf16 [(i,c), s]) -> O = A2 BT^T (cuBLAS, K = S) ->
epilogue (/ n, W_o, bias); backward: dgrad (dO = (dz / n) W_o into the grouped layout) -> dA = BT^T dO^T, dB = A2^T dO (cuBLAS, K = 32 N
each, same FLOPs as the forward product) -> dW_o off the kept O -> prologue backward (LayerNorm, projections' wgrad, dm).

FP8 route (same stages; FP8 only inside the three big GEMMs, everything else bf16 / fp32 as before):

- forward: the FP8 build of the wide prologue writes A2 / BT as e4m3 (`-DMINIWORLD_OPM_FP8`, the inference route's kernel) with the
  weight-bound power-of-two scales, recomputed every call on the device (`opm_f8_wscale`: one 64-thread kernel, bitwise
  `opm_fp8.bound_scales`; the weights move every step) -> cuBLASLt e4m3 product (fp32 accumulation, bf16 O kept for dW_o) -> default
  epilogue. The saved A2 / BT are e4m3 (half the bytes).
- backward: `opm_dgrad_f8` (the default dgrad kernel, `#if`-guarded FP8 path) writes dO in BOTH layouts [(i,c),(j,e)] and
  [(j,e),(i,c)] as e4m3 / e5m2 from one launch (gridDim.z = 2: the second plane reads dz transposed and W_o with (c, e) swapped; the
  count table is symmetric), TMA stores, one amax atomic per warp; dz / n and the dbo partials as the default. `opm_t8`: byte
  transposes A2 / BT -> [s, (i,c)]. dA, dB on cuBLASLt (TN FP8 needs K-major operands on both sides: hence the second dO plane and
  the transposes), e4m3 x e4m3 (or e4m3 x e5m2), fp32 accumulation, bf16 out, then the default dW_o / prologue backward.
- scales: per tensor, powers of two, from bounds -- nothing can saturate, no calibration, no amax history, no guard/redo:
  A2 / BT from the weights (inference `bound`); dO from Cauchy-Schwarz, |dO| <= max_ij(|dz_ij|_2 / n_ij) max_ce |W_o[:, ce]|_2,
  x (1 + 2^-6), over 448 (e4m3) / 57344 (e5m2) (`opm_f8_dscale`: one pass over dz + one block). Measured peaks: A2 / BT 11-22 of 448,
  dO 75-104 of 448 (e4m3) -- the bound is 4-6x loose, i.e. 2-2.5 binades of e4m3 range unused at the top, nothing at the bottom lost
  for these inputs. The mean normalisation stays exact: the fp32 integer count table of the default, divided in fp32 (dO) and in the
  default epilogue.
- determinism: no float atomics in any gradient path (the amax atomics are integer max on the diagnostics/scale buffer); two steps
  give identical bits (checked in every error row and in the test).

### Backward / step stages (measured, H100, torch.profiler kernel times, eager, my tree; ms per step)

64/128, AF3 order (registry training rows: L 128-768 at depth 1024; Boltz order also 384-2048), ragged MSA (last S/8 rows cover half
the tokens). "FP8" = `e4m3`.

| stage | GEMM / work | L384 S1024 default | FP8 | L768 S1024 default | FP8 | L768 S4096 default | FP8 |
|---|---|---|---|---|---|---|---|
| fwd prologue | LN + [S N, 64] x [64, 64] | 0.049 | 0.043 | 0.098 | 0.083 | 0.408 | 0.377 |
| fwd product O = A2 BT^T | M = N = 32 L, K = S | 0.405 | 0.261 | 1.608 | 1.056 | 6.514 | 3.788 |
| fwd epilogue | [L^2, 1024] x [1024, 128] | 0.130 | 0.129 | 0.461 | 0.458 | 0.465 | 0.461 |
| bwd dgrad (dO) | [L^2, 128] x [128, 1024] | 0.143 | 0.315 (2 planes) | 0.576 | 1.277 | 0.614 | 1.470 |
| bwd dO scale bound | pass over dz | - | 0.095 | - | 0.158 | - | 0.183 |
| bwd A2 / BT transposes | bytes | - | 0.026 | - | 0.048 | - | 0.191 |
| bwd dA + dB | 2 x (M = S, N = K = 32 L) | 0.751 | 0.426 | 3.025 | 1.765 | 12.90 | 7.076 |
| bwd dW_o | off O | 0.142 | 0.138 | 0.495 | 0.488 | 0.500 | 0.496 |
| bwd prologue backward | LN bwd + wgrad | 0.060 | 0.060 | 0.123 | 0.123 | 0.505 | 0.516 |
| bwd reduce + casts + misc | | ~0.13 | ~0.14 | ~0.20 | ~0.21 | ~0.47 | ~0.48 |
| GEMM share of the default step | | 63 % | | 69 % | | 82 % | |
| upper bound (GEMMs at 2x, rest unchanged) | | 0.69x | | 0.66x | | 0.59x | |

The three GEMMs ran at 1.55-1.8x (forward product 1.55x at K = 1024 -- its bf16 O write is a large part -- backward pair 1.76-1.82x).
What eats the rest: the FP8 dgrad (2.2x the default's time: it computes and writes two planes; ~0.17-0.86 ms extra), the dO-scale
pass (0.1-0.18 ms) and the byte transposes (0.03-0.19 ms). Large depth: the default at L384 S16384 trains (22.6 ms eager; GEMMs 92 %).

### Step time vs the 2.5.14 default (CUDA graph of fwd + bwd, same process, interleaved, same-cubin control)

`scripts/speed.py` (scratch `~/miniworld-engine-scratch/fp8t-opm/`, raw `res/speed2.jsonl`): graphs def / ctl (default again) / e4m3 /
e5m2; cool = 1.5 s idle then a ~40 ms burst (median of 3), sustained = after 1 s back-to-back, 3 blocks of ~60 ms (mean). Cells
`cool / sustained`.

| width, order | L | S | default ms | ctl / def | e4m3 / def | e5m2 / def |
|---|---|---|---|---|---|---|
| 64/128 AF3 | 128 | 1024 | 0.330 / 0.331 | 1.012 / 1.010 | 1.153 / 1.151 | 1.151 / 1.150 |
| 64/128 AF3 | 256 | 1024 | 0.907 / 0.937 | 0.999 / 1.025 | 0.941 / 0.935 | 0.943 / 0.961 |
| 64/128 AF3 | 384 | 1024 | 1.778 / 1.941 | 1.003 / 1.025 | 0.891 / 0.871 | 0.893 / 0.898 |
| 64/128 AF3 | 512 | 1024 | 3.072 / 3.396 | 1.004 / 1.029 | 0.858 / 0.841 | 0.853 / 0.861 |
| 64/128 AF3 | 768 | 1024 | 6.526 / 7.335 | 0.999 / 1.006 | 0.843 / 0.809 | 0.844 / 0.819 |
| 64/128 Boltz | 384 | 1024 | 1.774 / 1.915 | 1.000 / 1.036 | 0.904 / 0.894 | 0.916 / 0.922 |
| 64/128 Boltz | 768 | 2048 | 11.33 / 13.17 | 0.999 / 0.981 | 0.721 / 0.699 | 0.717 / 0.685 |
| 64/128 AF3 | 384 | 4096 | 5.576 / 6.577 | 1.001 / 0.984 | 0.690 / 0.648 | 0.685 / 0.643 |
| 64/128 AF3 | 768 | 4096 | 21.18 / 24.66 | 1.001 / 0.989 | 0.647 / 0.623 | 0.643 / 0.615 |
| 64/128 AF3 | 384 | 16384 | 20.71 / 25.34 | 0.999 / 0.976 | 0.627 / 0.579 | 0.621 / 0.569 |
| 128/256 AF3 | 384 | 1024 | 2.115 / 2.283 | 0.997 / 1.038 | 0.951 / 0.936 | 0.954 / 0.964 |
| 128/256 AF3 | 768 | 1024 | 7.497 / 8.552 | 0.995 / 1.005 | 0.888 / 0.860 | 0.889 / 0.872 |
| 128/384 AF3 | 384 | 1024 | 2.386 / 2.525 | 1.000 / 1.055 | 1.087 / 1.087 | 1.087 / 1.130 |
| 256/512 AF3 | 384 | 1024 | 2.962 / 3.047 | 1.000 / 1.023 | 1.142 / 1.134 | 1.139 / 1.172 |

Reading: at the registry training depth (1024) the module step is 0.84-0.95x from L 256 at 64/128 and 128/256, but SLOWER at L 128
(1.15x) and at the wide d_pair 384 / 512 (1.09-1.14x: there the dgrad is a K = 384 / 512 contraction and doubling it costs more than
the products save). Deep MSA (S >= 2048): 0.58-0.72x. The default gate (`MIN_S` 1024) does not exclude the losing cells; a caller
should gate by shape (see the recommendation).

### Gradient error against float64 PyTorch (rms / max relative, %)

`scripts/err.py` (raw `res/err1.jsonl`, `res/err2.jsonl`): the module in float64 on the PyTorch backend, same inputs, dz random;
rms = rms(g - g64) / rms(g64), max = max|g - g64| / max|g64|. All finite, no saturation, deterministic.

| case | tensor | bf16 default | FP8 e4m3 | FP8 e5m2 |
|---|---|---|---|---|
| 64/128 AF3 L384 S1024 | y | 0.42 / 0.60 | 2.64 / 1.46 | 2.64 / 1.46 |
| | d msa | 0.52 / 0.54 | 3.77 / 3.44 | 5.91 / 5.37 |
| | d ln gamma | 0.40 / 0.31 | 3.33 / 2.72 | 5.94 / 6.42 |
| | d ln beta | 0.33 / 0.29 | 3.19 / 2.82 | 5.68 / 5.55 |
| | d W_left | 0.42 / 0.39 | 3.40 / 2.62 | 5.85 / 4.75 |
| | d W_right | 0.42 / 0.29 | 3.72 / 3.06 | 6.60 / 6.18 |
| | d W_out | 0.40 / 0.42 | 2.79 / 1.52 | 2.79 / 1.52 |
| | d b_out | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| 64/128 Boltz L384 S1024 | y | 0.25 / 0.39 | 0.61 / 0.79 | 0.61 / 0.79 |
| | d msa / d W_left (others as AF3 row) | 0.52 / 0.42 | 3.77 / 3.40 | 5.91 / 5.85 |
| 64/128 AF3 L128 S1024 | d msa / d W_left | 0.52 / 0.41 | 3.79 / 3.66 | 5.93 / 6.41 |
| 128/256 AF3 L256 S1024 | d msa / d ln gamma (max) | 0.52 / 0.53 | 3.79 / 5.66 | 5.93 / 9.58 |

More cases (`res/err2.jsonl`; d msa / d ln gamma / d W_left / d W_out, rms / max %):

| case | y default / e4m3 | d msa default / e4m3 / e5m2 | d ln gamma e4m3 / e5m2 | d W_left e4m3 / e5m2 | d W_out e4m3 | dO peak e4m3 |
|---|---|---|---|---|---|---|
| 64/128 AF3 L768 S1024 | 0.46 / 2.39 | 0.52 / 3.77 / 5.90 | 4.04 / 6.13 (max 4.81 / 8.34) | 3.14 / 6.10 | 2.28 | 80 |
| 64/128 Boltz L384 S2048 | 0.24 / 0.49 | 0.52 / 3.77 / 5.91 | 3.44 / 5.58 | 3.13 / 6.27 | 2.37 | 84 |
| 64/128 AF3 L768 S4096 | 0.44 / 1.82 | 0.52 / 3.78 / 5.92 | 3.59 / 6.08 | 2.52 / 4.35 | 1.57 | 84 |
| 128/384 AF3 L384 S1024 | 0.45 / 2.57 | 0.52 / 3.78 / 5.92 | 3.91 / 5.36 | 2.87 / 5.35 | 2.53 | 63 |
| 256/512 AF3 L384 S1024 | 0.46 / 2.89 | 0.52 / 3.78 / 5.91 | 3.60 / 6.12 | 3.57 / 5.60 | 2.92 | 74 |


Reading: one e4m3 rounding of A2 / BT / dO has ~2.7-3.5 % rms per element and the K-sums of dA / dB do not average it down (random
signs), so the input / LN / projection gradients carry 3-4 % rms (e4m3 dO) or 5.5-6.6 % (e5m2 dO, 2-bit mantissa) against the
default's 0.3-0.5 %: 7-12x the bf16 error, far outside the engine tolerance (expected for FP8; the user decides). d W_out (off the
FP8-product O) 2.3-2.8 %, d b_out exact. e4m3 for dO is the better choice here: the bound-based scale leaves e4m3 enough range
(peak 75-104 of 448), so e5m2's range buys nothing and costs a mantissa bit. Risk not covered by these inputs: per-tensor dO scaling
with very uneven pair counts n_ij (dO ~ 1 / n_ij) pushes the small-1/n pairs toward e4m3 subnormals; e5m2 is the fallback there.

### Training dynamics

`scripts/train.py` (raw `res/train.jsonl`): student OPM (64/128 AF3, master weights, Adam lr 1e-3) distils a teacher OPM (other random
weights) on a fresh random MSA every step (L 256, S 1024, ragged mask), relative MSE loss, 300 steps, same seeds / data per route:

| route | loss step 0 | mean steps 140-159 | mean last 50 | last-50 gap vs default | max gap of the 20-step moving average |
|---|---|---|---|---|---|
| bf16 default | 1.8061 | 0.3297 | 0.1719 | - | - |
| FP8 e4m3 | 1.8044 | 0.3298 | 0.1722 | +0.18 % | 0.21 % |
| FP8 e5m2 | 1.8044 | 0.3298 | 0.1723 | +0.19 % | 0.22 % |

No NaN / inf; the curves overlap within 0.22 %.

### Bit-identity of the default and tests

- The FP8 code is all `#if defined(MINIWORLD_OPM_FP8)` in `csrc/opm_epilogue.cu` (the default builds preprocess to the same text) and
  the Python hook in `opm_train.outer_product_mean` is a `MODE is not None` check. `scripts/bit.py` (sha1 of y and every gradient of the training step, FP8 switches off), this tree vs the base 1a2bbe42e (= 2.5.14 + inference FP8, FP8 off) in separate processes: 9 / 9 cases identical (64/128 AF3 / Boltz at (256, 1024), (384, 2048), padded (200, 1000) and (192, 512), (128, 4096); 128/256, 128/384, 256/512). Inference outputs were not hashed in this session (the FP8 code is preprocessor-excluded from the default builds).
- `tests/gpu/test_opm_fp8_train.py` (new, 10 cases): off by default + no leaked state, dO^T plane = dO plane transposed bit for bit
  and dz/n, dbo equal to the default dgrad (e4m3, e5m2), dequantised dO within one FP8 rounding, weight scales = `bound_scales`,
  gradients vs fp64 for both formats / both orders / 3 widths with determinism: 10 / 10 pass. Existing files importing the touched
  modules: `tests/gpu/test_opm_fp8.py` (inference route; `opm_lt_f8` now also takes e5m2 operands), `tests/gpu/test_opm_deep_product_route.py`,
  `tests/cpu/test_capture_api.py` (`opm_fp8_train._PEAK` listed in NOT_WEIGHT_DERIVED): all pass (53 + 13). ruff check / format, pyright: clean on the touched Python files.

### What did not pay / open

- The FP8 dgrad writing two planes: 2.2x the bf16 dgrad (0.17-0.86 ms extra per step). A one-plane kernel with the transposed copy
  staged in shared memory needs 128 KB of staging per CTA (d_pair 128 only) -- not attempted. The first version (per-chunk atomics,
  plain stores) was 2.6x; TMA stores + one atomic per warp brought it to 2.2x.
- The dO-scale bound pass (0.1-0.18 ms) could move into the producer of dz (not ours) or become a delayed scale (amax of the previous
  step, guard + redo); kept the bound for zero saturation risk and graph safety.
- d_pair 384 / 512 and L 128: slower than the default; not fixed.
- Not done: O kept as e4m3 (dW_o / epilogue read O bf16: 0.13-0.5 ms each, memory-bound), prologue writing the [s, (i,c)] layout
  directly (would remove the transposes, 0.03-0.19 ms).

### Recommendation

Offer `MINIWORLD_OPM_FP8_TRAIN=e4m3` only as an experiment for deep-MSA training (S >= 2048: 0.58-0.72x of the module step) and for
L 256-768 at depth 1024 with d_pair 128 / 256 (0.81-0.95x); do not use it at d_pair 384 / 512 or L 128. Gradient error is 7-12x the
bf16 default's (3-4 % rms); the 300-step distillation curve matches within 0.2 %, but an end-level training run is needed before any
default.

### Progress log

- 00:25 worktree `fp8t/opm` from 1a2bbe42e; stage profile of the default (main checkout by mistake, then re-measured on this tree).
- 01:00 first route (FP8 dgrad planes, transposes, mixed cuBLASLt): correct, deterministic; 0.92x at L384 S1024 -- dgrad 2.6x slower.
- 01:35 TMA stores + one atomic per warp, device scale kernels: 0.89x / 0.84x / 0.65x (L384 / L768 S1024 / L768 S4096).
- 02:00-05:10 session stopped (API limit). 08:00 tests, bit-identity, error, training curve, speed sweep, this doc.

## OuterProductMean rounds 2 + 3 (branch `fp8t3/opm`, base 3.0.1 = origin/main f5be1df46, 2026-11-19)

Everything below is opt-in (`MINIWORLD_OPM_FP8_TRAIN=e4m3`), default OFF; with the switch off the training step is bit-identical to 3.0.1
(`scratch fp8t3/scripts_opm/bit.py`, 9 / 9 cases: this tree vs a 3.0.1 worktree, separate processes, identical sha1 of y and every gradient).

### What changed against round 1

- delayed dO scale (amax history of the previous steps, first call seeds from the Cauchy-Schwarz bound): no pass over dz (-0.1..0.18 ms).
- the FP8 prologue writes A2 / BT in both layouts: no byte-transpose pass.
- d_pair 128: single-pass dual-plane dgrad (`opm_dgrad_kernel<SP=true>`): one GEMM + one dz read, the transposed dO plane is staged in shared
  memory (DG_T_BYTES) and flushed with coalesced stores; weights double buffered, one CTA per SM. d_pair 256 / 384: two plane launches
  (round 1), weights double buffered for KZ 256..511 (`dg_wbuf`; a refactor in the WIP had dropped the 256 case, which put d256 back at 1.02-1.04x;
  restored in round 3).
- shape gate `MIN_L_BY_CZ` (served only where the step is faster, see below): d128 from L128, d256 from L256, d384 from L384, d512 never.
  `MAX_CZ = 0` turns the gate off (benchmarks / tests).
- `tools/bench_opm_fp8_train.py` (ready to run, see its docstring).

### Step time vs the 3.0.1 default (CUDA graph fwd + bwd, same process, interleaved, same-cubin control, S 1024 unless noted)

`tools/bench_opm_fp8_train.py`, raw `scratch fp8t3/res/bench_final.jsonl`. Cells = e4m3 / default, `cool / sustained`; ctl = default again (same cubin),
its sustained ratio shows the run-to-run + thermal order bias (1.00-1.04): read the sustained column against ctl.

| d_msa/d_pair | order | L | default ms (cool) | ctl cool / sus | e4m3 cool / sus | served by gate |
|---|---|---|---|---|---|---|
| 64/128 | AF3 | 128 | 0.332 | 0.996 / 0.996 | 0.907 / 0.900 | yes |
| 64/128 | AF3 | 256 | 0.902 | 0.999 / 1.027 | 0.854 / 0.874 | yes |
| 64/128 | AF3 | 384 | 1.777 | 1.001 / 1.031 | 0.843 / 0.853 | yes |
| 64/128 | AF3 | 512 | 3.052 | 1.000 / 1.045 | 0.826 / 0.833 | yes |
| 64/128 | AF3 | 768 | 6.483 | 1.000 / 1.013 | 0.830 / 0.798 | yes |
| 64/128 | Boltz | 128 | 0.308 | 1.008 / 1.007 | 0.976 / 0.975 | yes |
| 64/128 | Boltz | 256 | 0.880 | 1.000 / 1.026 | 0.872 / 0.888 | yes |
| 64/128 | Boltz | 384 | 1.761 | 1.003 / 1.023 | 0.852 / 0.849 | yes |
| 64/128 | Boltz | 768 | 6.472 | 1.002 / 0.994 | 0.829 / 0.777 | yes |
| 64/128 | Boltz, S 2048 | 768 | 11.22 | 0.999 / 1.013 | 0.711 / 0.688 | yes |
| 64/128 | AF3, S 4096 | 384 / 768 | 5.51 / 20.9 | 1.000 / 0.995, 1.000 / 0.982 | 0.668 / 0.636, 0.643 / 0.602 | yes |
| 64/128 | AF3, S 16384 | 384 | 20.46 | 0.999 / 0.982 | 0.625 / 0.566 | yes |
| 128/256 | AF3 | 128 | 0.382 | 1.003 / 1.004 | 0.990 / 0.991 | no (noise-level) |
| 128/256 | AF3 | 256 | 1.014 | 0.997 / 1.029 | 0.961 / 1.000 | yes (marginal) |
| 128/256 | AF3 | 384 | 2.098 | 0.997 / 1.022 | 0.938 / 0.961 | yes |
| 128/256 | AF3 | 512 | 3.459 | 1.001 / 1.038 | 0.926 / 0.943 | yes |
| 128/256 | AF3 | 768 | 7.405 | 1.001 / 1.033 | 0.935 / 0.959 | yes |
| 128/256 | Boltz | 384 | 2.099 | 0.997 / 1.017 | 0.936 / 0.951 | yes |
| 128/384 | AF3 | 128 | 0.413 | 1.000 / 0.998 | 1.040 / 1.039 | no (loses) |
| 128/384 | AF3 | 256 | 1.144 | 1.000 / 1.025 | 0.985 / 1.017 | no (parity) |
| 128/384 | AF3 | 384 | 2.367 | 0.999 / 1.021 | 0.977 / 0.988 | yes (marginal) |
| 128/384 | AF3 | 512 | 3.964 | 1.002 / 1.018 | 0.963 / 0.974 | yes |
| 128/384 | AF3 | 768 | 8.496 | 0.998 / 1.019 | 0.968 / 0.960 | yes |
| 128/384 | Boltz | 384 | 2.370 | 1.000 / 1.038 | 0.979 / 1.004 | yes (marginal, sustained at parity) |
| 256/512 | AF3 | 384 | 2.947 | 0.998 / 1.018 | 1.025 / 1.060 | no: NO-GO |

d_pair 128 wins 2-22 % everywhere (0.57-0.98x), d256 4-7 %, d384 2-4 % (marginal: at parity in the sustained column of one cell). d_pair 512 is a
no-go: dgrad is a K = 512 contraction written twice (no staging room for a single pass: dz tile 128 KiB), products save less than the second plane
costs (1.025x cool / 1.06x sustained). d_pair 256 / 384 at L128-256 are at or above parity: gated off.
The 3.0.1 default is the baseline (3.0.1 sped the Boltz-order default up; the FP8 route still wins 0.85x at L384 there).

### Gradient error against float64 PyTorch (rms / max relative, %), current tree (`res/err_r3.jsonl`)

e4m3 dO (delayed scale; peaks of dO 149-208 of 448, no saturation, all finite and deterministic, run-to-run identical bits). Default = bf16 route.

| case | y default / e4m3 | d msa | d ln gamma | d W_left | d W_out |
|---|---|---|---|---|---|
| 64/128 AF3 L384 | 0.42 / 2.64 | 0.52 / 3.77 | 0.40 / 3.33 | 0.42 / 3.40 | 0.40 / 2.79 |
| 64/128 Boltz L384 | 0.25 / 0.61 | 0.52 / 3.77 | 0.44 / 3.33 | 0.42 / 3.40 | 0.40 / 2.79 |
| 64/128 AF3 L768 | 0.46 / 2.39 | 0.52 / 3.77 | 0.56 / 4.04 | 0.40 / 3.14 | 0.39 / 2.28 |
| 64/128 AF3 L768 S4096 | 0.44 / 1.82 | 0.52 / 3.78 | 0.46 / 3.59 | 0.29 / 2.52 | 0.34 / 1.57 |
| 128/256 AF3 L256 | 0.44 / 2.36 | 0.52 / 3.79 | 0.48 / 3.95 | 0.45 / 3.71 | 0.39 / 2.35 |
| 64/128 AF3 L128 | 0.43 / 2.32 | 0.52 / 3.79 | 0.43 / 3.12 | 0.41 / 3.66 | 0.37 / 2.26 |

Same picture as round 1: e4m3 rounding of A2 / BT / dO gives 3-4 % rms on the input / projection gradients (7-10x the bf16 default's 0.3-0.5 %).
The delayed scale did not change the error. This is the FP8 price; the user decides.

### Training curve (`res/train_r3.jsonl`, distillation, 300 steps, same seeds / data)

| case | loss step 0 default / e4m3 | mean steps 140-159 default / e4m3 | mean last 50 default / e4m3 |
|---|---|---|---|
| 64/128 AF3 L256 S1024 | 1.8061 / 1.8044 | 0.3676 / 0.3677 | 0.1444 / 0.1447 (+0.22 %) |
| 64/128 Boltz L384 S2048 | 2.0531 / 2.0530 | 0.0181 / 0.0181 | 0.02199 / 0.02199 |

### Tests

`tests/gpu/test_opm_fp8_train.py` 14 / 14 pass on this tree (gate table, delayed scale, both planes, gradients vs fp64, determinism).
ruff / pyright were not re-run in round 3 (no login-node python): run them before any release.

### Open

- d_pair 384 gain is 2-4 %: marginal; d_pair 512 and d256/384 at short L: no-go (numbers above).
- Not done: O kept as e4m3, prologue writing [s,(i,c)] directly; end-level training run (error is 7-10x bf16); single-pass dual-plane for d_pair 256+ (staging does not fit).
- The 3 widths served need prebuilt FP8 variants (`opm_train._ext(fp8=True ...)`: 4 modules, ~100 s each when built); prebuilding into the wheel is not done on this branch.
