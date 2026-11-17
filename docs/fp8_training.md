# FP8 training routes (H100, opt-in, engine 3.1.0 line)

Everything here is EXPERIMENTAL, opt-in and off by default. Base: `rel/3.0.0` 1a2bbe42e (= 2.5.14 + the inference FP8 routes,
[fp8.md](fp8.md)). With no training switch set every route is bit-identical to 2.5.14 (sha1 of the output and every gradient, separate
processes). The training switch is separate from the inference `MINIWORLD_FP8` presets.

## Token DiT (`conditioned_transition.cuda.token_dit_fp8_train`, branch `fp8t/tokendit`)

Scope: the H100 bf16 (`master`) training block `kernels/integrations/token_dit_train.py` (TokenDiTProtenix / MiniWorld default options,
width 768, 16 x 48 heads, cond 384, pair 128). Scripts / raw rows (not in the tree): `~/miniworld-engine-scratch/fp8t-tokendit/{scripts,res,logs}`. Registry training rows that matter: `token_single_shared_cond` A = 48 (and 8) samples,
L 128-768 (M = A L rows up to 36864).

```
MINIWORLD_FP8_TRAIN=1                         # training switch: the token DiT takes RECOMMENDED = "fast" (expand, squeeze)
MINIWORLD_TOKEN_DIT_FP8_TRAIN=<places>        # explicit: expand, squeeze, qkvg, out (forward); dexpand, dsqueeze, dqkvg, dout (dgrad);
                                              # presets fast, transition, attention, all; "" / 0 = off (wins over MINIWORLD_FP8_TRAIN)
MINIWORLD_TOKEN_DIT_FP8_TRAIN_GRAD=e4m3|e5m2  # 8-bit format of the incoming gradients (dgrad A operands); default e4m3
MINIWORLD_TOKEN_DIT_FP8_TRAIN_MIN_ROWS=16384  # row gate (default 16384; every measurement below ran with 0)
```

Recipe: every activation / gradient operand gets a dynamic per-row (per-token) scale, row amax / FMAX (e4m3 448, e5m2 57344), written by
the row pass that makes the operand (`token_dit_fp8_train_rows.cu`: `q8_rows`, `swiglu_q8` = SwiGLU + quantize, h never stored in bf16;
`swiglu_bwd_q8` = swiglu_bwd + 8-bit dab); weights get a per-output-channel scale (dgrad: per input channel of the transposed weight),
made once per weight version and kept in the bf16 pack. Products: `torch._scaled_mm` (cuBLASLt, row-wise scales, fp32 accumulation,
bf16 out). Nothing saturates by construction: no calibration, no overflow guard, no amax history. Master weights / optimizer unchanged,
saved activations stay bf16 (the wgrad reads them). Determinism: no atomics added; the FP8 route is bitwise repeatable run to run (test).

### Where the step goes (A48, eager torch profiler, kernel time; `scripts/prof.py`)

Step = one block forward + backward, every input and parameter differentiated. A48 L384 (M 18432): 3466 us; A48 L768 (M 36864): 7754 us.

| part | L384 us (share) | L768 us (share) |
|---|---|---|
| all GEMMs (cuBLAS bf16, 18 per step) | 1377 (40 %) | 2601 (34 %) |
| attention core fwd + bwd (`attn_fwd`, `attn_dqb`, `attn_dkv`) | 681 (20 %) | 2259 (29 %) |
| row kernels (AdaLN, residual / gate, SwiGLU, QK-norm, pair bias, their backwards, final reduces) + torch elementwise | ~1400 (40 %) | ~2900 (37 %) |

Per GEMM stage (bf16 kernel us from the profile; FP8 ratio = pure `_scaled_mm` against cuBLAS bf16 in the same layout, operands ready,
`scripts/gemmbench.py`; upper bound = bf16 x (1 - ratio), no quantization cost):

| stage | shape (M = A L) | GF @L384 | bf16 us L384 / L768 | e4m3 row-wise ratio L384 / L768 | upper bound us L384 / L768 | placement |
|---|---|---|---|---|---|---|
| fwd cond G + Gg | [M,384]x[384,4608] | 65 | 117 / 226 | 0.92 / 0.93 | 9 / 16 | none (K = 384: no gain; see note) |
| fwd q\|k\|v\|g | [M,768]x[768,3072] | 87 | 130 / 255 | 0.75 / 0.78 | 33 / 56 | `qkvg` (xa row-q8) |
| fwd out | [M,768]x[768,768] | 22 | 34 / 66 | 0.80 / 0.76 | 7 / 16 | `out` (og row-q8) |
| fwd expand | [M,768]x[768,3072] | 87 | 125 / 244 | 0.75 / 0.77 | 31 / 56 | `expand` (xt row-q8) |
| fwd squeeze | [M,1536]x[1536,768] | 43 | 59 / 120 | 0.68 / 0.67 | 19 / 39 | `squeeze` (h from `swiglu_q8`) |
| dgrad dh = dz Wsq | [M,768]x[768,1536] | 43 | 68 / 131 | 0.79 / 0.78 | 14 / 29 | `dsqueeze` (dz row-q8) |
| dgrad dxt = dab Wab | [M,3072]x[3072,768] | 87 | 115 / 230 | 0.62 / 0.60 | 44 / 92 | `dexpand` (dab from `swiglu_bwd_q8`) |
| dgrad dog = dy Wo | [M,768]x[768,768] | 22 | 36 / 70 | 0.79 / 0.75 | 8 / 17 | `dout` (dy row-q8) |
| dgrad dxa = dqkvg Wqkvg | [M,3072]x[3072,768] | 87 | 117 / 229 | 0.62 / 0.59 | 45 / 93 | `dqkvg` (dqkvg row-q8) |
| dgrad dchat + dcg | [M,4608]x[.,384] | 65 | 109 / 206 | 0.66 / 0.61 | 37 / 80 | none (conditioning; see note) |
| wgrad dWsq / dWab / dWo / dWqkvg / dWn / dWg | [N,M]x[M,K] | 304 | 70+118+43+117+70+42 = 460 / 834 | e4m3 row 0.67-1.35; e5m2 x e4m3 tensor-wise 0.53-0.89 | ~190 / ~350 (tensor-wise) | none (below) |
| attention core | QK + PV 22 GF fwd | | 681 / 2259 | - | - | none: latency bound (inference notes, tensor pipe 24-29 %) |

Upper bound of every GEMM in FP8 with free quantization: ~440 us of 3466 (12.7 %) at L384, ~850 of 7754 (11 %) at L768. Not reachable:
- every 8-bit operand needs a row pass: q8 of a 768-wide bf16 tensor costs 15.5 us at M 18432 (a 3072-wide one ~45 us); fusing it into
  the producing row kernels would touch the default route's `token_dit_train_rows.cu` (not done: the default kernels stay byte-identical);
- wgrad contracts over the tokens: cuBLASLt FP8 needs both operands token-major (transposed 8-bit copies of dY and X, 28-113 MB each at
  L384; a torch transpose of a 768-wide tensor is 108 us) with a per-channel scale over all 18432 tokens (a second pass), i.e. the copies
  cost more than the GEMM saves (54 us at best for dWab); per-token scales cannot be used along the contraction; not built.
- measured FP8 kernels in the route (L384): expand 124.7 -> 15.5 (q8 xt) + 92.8; squeeze 58.7 -> 40.0 (h8 from `swiglu_q8` 56.8 us
  replacing `swiglu_rows` 62.3); dsqueeze 67.9 -> 15.5 + 50.1 (net 2 us); dexpand 115.0 -> 68.9, but `swiglu_bwd_q8` 139.4 against
  `swiglu_bwd` 114.0 (the extra 57 MB 8-bit write; the pass is at HBM speed); plus per optimizer step the weight quantization (dgrad
  forms: transpose + q8 per weight). A fused transpose + quantize weight kernel (`wq8t`, one block per 32 columns) was tried and was
  slower in the graph (transition 0.989 -> 1.011 at L384): removed.

Note, not FP8: with the registry's shared conditioning (cond [L, 384] expanded over the 48 samples) the conditioning GEMMs G / Gg and
their backward (dchat, dcg, dWn, dWg) run on M = A L identical rows: 338 us at L384 (~10 % of the step), 48x more work than needed. Doing
them on L rows (and dG summed over the samples first) is a larger, accuracy-neutral win than any FP8 placement here (changes summation
order, so not bit-identical; not done on this branch).

### Activation / gradient amax over the noise spread (`scripts/amax.py`)

A48 L384, per-sample input scale sigma = logspace(-1, 1), per-sample dY weight 1 / (1 + sigma^2) (an EDM-like loss weighting).
Row amax = the scale each quantizer made x FMAX.

| operand | median row amax @ sigma 0.1 / 1 / 10 | max / min row amax over the tensor | max / min of the per-sample medians |
|---|---|---|---|
| xa (q\|k\|v\|g in) | 3.91 / 3.86 / 3.86 | 2.3 | 1.0 |
| og (out in) | 0.61 / 0.59 / 0.60 | 8.9 | 1.1 |
| xt (expand in) | 3.95 / 3.84 / 3.88 | 2.4 | 1.0 |
| h (squeeze in) | 7.26 / 7.00 / 7.07 | 5.2 | 1.1 |
| dz (dsqueeze in) | 2.34 / 1.10 / 0.024 | 240 | 100 |
| dab (dexpand in) | 2.63 / 1.26 / 0.026 | 555 | 100 |
| dy (dout in) | 7.31 / 1.16 / 0.023 | 803 | 317 |
| dqkvg (dqkvg in) | 1.01 / 0.17 / 0.0036 | 1555 | 279 |

Forward operands are LayerNorm / AdaLN outputs: their scale does not move with sigma (a static per-tensor scale would do). The incoming
gradients move by 100-300x between samples and up to 1555x row to row: a per-tensor scale would put the large-sigma samples' gradients
2^8-2^10 below the max (e4m3 subnormals, 1-2 significant bits). Per-row scales make that irrelevant; with them e5m2 only loses precision
(table below), so the default gradient format is e4m3.

### Speed: step time (fwd + bwd) against the 2.5.14 default (`scripts/speed.py`)

One CUDA graph per variant (forward + backward, single / cond / pair / all parameters differentiated, the weight pack and FP8 weight
quantization inside the graph, i.e. paid every step), same process, interleaved with the default captured twice (same-cubin control);
cool = 2 s idle + one interleaved replay each, 5 reps, median; sustained = 2 s idle, 1 s back-to-back, 3 blocks x 5, mean. H100 node01,
2026-11-18 job 57865 (`res/speed3.jsonl`; jobs 57824 / 57843 in `speed.jsonl` / `speed2.jsonl`). Cells `cool / sustained`, ratio to the
default; MIN_ROWS = 0.

| A:L (M) | 2.5.14 default ms | control | fast (expand, squeeze) | transition (+ dexpand, dsqueeze) | attention (qkvg, out, dqkvg, dout) | all eight |
|---|---|---|---|---|---|---|
| 48:384 (18432) | 3.426 / 3.415 | 0.989 / 1.000 | **0.978 / 0.991** | 0.987 / 0.997 | 1.004 / 1.017 | 1.000 / 1.012 |
| 48:768 (36864) | 7.626 / 7.732 | 0.996 / 1.000 | **0.989 / 0.990** | 0.986 / 0.990 | 1.005 / 1.007 | 0.996 / 0.998 |
| 48:128 (6144) | 1.170 / 1.144 | 0.976 / 0.997 | 0.973 / 0.996 | 1.012 / 1.034 | 1.026 / 1.050 | 1.061 / 1.086 |
| 8:384 (3072) | 0.837 / 0.813 | 0.970 / 0.997 | 0.985 / 1.011 | 1.039 / 1.066 | 1.025 / 1.052 | 1.094 / 1.119 |

Earlier runs of the same variants (forward-only transition `tr_fwd` = `fast`): 48:384 0.984 / 0.989 and 0.985 / 0.991, 48:768 0.988 /
0.990 (twice), 48:256 (12288) 0.980 / 0.993 (control 0.989 / 1.000). Reading: the forward transition placement saves 1-2 % of the step
from M 18432 up, within the control below; the dgrad placements and the attention side do not pay (their row passes and per-step weight
transposes cost what the GEMMs save); `all` loses up to +12 % at small M. Hence `fast` + MIN_ROWS 16384.

### Gradient error against fp64 (`scripts/err.py`; A48 L128, MIN_ROWS 0)

fp64 PyTorch reference on the same weights and the bf16-rounded inputs / dY. Cells: rel rms ||g - g64|| / ||g64|| (%); `unit` = N(0, 1)
inputs and dY; `sigma` = per-sample input scale logspace(-1, 1) and dY scale logspace(-4, 0). Raw: `res/err.jsonl`, table `res/err_table.md`.

| variant | out | d single | d cond | d pair | dW query | dW out | dW expand_a | dW squeeze | dW to_bias | dW adaln (attn) | params median / max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 default (unit) | 0.31 | 0.25 | 0.72 | 0.88 | 1.01 | 0.84 | 0.65 | 0.62 | 0.89 | 0.99 | 0.86 / 1.01 |
| expand | 2.16 | 1.22 | 4.23 | 1.48 | 1.56 | 1.46 | 4.51 | 5.40 | 1.51 | 1.55 | 1.59 / 5.78 |
| squeeze | 1.49 | 0.25 | 1.25 | 0.88 | 1.01 | 0.84 | 0.65 | 0.62 | 0.89 | 0.99 | 0.88 / 3.86 |
| dexpand | 0.31 | 1.09 | 3.51 | 1.37 | 1.45 | 1.35 | 0.65 | 0.62 | 1.36 | 1.44 | 1.39 / 3.94 |
| dsqueeze | 0.31 | 1.09 | 3.54 | 1.39 | 1.48 | 1.35 | 3.79 | 0.62 | 1.38 | 1.45 | 1.45 / 3.82 |
| qkvg | 0.68 | 0.64 | 2.09 | **6.90** | **7.74** | **7.01** | 0.80 | 0.75 | **7.04** | **7.24** | 6.47 / 7.79 |
| out | 0.45 | 0.26 | 0.78 | 0.88 | 1.01 | 0.84 | 0.70 | 0.66 | 0.88 | 0.99 | 0.87 / 3.85 |
| dqkvg | 0.31 | 0.39 | 1.20 | 0.88 | 1.01 | 0.84 | 0.65 | 0.62 | 0.89 | 3.83 | 0.86 / 3.94 |
| dout | 0.31 | 0.39 | 1.20 | 3.89 | 3.90 | 0.84 | 0.65 | 0.62 | 4.06 | 3.86 | 0.88 / 4.06 |
| transition (4) | 2.60 | 1.92 | 6.50 | 2.12 | 2.18 | 2.08 | 5.84 | 5.40 | 2.10 | 2.15 | 2.18 / 6.89 |
| transition, dY e5m2 | 2.60 | 2.64 | 8.82 | 2.76 | 2.82 | 2.76 | 7.39 | 5.40 | 2.83 | 2.79 | 2.83 / 9.31 |
| attention (4) | 0.75 | 0.76 | 2.51 | **7.85** | **8.56** | 7.01 | 0.84 | 0.79 | **7.96** | **8.93** | 7.06 / 9.14 |
| all (8) | 2.69 | 2.06 | 6.95 | 8.08 | 8.78 | 7.27 | 5.87 | 5.42 | 8.12 | 9.13 | 7.32 / 9.29 |
| all, dY e5m2 | 2.69 | 2.78 | 9.29 | 9.45 | 10.06 | 7.49 | 7.42 | 5.42 | 9.56 | 11.26 | 9.21 / 11.83 |
| bf16 default (sigma) | 0.19 | 0.17 | 0.72 | 0.87 | 1.00 | 0.82 | 0.64 | 0.62 | 0.87 | 0.98 | 0.85 / 1.04 |
| transition (sigma) | 0.84 | 0.32 | 6.49 | 0.90 | 1.04 | 0.86 | 5.82 | 5.41 | 0.88 | 1.01 | 1.04 / 7.03 |
| attention (sigma) | 0.33 | 0.19 | 2.37 | 7.78 | 8.55 | 6.95 | 0.64 | 0.62 | 7.84 | 8.97 | 7.23 / 9.39 |

(`fast` = the expand + squeeze rows: out ~2.6 %, d cond ~4-5 %, dW expand / squeeze ~5 %, everything else at or near the default.)
Reading: an e4m3 GEMM operand puts ~2.5-4 % on its own gradient path; the transition placements stay on the transition's parameters and
the conditioning gradient; the attention-side placements spread to every attention parameter and to the pair gradient (q|k|v|g in FP8
moves the logits: d pair 0.9 -> 6.9 %, attention weights 1 -> 7-9 %), the same placement that moved AF3 FoldBench in inference. No NaN /
inf in any case; no saturation possible (row scales).

### Training dynamics (`scripts/curve.py`, `scripts/gaps.py`)

3-block stack, master weights, AdamW lr 3e-4, 500 steps, distilled onto a fixed random teacher stack; per step fresh A48 L128 batches
with per-sample sigma in logspace(-1, 1) and loss weight 1 / (1 + sigma^2); same data and init for every variant (MIN_ROWS 0).
Loss goes 0.74 -> 0.362. `res/curve500.jsonl`.

| variant | final loss (mean last 50) | mean abs rel gap vs bf16 per step | max rel gap (10-step smoothed) | last-50 rel gap | param distance to the bf16 run / bf16 total update | NaN |
|---|---|---|---|---|---|---|
| bf16 default | 0.36180 | - | - | - | 0 | no |
| fp32 (TF32) | 0.36177 | 0.007 % | -0.010 % | -0.008 % | 0.006 | no |
| fast (expand, squeeze) + dexpand | 0.36182 | 0.035 % | -0.223 % | +0.005 % | 0.026 | no |
| transition | 0.36183 | 0.035 % | -0.228 % | +0.009 % | 0.027 | no |
| transition, dY e5m2 | 0.36185 | 0.039 % | -0.225 % | +0.015 % | 0.028 | no |
| attention | 0.36180 | 0.005 % | -0.013 % | +0.002 % | 0.025 | no |
| all | 0.36184 | 0.039 % | -0.234 % | +0.012 % | 0.037 | no |
| all, dY e5m2 | 0.36187 | 0.043 % | -0.235 % | +0.020 % | 0.038 | no |

(The "fast" row of this run was `expand, squeeze, dexpand`; the final `fast` = expand + squeeze, whose forward-only run `tr_fwd` gave
last-50 gap +0.0005 %, mean 0.036 %.) Every FP8 variant ends within 0.02 % of the bf16 loss; the trajectories differ by up to 0.23 %
early. The weights drift 4-6x further from the bf16 run than fp32 does (2.5-3.8 % of the total update). This synthetic distillation is
not sensitive to the attention placements (their loss gap is the smallest although their gradient error is the largest): it does not
clear them for real models.

### Verification

- Switch off = 2.5.14: sha1 of the output and all 27 gradients (single, cond, pair, 24 parameters), separate processes, this tree
  (no switch) against `git archive 1a2bbe42e`: 6 / 6 identical (A48 L384, A8 L384, A48 L128, A6 L200 padded, A5 L384 owned GEMMs, fp32
  A4 L256), repeated after the last change (`res/bit_{base,new,new2}.json`). The FP8 code is a separate extension
  (`token_dit_fp8_train_sm90`); `token_dit_train_rows.cu` and every default kernel are untouched.
- Tests: `tests/gpu/test_token_dit_fp8_train.py` (q8 rows vs torch within one 8-bit rounding, e4m3 / e5m2, widths 768 / 1536 / 3072;
  `swiglu_q8` / `swiglu_bwd_q8` vs the default kernel; each preset's gradients within 15 % of the bf16 route and finite; FP8 route bitwise
  repeatable; switch off after an FP8 call = default bytes) + `tests/gpu/test_token_dit_train_accumulation.py`; CPU gate ruff check /
  format, pyright on the touched files, `tests/cpu/test_fp8_switch.py`.

### Verdict (token DiT)

- Pays, barely: `fast` (forward expand + squeeze in e4m3): -1.1 .. -2.2 % cool, -0.9 .. -1.0 % sustained of the block's training step at
  M >= 18432; within the control below (gate 16384). Gradient error: transition parameters / cond gradient ~5 % (default 0.6-0.7 %),
  others unchanged; loss curve gap <= 0.02 % at the end.
- Does not pay: dgrad placements (dexpand, dsqueeze: row passes + per-step weight transposes = the GEMM gain), all attention-side
  placements (qkvg / out / dqkvg / dout: +0.4 .. +2.6 %; and 7-9 % gradient error on every attention parameter and the pair gradient),
  wgrad (transposed 8-bit copies cost more than the GEMM saves), attention core (latency bound).
- The ceiling is low because 60 % of the step is not GEMM (attention core 20-29 %, row kernels ~40 %). The larger lever for this block
  is the shared-conditioning GEMMs on L instead of A L rows (~10 % of the step, no FP8).
