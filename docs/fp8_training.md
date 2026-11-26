# FP8 training routes (H100, opt-in, engine 3.1.0 line)

Everything here is EXPERIMENTAL, opt-in and off by default. Base: `rel/3.0.0` 1a2bbe42e (= 2.5.14 + the inference FP8 routes,
[fp8.md](fp8.md)). With no training switch set every route is bit-identical to 2.5.14 (sha1 of the output and every gradient, separate
processes). The training switch is separate from the inference `FOLDJET_ENGINE_FP8` presets.

## Token DiT (`conditioned_transition.cuda.token_dit_fp8_train`, branch `fp8t/tokendit`)

Scope: the H100 bf16 (`master`) training block `kernels/integrations/token_dit_train.py` (TokenDiTProtenix / MiniWorld default options,
width 768, 16 x 48 heads, cond 384, pair 128). Scripts / raw rows (not in the tree): `~/miniworld-engine-scratch/fp8t-tokendit/{scripts,res,logs}`. Registry training rows that matter: `token_single_shared_cond` A = 48 (and 8) samples,
L 128-768 (M = A L rows up to 36864).

```
FOLDJET_ENGINE_FP8_TRAIN=1                         # training switch: the token DiT takes RECOMMENDED (round 4: nox + hsave, bit-identical)
FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN=<places>        # explicit: expand, squeeze, qkvg, out (forward); dexpand, dsqueeze, dqkvg, dout (dgrad);
                                              # presets fast, transition, attention, all; "" / 0 = off (wins over FOLDJET_ENGINE_FP8_TRAIN)
FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN_GRAD=e4m3|e5m2  # 8-bit format of the incoming gradients (dgrad A operands); default e4m3
FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN_MIN_ROWS=16384  # row gate (default 16384; every measurement below ran with 0)
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

## Token DiT round 3 (branch `fp8t3/tokendit`): against the 3.0.1 default

Base: `fp8t2/tokendit` (round 2: `nox` / `hsave` byte reductions, behind `FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN`) merged with origin/main 3.0.1 (shared
conditioning on L rows, pair bias on the APB kernels). The baseline is now the 3.0.1 default (A48 L384 3.03 ms; it was 3.43 ms on 2.5.14).
Training placements live in `token_dit_fp8_train.py`, separate from the inference gate `token_dit_fp8.active()`.
Scripts: `tools/fp8_train/tokendit/` (`speed.py`, `run.sh`); raw rows `~/miniworld-engine-scratch/fp8t3-tokendit/res/`.

Placements: `nox` (no fp32 residual copy: the backward reads the block input) and `hsave` (the forward keeps h; swiglu_bwd does not rewrite it)
are byte reductions, bit-identical (no FP8; the fp64 error table below shows identical numbers); `expand` / `squeeze` are the e4m3 forward GEMMs.
`RECOMMENDED` (= `FOLDJET_ENGINE_FP8_TRAIN=1`, preset `recommended`) is now `expand + nox + hsave`; `squeeze` is dropped (see below). Gate: the byte reductions apply at
every M; the FP8 GEMM at M = A L >= 16384.

### Step time (fwd + bwd, one CUDA graph per variant, same process, interleaved; H100 node02, job 58892 `res/speed3.jsonl`)

Cells `cool / sustained`, ratio to the 3.0.1 default; control = the default captured again (same cubin); shipped gate (MIN_ROWS 16384), so at
M < 16384 `rec` = `bytes`.

| A:L (M) | 3.0.1 default ms (cool / sust.) | control / control2 | bytes only (nox, hsave) | rec (expand + bytes) |
|---|---|---|---|---|
| 48:384 (18432) | 3.028 / 3.012 | 0.989 / 0.999, 0.990 / 0.999 | 0.979 / 0.989 | **0.974 / 0.984** |
| 48:768 (36864) | 6.787 / 6.850 | 0.998 / 1.001, 0.998 / 1.000 | 0.985 / 0.989 | **0.981 / 0.986** |
| 48:256 (12288) | 1.979 / 1.957 | 0.989 / 1.001, 0.988 / 1.001 | 0.976 / 0.988 | 0.974 / 0.988 (gated: = bytes) |
| 48:128 (6144) | 1.060 / 1.031 | 0.968 / 0.994, 0.966 / 0.993 | 0.955 / 0.981 | 0.956 / 0.982 (gated) |
| 8:384 (3072) | 0.785 / 0.757 | 0.962 / 0.996, 0.950 / 0.983 | 0.945 / 0.978 | 0.948 / 0.981 (gated) |

Earlier runs (jobs 58581, 58592, same variants) agree within 0.3 %. Ungated ablation at M 18432 / 36864 (cool / sustained):
`bytes` 0.979 / 0.989 and 0.984 / 0.991; `expand + bytes` 0.974 / 0.984 and 0.981 / 0.987; `expand + squeeze + bytes` 0.973 / 0.983 and 0.979 / 0.986.
Reading: against the same-cubin control (cool control sits 1 % below the default because of capture order; sustained control = 1.000) the
route saves about 1.5-1.6 % sustained at M 18432 / 36864; the byte reductions are 1.0 % of that, the e4m3 expand GEMM 0.5 %, squeeze +0.1 %
(not worth 5 % dW squeeze error). At M < 16384 only the byte reductions are on: 1.0-1.5 % sustained against control 0.4-0.6 %: smaller than the cool-vs-sustained spread.
Upper bound left for FP8 on this block: expand + squeeze GEMMs ~50 us of 3028 (1.7 %), minus the 15.5 us q8 pass; fusing it into
`res_adaln_b` would give ~0.3 % more (not built).

### Gradient error vs fp64 (A48 L128, MIN_ROWS 0; `res/err2.jsonl`), rel rms %

| variant | out | d single | d cond | d pair | dW expand | dW squeeze | dW query |
|---|---|---|---|---|---|---|---|
| 3.0.1 default (unit) | 0.31 | 0.25 | 0.68 | 0.91 | 0.65 | 0.62 | 1.01 |
| bytes (nox, hsave) | 0.31 | 0.25 | 0.68 | 0.91 | 0.65 | 0.62 | 1.01 |
| expand (= rec) | 2.16 | 1.22 | 4.21 | 1.49 | 4.51 | 5.40 | 1.56 |
| expand + squeeze + bytes | 2.60 | 1.22 | 4.33 | 1.49 | 4.51 | 5.40 | 1.56 |
| default (sigma spread) | 0.19 | 0.17 | 0.68 | 0.89 | 0.64 | 0.62 | 1.01 |
| expand (sigma) | 0.70 | 0.24 | 4.25 | 0.91 | 4.50 | 5.41 | 1.02 |
| expand + squeeze + bytes (sigma) | 0.84 | 0.24 | 4.37 | 0.91 | 4.50 | 5.41 | 1.02 |

### Training curves (3-block distillation, per step fresh A48 L128, MIN_ROWS 0; `res/curve2.jsonl`, `res/curve3.jsonl`)

Mean loss of the last 50 steps. 500 steps, AdamW 3e-4: default 0.361798, fp32 (TF32) 0.361768, bytes 0.361798 (identical), expand + bytes 0.361789,
expand + squeeze + bytes 0.361804 (gaps within 0.003 %). 2000 steps, lr 1e-3: default 0.333832, expand + bytes 0.333780 (-0.016 %),
expand + squeeze + bytes 0.333761 (-0.021 %). No NaN. This is a synthetic distillation, not an end-level result (same caveat as before, and the
qkvg FoldBench precedent: module error did not predict end-level effect). No end-level run (real-model finetune loss curve / FoldBench) was done in this
round; the e4m3 `expand` is the placement that the inference `safe` preset already ships (FoldBench-checked in inference), `squeeze` has no such precedent.

### Verification

- Switch off = origin/main 3.0.1: sha1 of the output and all 27 gradients, six cases (A48 L384, A8 L384, A48 L128, A6 L200, A5 L384, fp32 A4 L256),
  this tree against `git archive origin/main`, separate processes: 6 / 6 identical (`res/bit_new.json` = `res/bit_base.json`). The `nox` / `hsave` bits change nothing in
  the gradients (the error table above, equal numbers).

### Verdict (round 3)

- The route (`expand + nox + hsave`) beats the 3.0.1 default by ~1.5 % (sustained; 2.0-2.6 % cool) at M >= 18432 and ~1.2 % below; but 2/3 of the win comes from
  the bit-identical byte reductions, not from FP8. FP8 alone: ~0.5 % for 4-5 % rms gradient error on the transition parameters and the cond gradient (default 0.65 %).
  The honest ledger: FP8 training for the token DiT is a marginal win (inside the 1 % control spread), so recommended = byte reductions as the main change; `expand`
  stays opt-in (`FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN=expand`, gate 16384) until an end-level training curve exists. Candidates for the main line: `nox` + `hsave` (bit-identical, 1.0-1.5 %) as default-route changes with a
  `backend_versions` bump (cuda only) — a decision for the release owner, not done here.
- Does not pay (unchanged): dgrad placements, attention-side projections, wgrad, attention core.

## Token DiT round 4 (branch `fp8t4/tokendit`): bytes, finer scales, fused cast, real end-level check

Base `fp8t3/tokendit` (71d7d86d4). Scripts `~/miniworld-engine-scratch/fp8t4-tokendit/{scripts,res,ft}`; same-process CUDA-graph timing as round 3
(`tools/fp8_train/tokendit/`), baseline = the 3.0.1 default. H100 node01 / node02, 2026-11-19.

### (1) Byte ledger (default route, bf16 master, 3.0.1)

Saved activations (what autograd keeps between forward and backward, `memled.py`; u = one [M, 768] bf16 tensor = 1.5 KB per row):

| tensor | KB / row | share | consumer in the backward |
|---|---|---|---|
| qkvg (q, k, v, g raw) | 6.1 (4 u) | 19 % | gate g, QK-norm backward |
| ab = [a, b] of the SwiGLU (expand output) | 6.1 (4 u) | 19 % | swiglu_bwd |
| q, k, v copies (qn, kn, vc) | 4.6 (3 u) | 14 % | attention core backward |
| h (hsave: the forward's SwiGLU output) | 3.1 (2 u) | 10 % | dWsq wgrad |
| x1 (fp32 residual after the attention branch) | 3.1 (2 u) | 10 % | AdaLN-b backward |
| xa, og, y, xt, z | 7.7 (5 u) | 24 % | wgrads, gate gradients |
| rest (cond tables, stats, pair, bias) | per call, not per row | | |
| total (A48 L384) | 31.9 | 587 MB | |

| A:L (M) | saved MB | peak fwd MB | peak fwd + bwd MB (over weights + inputs) | `rec` peak | rx1 saved / peak |
|---|---|---|---|---|---|
| 48:128 (6144) | 192 | 220 | 665 | 647 | |
| 48:256 (12288) | 385 | 442 | 1264 | 1225 | |
| 48:384 (18432) | 587 | 672 | 1892 | 1836 | 531 / 1780 |
| 48:768 (36864) | 1167 | 1337 | 3789 | 3675 | 1054 / 3562 |
| 8:384 (3072) | 104 | 118 | 422 | 413 | |

The backward temporaries (~1.2 GB at M 18432) dominate the single-block peak; for a stack of N blocks the saved column is N times.
Largest A at L 384 on one H100 (80 GB, fwd + bwd of one block, `res/oom1.jsonl`, `oom2.jsonl`): default A 1536 fits (M 589824, peak 56.0 GB), 1792 OOM;
`rec` (expand + nox + hsave) 1792 fits (M 688128, peak 77.4 GB), 2048 OOM; `rec` + rx1 1792 (75.3 GB). I.e. +17 % rows from the byte reductions.

Per-kernel DRAM bytes (ncu, A48 L384 default step, `res/ledger_d384.md`): 3274 MB read + 1561 MB written per step = 4.8 GB in 3.38 ms (ncu clocks): 1.4 TB/s on
average. The step is not DRAM-bound as a whole: GEMMs ~1.1 ms, attention (fwd, dqb, dkv) ~0.8 ms at the tensor / latency limits; the row kernels (~1.1 ms, 33 %) run at
2.3-2.9 TB/s: res_adaln_b_bwd 350 MB / 150 us, swiglu_bwd 317 MB / 110 us, qknorm_bwd 267 MB / 109 us, adaln_a_bwd 239 MB / 96 us, sample_sum2 173 MB / 62 us.
A byte only pays where a kernel that has to run anyway stops writing / reading it.

### (2) Saved activations in 8 bit: no-go; the bit-identical ones: yes

An 8-bit copy of a saved tensor costs an extra write in the kernel that makes it (the producers are cuBLAS GEMMs, which cannot write e4m3 with row scales)
and saves the same bytes at the read: time-neutral at best; the gain is memory only. Measured error of the dominant candidate, `ab` stored as e4m3 with
a per-row (or per-half-row) scale and dequantized in `swiglu_bwd` (emulation in the real kernel path, `abq_patch.py`, A48 L128, rel rms % vs fp64):

| variant | out | d single | d cond | dW expand | dW squeeze | dW query |
|---|---|---|---|---|---|---|
| default | 0.31 | 0.25 | 0.68 | 0.65 | 0.62 | 1.01 |
| default + ab e4m3 (row scale) | 0.31 | 0.89 | 2.84 | 2.89 | 4.18 | 1.32 |
| default + ab e4m3 (per half) | 0.31 | 0.89 | 2.85 | 2.89 | 4.18 | 1.32 |
| default + ab fp16 (same bytes as bf16) | 0.31 | 0.25 | 0.68 | 0.65 | 0.62 | 1.01 |
| rec (expand) | 2.16 | 1.22 | 4.21 | 4.51 | 5.40 | 1.56 |
| rec + ab e4m3 | 2.16 | 1.48 | 5.05 | 5.31 | 5.40 | 1.79 |

ab in e4m3 saves 3 KB / row (9.5 % of the saved bytes, 56 MB at M 18432) and moves the gradient error of the transition 4x (0.7 -> 2.9 %). Not built as a kernel.
qkvg / qn / kn / vc in e4m3 is the placement that moved FoldBench in inference: not tried. x1 as bf16 would change the forward output (the residual stream is fp32).
Bit-identical byte lever tried: `rx1` (x1 is not stored: `res_c` and the AdaLN-b backward recompute x1 = x + sigmoid(Gg + bg1) y from the block input; same fp32
expression, output and every gradient bit-identical, tests): saved 589 -> 531 MB (-9.9 %), peak -3 %, but the step is 0.3-0.5 % SLOWER (the recomputation costs more than
the 2 u write it saves; bytes 0.978 / 0.989 -> 0.983 / 0.992 at M 18432). Memory option only (`FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN=...,rx1`), not in `recommended`.

### (3) Finer-grained scales: no help (the error is the 3-bit mantissa, not the scale)

Error floor of the expand GEMM alone (real block activation xt [3072, 768] x [Wa; Wb]^T, fake quantization, fp32 accumulate, rel rms vs fp64, `gemm_floor.py`):

| quantization | rel rms % |
|---|---|
| bf16 cuBLAS | 0.15 |
| shipped: activation per row, weight per output channel (e4m3) | 3.34 |
| activation 1x128 tiles, weight 128x128 blocks | 3.31 |
| activation 1x64, weight 64x64 | 3.26 |
| activation per row only (weight exact) | 2.36 |
| weight per channel only (activation exact) | 2.36 |
| activation hi + lo (2 x e4m3), weight per channel | 2.36 |
| activation hi + lo and weight hi + lo (4 GEMMs, ~fp16) | 0.08 |

An e4m3 element has a 2.6 % rms rounding error whatever the scale granularity (the data have no dynamic-range problem: per-row amax / median ~ 4); tiles do not move it. Only
both operands in two e4m3 pieces (4x the GEMM work) reach 0.08 %, which cannot beat bf16. DeepSeek's 0.25 % is a loss-level number with fp32 promotion over 128-element
tiles on far larger models; the target <= 1.5 % rms gradient error is not reachable with a single e4m3 pass of either operand (each one alone gives 2.4 %).

### (C) Fused cast: `xt` e4m3 made in `res_adaln_b` (`res_adaln_b_q8`)

The row max of the bf16-rounded xt is a block max in the AdaLN-b kernel (one block per row); the e4m3 copy and the row scale are written next to xt. Bit-identical to the
separate `q8_rows` pass (sha1 of output and 27 gradients, A48 L384, A8 L384, A5, A6 L200, A48 L128). Step (`rec` = expand + nox + hsave, cool / sustained, vs control):
unfused 0.973 / 0.983 -> fused 0.971 / 0.980 at M 18432 (0.981 / 0.987 -> 0.980 / 0.985 at 36864): -0.2 / -0.3 %, as predicted. `FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN_FUSE_Q8=0` restores the pass.

### (4) Step time against the 3.0.1 default (same process, one CUDA graph per variant, cool / sustained, final tree, job 59085 `res/speed6.jsonl`)

| A:L (M) | 3.0.1 default ms | control / control2 | bytes (nox, hsave) | rec unfused | rec (expand + nox + hsave, fused q8) |
|---|---|---|---|---|---|
| 48:384 (18432) | 3.038 / 3.033 | 0.989 / 0.999, 0.990 / 0.999 | 0.979 / 0.988 | 0.973 / 0.983 | **0.971 / 0.980** |
| 48:768 (36864) | 6.830 / 6.946 | 0.994 / 1.002, 0.996 / 1.001 | 0.985 / 0.993 | 0.981 / 0.987 | **0.980 / 0.985** |
| 48:256 (12288) | 1.985 / 1.967 | 0.984 / 1.001, 0.987 / 0.999 | 0.973 / 0.989 | gated = bytes | 0.975 / 0.989 (gated) |
| 48:128 (6144) | 1.061 / 1.032 | 0.971 / 0.997, 0.966 / 0.992 | 0.955 / 0.981 | gated | 0.954 / 0.980 (gated) |
| 8:384 (3072) | 0.776 / 0.745 | 0.958 / 0.997, 0.960 / 0.997 | 0.953 / 0.991 | gated | 0.952 / 0.990 (gated) |

Against the same-cubin control (sustained 0.999-1.002) `rec` saves 1.7-1.9 % at M >= 18432 and `bytes` alone 0.9-1.1 %; below the gate (M < 16384) only `bytes` is on: 0.2-1.1 % sustained
(at the noise level for 3072 rows). Earlier job 59002 (rx1 included, `res/speed5.jsonl`) agrees within 0.3 %. The FP8 part of `rec` is therefore ~0.7-0.9 % of the step.

### (5) Real end-level check: Protenix v1 fine-tune, paired A/B (`ft/`, `res/real_ab.md`)

Real recipe (`scripts/train_protenix1.py --recipe protenix`, resumed from the 1400-step engine fine-tune checkpoint, the lr at 7e-4), real PDB data stream (same seed -> the same
12 batches / noise draws in each run), effective batch 256 on ONE GPU, `--precision bf16` (the whole network, the token DiT included, under the bf16 autocast, so the block
under test is the bf16 training block at A 48, L 384: M 18432 >= the FP8 gate; the production recipe `af3_default` keeps the diffusion head in fp32, where this route is inactive), `--step-graph`.
Diffusion loss per step (mean of the 48 samples, the same samples):

| run | steps 1400 .. 1411 mean diffusion loss | paired gap vs bf16 default |
|---|---|---|
| A bf16 default (3.0.1 route) | 0.0808 | - |
| B FP8 `rec` (expand + nox + hsave) | 0.0824 | +1.96 % (every step +1.5 .. +2.7 %, all 12 positive) |
| C `af3_default` (fp32 diffusion head), 6 steps | 0.0775 (A over the same 6: 0.0814) | -4.8 % |

Bit-identical byte reductions leave the loss unchanged by construction. The FP8 `expand` in 24 DiT blocks adds a systematic +2 % to the diffusion loss at fixed weights
(about a third of the penalty of running the diffusion head in bf16 instead of fp32 at all, which the production recipe forbids). The synthetic distillation (<= 0.02 %)
did not predict this: the FoldBench lesson again. Round-trip: the first run (job 59009) gave NaN from step 1400: a real weight output channel with amax ~ 1e-37 made
the per-channel scale 1e-39 and 1 / scale = inf; every quantizer now floors the scale at 1e-30 (`test_tiny_and_zero_rows_stay_finite`), the unit tests and the random-weight runs had never seen such a channel.

### Verdict (round 4)

- FP8 `expand` for training: NO-GO as a recommendation. The step gain is 0.7-0.9 % (inside the 1 % control spread), the gradient error of the transition 4-5 %, and the real
  fine-tune shows +2 % diffusion loss at fixed weights. Kept opt-in (`FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN=expand`, gate 16384, default off) for the record; `RECOMMENDED` (`FOLDJET_ENGINE_FP8_TRAIN=1`) is now `nox + hsave` only.
- The bit-identical byte reductions `nox` + `hsave` are the whole win: 0.9-1.1 % sustained (1.5-2.1 % cool) at M >= 18432, 0.5-1.1 % sustained below, peak -3 %, largest A at L 384 1536 -> 1792 (+17 %), output and 27 gradients
  bit-identical to 3.0.1. Release candidate: branch `fp8t4/tokendit-bytes-default` (see the report: diff, `TokenDiTProtenix.backend_versions["cuda:h100_sm90"]` 5 -> 6, 292 accuracy + 292 speed cells owed on h100_sm90).
- Finer scales, 8-bit saved activations (ab), `rx1`: no-go (above). Fused q8: kept (-0.2 / -0.3 %, bit-identical).

### Round 4 verification and the release candidate

- Real fine-tune D (job 59102, `nox + hsave` only, steps 1400-1402): loss and diffusion loss equal to the default run A to the printed 4 digits (0.3677 / 0.3644 / 0.3457), as
  the bit-identity says; A is also the evidence that the run is repeatable.
- Tests (H100): `test_token_dit_fp8_train.py` + `_bytes.py` + `train_accumulation` + `shared_cond` + `pair_apb`: 59 passed (this branch); 62 passed (release branch);
  `tests/cpu/test_fp8_switch.py` 19 passed; pyright 0 errors; ruff check / format clean on the touched files (`token_dit.py`'s pre-existing format drift left alone).
- Release candidate (a decision for the user): branch `fp8t4/tokendit-bytes-default` = this branch + (i) `DEFAULT_BYTES = {nox, hsave}` in `token_dit_fp8_train.active()` (default-on for H100 bf16
  default-option width-768 blocks; `FOLDJET_ENGINE_TOKEN_DIT_TRAIN_BYTES=0` restores the 3.0.1 route), (ii) `TokenDiTProtenix.backend_versions["cuda:h100_sm90"]` 5 -> 6 (TokenDiTMiniWorld inherits it;
  Boltz2 / RF3 / Chai1 keep their own 7 because their blocks are not the default-option block), nothing else (no module `version`, no kernel / table version). Output and 27 gradients are
  bit-identical to 3.0.1 (sha1, 6 shapes, `res/bit_release.json` = `res/bit_base301.json`), so the re-measure is for the record only: `judge owed` with the bump = 292 accuracy + 292 speed cases
  on h100_sm90 (`res/owed_bump.txt`; 0 owed before). The kernel change is in `token_dit_train_rows.cu` (new template instantiations, new kernels `res_adaln_b_q8`; the
  default kernels' existing instantiations are untouched) and `token_dit_train.py`; the FP8 module `token_dit_fp8_train.py` ships opt-in as in this branch. The prebuilt kernels wheel must be rebuilt (the rows source changed).

## OuterProductMean (`kernels/integrations/opm_fp8_train.py`, branch `fp8t/opm`, 2026-11-18)

### Switch

| env | effect |
|---|---|
| `FOLDJET_ENGINE_OPM_FP8_TRAIN=e4m3` | FP8 forward + backward products, dO quantised to e4m3 (**recommended form**: more accurate than e5m2 here, see the error table) |
| `FOLDJET_ENGINE_OPM_FP8_TRAIN=e5m2` | same, dO in e5m2 (e4m3 x e5m2 products) |
| `FOLDJET_ENGINE_FP8_TRAIN=1` | turns the route on as `e4m3` unless `FOLDJET_ENGINE_OPM_FP8_TRAIN` is set |
| `FOLDJET_ENGINE_OPM_FP8_TRAIN_MINS=<s>` | padded-depth gate (default 1024; 0 = none) |

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

- The FP8 code is all `#if defined(FOLDJET_ENGINE_OPM_FP8)` in `csrc/opm_epilogue.cu` (the default builds preprocess to the same text) and
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

Offer `FOLDJET_ENGINE_OPM_FP8_TRAIN=e4m3` only as an experiment for deep-MSA training (S >= 2048: 0.58-0.72x of the module step) and for
L 256-768 at depth 1024 with d_pair 128 / 256 (0.81-0.95x); do not use it at d_pair 384 / 512 or L 128. Gradient error is 7-12x the
bf16 default's (3-4 % rms); the 300-step distillation curve matches within 0.2 %, but an end-level training run is needed before any
default.

### Progress log

- 00:25 worktree `fp8t/opm` from 1a2bbe42e; stage profile of the default (main checkout by mistake, then re-measured on this tree).
- 01:00 first route (FP8 dgrad planes, transposes, mixed cuBLASLt): correct, deterministic; 0.92x at L384 S1024 -- dgrad 2.6x slower.
- 01:35 TMA stores + one atomic per warp, device scale kernels: 0.89x / 0.84x / 0.65x (L384 / L768 S1024 / L768 S4096).
- 02:00-05:10 session stopped (API limit). 08:00 tests, bit-identity, error, training curve, speed sweep, this doc.

## OuterProductMean rounds 2 + 3 (branch `fp8t3/opm`, base 3.0.1 = origin/main f5be1df46, 2026-11-19)

Everything below is opt-in (`FOLDJET_ENGINE_OPM_FP8_TRAIN=e4m3`), default OFF; with the switch off the training step is bit-identical to 3.0.1
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
- `tools/fp8_train/opm_step_bench.py` (ready to run, see its docstring).

### Step time vs the 3.0.1 default (CUDA graph fwd + bwd, same process, interleaved, same-cubin control, S 1024 unless noted)

`tools/fp8_train/opm_step_bench.py`, raw `scratch fp8t3/res/bench_final.jsonl`. Cells = e4m3 / default, `cool / sustained`; ctl = default again (same cubin),
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

## OuterProductMean round 4: saved-activation bytes, finer scales (branch `fp8t4/opm`, 2026-11-19)

Scripts / raw rows: scratch `~/miniworld-engine-scratch/fp8t4/` (`scripts_opm/memprof.py`, `emu.py`; `res/mem4b.jsonl`, `fit4.jsonl`, `emu.jsonl`, `bench4.jsonl`, `err_r4b.jsonl`).
Default route unchanged and bit-identical to 3.0.1 with the switches off (9 / 9 sha1 cases, `res/bit4_tree.jsonl`); tests 16 / 16.

### (1) What the OPM block saves for backward (eager step, `torch.autograd.graph.saved_tensors_hooks`, GB = 10^9 B)

Saved bytes per block, 64/128 unless noted. The product O ([32N, 32N] bf16) is 85 % of the default's saved bytes at S 1024 (L768: 1.21 of 1.42 GB):
the A2 / BT operands (bf16 or e4m3) and `m` are small next to it, so "FP8 saved activations" of the operands alone buys nothing (round 3's e4m3 A2 / BT: -0.05 GB at L768).

| shape | default saved | round 3 (e4m3 A2/BT, bf16 O) | O as e4m3 (`fp8`) | O recomputed (`recompute`) | peak of the step: default / r3 / fp8 / recompute |
|---|---|---|---|---|---|
| 64/128 L256 S1024 | 0.205 | 0.188 | 0.121 | 0.071 | 0.372 / 0.356 / 0.423 / 0.373 |
| 64/128 L384 S1024 | 0.408 | 0.383 | 0.232 | 0.106 | 0.786 / 0.760 / 0.911 / 0.785 |
| 64/128 L768 S1024 | 1.421 | 1.370 | 0.766 | 0.212 | 2.930 / 2.880 / 3.484 / 2.930 |
| 64/128 L384 S4096 | 0.723 | 0.621 | 0.470 | 0.420 | 1.099 / 0.999 / 1.150 / 1.099 |
| 64/128 L768 S4096 | 2.051 | 1.846 | 1.242 | 0.840 | 3.557 / 3.356 / 3.960 / 3.558 |
| 128/256 L768 S1024 | 1.522 | 1.471 | 0.867 | 0.313 | 3.232 / 3.183 / 3.787 / 3.233 |
| 128/384 L768 S1024 | 1.523 | 1.471 | 0.867 | 0.314 | 3.534 / 3.485 / 4.089 / 3.536 |

The peak inside one block's step is set by backward's transients (bf16 O, dO planes): `recompute` keeps it at the default's, `fp8` adds the conversion temporaries. What
shrinks is what stays resident between forward and backward, i.e. what a stack of blocks keeps. Four OPM blocks run forward, one backward (64/128, S 1024, H100 80 GB, `memprof.py fit4`,
largest L that fits, step 128): default **2560**, round 3 (`e4m3`) 2688, `e4m3` + `recompute` **3584** (1.96x the pair tokens). A single block (one forward + backward) is limited by
the transient O: all routes fit L 4096 (OOM at 4224); depth: all routes stop at S 58080 (the prologue's shared-memory tile), not by memory.

### (2) FP8 storage of O: measured, no-go as a Python-level cast; recompute is the memory lever

`FOLDJET_ENGINE_OPM_FP8_TRAIN_OSTORE=fp8|recompute|bf16` (default `bf16` = round 3). Step time vs the 3.0.1 default (CUDA graph, same-cubin control, cool / sustained; ctl 1.00 / 1.01-1.04):

| shape | round 3 (`e4m3`) | `fp8` (e4m3 O, chunked torch casts) | `recompute` |
|---|---|---|---|
| 64/128 L256 S1024 | 0.853 / 0.864 | 1.79 / 1.76 | 0.988 / 0.965 |
| 64/128 L384 S1024 | 0.838 / 0.859 | 1.87 / 1.84 | 0.987 / 0.946 |
| 64/128 L768 S1024 | 0.822 / 0.817 | 1.91 / 1.85 | 0.984 / 0.926 |
| 64/128 L384 S4096 | 0.664 / 0.629 | 0.994 / 0.918 | 0.819 / 0.747 |
| 64/128 L768 S4096 | 0.645 / 0.607 | 0.982 / 0.966 | 0.809 / 0.764 |
| 128/256 L384 S1024 | 0.931 / 0.963 | 1.80 / 1.78 | 1.059 / 1.024 |
| 128/256 L768 S1024 | 0.928 / 0.947 | 1.88 / 1.84 | 1.066 / 1.027 |
| 128/384 L768 S1024 | 0.963 / 0.977 | 1.80 / 1.77 | 1.087 / 1.042 |

- `fp8` (amax + cast in forward, dequant in backward, separate passes, no fused kernel): about 1-7 ms of extra elementwise passes per step (L256-768) erases the FP8 win (1.8-1.9x of the default at S 1024),
  and dW_out error rises 2.8 -> 3.9 % rms (3.4 % max; every other gradient unchanged). A fused version (cast inside the epilogue, dequant inside `opm_dwo`) is the only way it could pay;
  not built (the epilogue kernel's O load path is the wgmma / TMA one; ideal gain ~0.4 ms of 5.4 at L768 for +0.4 % y error). No-go at this level.
- `recompute` (redo the FP8 product in backward from the saved A2 / BT, bit-identical gradients to round 3): saved bytes -70 to -85 % at S 1024 (L768: 1.42 -> 0.21 GB), -59 % at S 4096;
  step time at parity to 7 % faster than the 3.0.1 default for 64/128 at S 1024 (0.93-0.99), 18-25 % faster at S 4096, but 3-9 % SLOWER than the default for d_pair 256 / 384 (gate it to d_pair 128).
  It trades the speedup for memory: the FP8 route's step time at S 1024 goes from 0.82-0.86x to 0.93-0.99x of the default while the resident saved bytes drop 85 %.
  Recommendation: `OSTORE=recompute` when the OPM blocks are stacked and memory bound (crop / batch headroom: +40 % L on four blocks), `bf16` when step time is what counts.

### (3) Finer-grained scales: no-go (emulation, `scripts_opm/emu.py`)

Float32 emulation of the product chain with fake-quantised e4m3 operands, each GEMM operand quantised along its own contraction axis (both layouts from the source, as the prologue does), gradient
rms / max error vs a float64 reference, 64/128 L192 S512 and 128/256 L128 S512 (`res/emu.jsonl`):

| scheme | y | d msa | d W_left | d W_out | cost |
|---|---|---|---|---|---|
| per tensor (pow2) | 2.59 / 1.49 | 3.75 / 3.26 | 3.43 / 3.71 | 2.85 / 1.73 | 1x e4m3 |
| per row / column (one pow2 scale per non-contracted index) | identical to the digit | identical | identical | identical | epilogue scale vectors |
| 1 x 128 blocks along K (DeepSeek-style) | identical | identical | identical | identical | block rescale in the accumulation |
| hi + lo e4m3 residual on all operands (3 products) | 0.06 / 0.03 | 0.09 / 0.08 | 0.09 / 0.08 | 0.07 / 0.05 | 3 FP8 GEMMs = 1.7x bf16 time |

Reason: with power-of-two scales the e4m3 rounding error (3 mantissa bits, ~2.7 % rms per element) does not depend on the scale; scaling granularity only helps when a tensor has outliers that make the
small values underflow, and these activations do not (peaks 11-24 of 448, no subnormals). A K-sum of independently rounded terms keeps the per-element relative error (random signs), so the target <= 1.5 %
rms is not reachable with single e4m3 products at any scale granularity; a hi + lo split reaches 0.1 % (below bf16's 0.4 %) but needs three FP8 GEMMs per product (slower than bf16). The error stays
3-4 % rms (round 3). The DeepSeek-V3 numbers come from fp32 accumulation promotion + fine scales on activations with outliers, and from a loss-level (not per-gradient) error.

### (4) Cross-check of the numbers

- Gradient error of the three O modes (L384 AF3, 64/128, `res/err_r4b.jsonl`): `e4m3` and `recompute` identical (y 2.64 / 1.46, d msa 3.77 / 3.44, d W_out 2.79 / 1.52 %); `fp8` identical except d W_out 3.94 / 3.36.
- Training curve: `recompute` gradients are bit-identical to round 3, so round 3's curve (+0.22 % at the last 50 steps) applies; `fp8` was not run (no-go on time).

### Open

- Fused O quantisation (epilogue writes e4m3 O + scale, `opm_dwo` dequantises) is the one lever that could cut both bytes and time; ~0.4 ms of 5.4 at L768; not built.
- C (fused cast + transpose from the high-precision source) is already in the route since round 2 (prologue writes both layouts; dgrad writes both dO planes).
- To make crop headroom usable, `recompute` has to be chosen by the caller (env or `opm_fp8_train.OSTORE`); default stays `bf16`.

## MSAPairWeightedAveraging (`kernels/integrations/pwa_fp8_train.py`, branch `fp8t/pwa`, base 1a2bbe42e = rel/3.0.0)
**Result (H100, 8 x 32 widths):** FP8 o / dv / dw (cuBLASLt, operands written by fused producers, do e4m3, delayed per-tensor scales) makes
the training step (fwd + bwd) **0.85-0.91x** of the 2.5.14 default (cool; 0.88-0.92 sustained) at the registry training shapes; gradient error
against fp64 rises from 0.3-0.7 % (default) to 2-5 % rms; a synthetic 1000-step distillation ends ~1 % higher in loss. Opt-in only
(`FOLDJET_ENGINE_PWA_FP8_TRAIN=1`), default bytes unchanged. The `pwa_wide` (AF3-family) widths are not served (upper bound 4-7 %).

### Where the training step's time goes (2.5.14 default, measured)

Step = forward + backward of one module call (`torch.autograd.grad` of the output, all inputs and parameters), master weights, p_drop 0.25,
one CUDA graph (median of 3 x 20 replays); stage times = torch profiler kernel times of eager steps (mean of 5). Scripts / raw rows:
`~/miniworld-engine-scratch/fp8t-pwa/{scripts,res}` (`prof.py`, `res/prof0.jsonl`).

8 x 32 (`pwa_train`, native kernels), S 1024, N 768, d_pair 128: step 3.17 ms.

| stage | work | us | share | bound (ncu not run; from the GEMM rate / bytes) | FP8 placement |
|---|---|---|---|---|---|
| `pair_fwd3` | LN_z, z [N^2,128] x Wb, softmax | 68 | 2 % | memory (z read) | none (z is the module input) |
| `ln_vg` | LN_m, y [S N, 64] x Wv^T -> v [H][N][S C] | 216 | 7 % | memory (v write) | v as e4m3 (vT + natural) |
| contraction o = w v | per head [N,N] x [N, S C], 309 GFLOP | 446 | 14 % | tensor (cuBLAS bf16 ~690 TFLOP/s) | **w e4m3, v e4m3** |
| `pwa_gateout` | gate GEMM, sigmoid, out GEMM, dropout, residual | 325 | 10 % | latency (inference ncu: DRAM 61 %, tensor 20 %) | none (o e4m3 tried in inference: no gain) |
| `pwa_glue3` | do = (dout Wo) g, dgp, dWo partials | 643 | 20 % | latency / memory (reads o, y, dout; writes do, dgp) | do written as FP8 (natural + transposed) |
| dv = w^T do | per head [N,N]^T x [N, S C], 309 GFLOP | 422 | 13 % | tensor | **w e4m3, do e5m2 / e4m3** |
| dw = do v^T | per head [N, S C] x [S C, N], 309 GFLOP, K = 32768 | 340 | 11 % | tensor | **do e5m2 / e4m3, v e4m3** |
| `dgv_bwd` | dy = [dgp\|dv][Wg;Wv], LN backward, dWgv | 422 | 13 % | memory (reads dgv [S N 512] + dv) | none (dWgv: 2 x 25 GFLOP inside a memory-bound pass) |
| `pair_bwd3` | softmax / proj_z / LN_z backward | 231 | 7 % | memory | none |
| casts, reduces, mask, transposes | | ~50 | 2 % | launch | none |

The three key contractions are 38 % of the step (1208 us). The projection GEMMs (Wv, Wg, Wo and their dgrad / wgrad) live inside the fused
memory / latency-bound kernels (`ln_vg`, `pwa_gateout`, `pwa_glue3`, `dgv_bwd`), 25-51 GFLOP each at 15-20 % tensor use: e4m3 operands there
cannot shorten an MMA-bound part (there is none). Other shapes: S 2048 N 768 contractions 896 + 838 + 671 = 2405 of 5923 us (41 %);
S 4096 N 768 4808 of 11443 us (42 %); S 1024 N 384 (fused `pwa_fwd2` forward 302 us, dv 152, dw 158) 22 % of 1410 us (+ the fused forward).
Not native (plain builds; default at every shape but the six): S 1024 N 512 2.12 ms, S 384 N 768 1.67 ms -- the plain forward / `pwa_plain2`
dv / Triton pair backward.

FP8 GEMMs alone (`gemm.py`, `res/gemm0.jsonl`, CUDA graph, us, bf16 cuBLAS bmm -> cuBLASLt FP8 with fp32 accumulation):

| C:S:N | GFLOP | o | dv | dw | sum bf16 -> FP8 |
|---|---|---|---|---|---|
| 32:1024:384 | 77 | 161 -> 115 | 161 -> 115 | 166 -> 76 | 488 -> 306 (0.63x) |
| 32:1024:768 | 309 | 454 -> 300 | 512 -> 331 | 416 -> 243 | 1382 -> 874 (0.63x) |
| 32:2048:768 | 618 | 907 -> 698 | 1000 -> 649 | 813 -> 481 | 2720 -> 1828 (0.67x) |
| 32:4096:768 | 1237 | 1988 -> 1321 | 1988 -> 1323 | 2111 -> 1120 | 6087 -> 3764 (0.62x) |
| 32:384:768 | 116 | 178 -> 113 | 178 -> 125 | 156 -> 93 | 512 -> 331 (0.65x) |
| 32:1024:512 | 137 | 235 -> 164 | 247 -> 167 | 194 -> 104 | 676 -> 435 (0.64x) |
| 32:1024:1024 | 550 | 827 -> 563 | 824 -> 520 | 727 -> 418 | 2378 -> 1501 (0.63x) |
| 8:1024:768 (AF3 width) | 77 | 122 -> 79 | 122 -> 80 | 104 -> 63 | 348 -> 222 (0.64x) |
| 16:1024:768 | 155 | 233 -> 153 | 249 -> 164 | 204 -> 134 | 686 -> 451 (0.66x) |

Operand layouts. Hopper's FP8 MMA takes K-major operands only, and each contraction runs over a different index: o over j (w row-major,
v needs j-contiguous = vT), dv over i (w needs w^T, do needs do^T), dw over S*C (do and v as they are). So v must exist as vT and natural,
do as natural and do^T, w as w and w^T. A standalone quantize / transpose pass over a [H, N, S C] tensor costs more than the whole GEMM
gain (PyTorch: 2.1 ms quantize, 1.4 ms byte transpose at S1024 N768 against a 0.5 ms saving), so the producers write them:

- `ln_vt8n` (`csrc/pwa_vt8.cu` built with `-DVT_NAT=1`): `ln_vt8` (inference FP8 route's producer: y bf16 = `ln_vg`'s bytes, vT e4m3) plus
  v e4m3 natural from a second wgmma with the roles swapped (rows n, columns hc: consecutive channels pack into one e4m3x2), one more TMA
  box store. Replaces `ln_vg`; v bf16 is never written (training saves v e4m3: half the bytes).
- `pwa_glue3_f8` (`csrc/pwa_glue3_f8.cu`): `pwa_glue3` with the do epilogue writing e5m2 (or e4m3) do [64 i][64 x] and do^T [64 x][64 i]
  tiles (two 4 KiB TMA boxes in the bf16 tile's staging) and the amax; no bf16 do.
- `w_quant` (inference route) / `w_quant_t` (new): w and w^T as e4m3 of 256 w (exact power of two; w <= 1 cannot saturate), N^2 H bytes.
- `scale_update`: the delayed scale of one site in one launch.

Upper bound at S1024 N768: the contractions at the isolated FP8 rate save 1208 x (1 - 0.63) = ~450 us = 14 % of the step; producers
cost ~0 if fused at equal bytes (vT + v natural e4m3 = v bf16 bytes; do + do^T FP8 = do bf16 bytes) -> ~0.86x.

### Enable

| env / attribute | effect |
|---|---|
| `FOLDJET_ENGINE_PWA_FP8_TRAIN=1` (or `all`; or `FOLDJET_ENGINE_FP8_TRAIN=1`) / `pwa_fp8_train.PLACES` | FP8 o, dv, dw (fused producers). A subset (`fwd`, `dv`, `dw`, comma list) runs the PyTorch-quantized prototype (ablation only: its quantize passes cost 2+ ms) |
| `FOLDJET_ENGINE_PWA_FP8_TRAIN_DO` | `e4m3` (default) or `e5m2` for do |
| `FOLDJET_ENGINE_PWA_FP8_TRAIN_MARGIN` / `_HISTORY` | delayed-scale headroom (2) / amax history length (16 calls) |
| `FOLDJET_ENGINE_PWA_FP8_TRAIN_IMPL=proto` | PyTorch quantize passes instead of the fused producers |
| `pwa_fp8_train.overflow_count()` / `reset()` | saturated calls since reset (one host sync) / forget the scale histories |

Scope: `pwa_train` widths (d_msa 64, 8 x 32; d_pair 128 / 256 / 384 / 512), H100, N % 128 == 0, N <= 1024, S even. With the switch on the
route runs the native kernels at every such shape (the bf16 default runs them only at the six qualified shapes). Not served: the `pwa_wide`
widths (AF3 family 8 x 8 / 8 x 16, BIG), B200, inference (the inference FP8 route is `pwa_fp8`). Scales: v and do per-tensor power of two,
delayed (max of the last 16 calls' amax x 2 / FMAX; recorded by the producers, one `scale_update` launch per site, CUDA-graph safe; first call
of a module seeds from its own tensor, eager; an all-zero first tensor -- zero-initialised `to_out` -- re-seeds on the next call); w fixed 2^-8.
Saturation: satfinite cast, counted (`overflow_count`), the history covers it on the next call; 0 overflows in every run here. The parameter-
gradient reductions are unchanged (native fixed-order partials): the FP8 step is deterministic (test: eager == eager == graph replay, bytes).

### Step time against the 2.5.14 default (`speed.py`, `res/sp2.jsonl`, `res/sp1.jsonl`)

One process; fwd + bwd (`autograd.grad` of the output w.r.t. msa, pair and every parameter) of one module call, master weights, p_drop 0.15,
each variant one CUDA graph; def / ctl (same-cubin control) / FP8 interleaved; cool = 2 s idle + short burst, median of 3; sustained = after
1 s of back-to-back replays, mean of 3. Same-cubin control 0.986 .. 1.006. Cells FP8 / default, `cool / sustained`.

| S:N[:d_pair] | default ms (cool) | FP8 do e4m3 | FP8 do e5m2 | native bf16 (attribution) |
|---|---|---|---|---|
| 1024:768 | 3.27 | 0.880 / 0.906 | 0.875 / 0.900 | 1.00 |
| 1024:384 | 1.44 | 0.910 / 0.918 | 0.908 / 0.916 | 1.00 |
| 2048:768 | 6.18 | 0.860 / 0.879 | 0.852 / 0.883 | 1.00 |
| 4096:768 | 11.77 | 0.871 / 0.884 | 0.861 / 0.883 | 1.00 |
| 2048:384 | 2.75 | 0.888 / 0.895 | 0.881 / 0.890 | 1.00 |
| 1024:768:256 (Chai-1) | 3.47 | 0.889 / 0.907 | 0.886 / 0.900 | 1.00 |
| 384:768 (plain default) | 1.70 | 0.801 / 0.808 | 0.796 / 0.806 | 0.877 |
| 1024:512 (plain default) | 2.16 | 0.810 / 0.825 | 0.810 / 0.828 | 0.895 |

At shapes outside the six the default runs the plain builds; the "native bf16" column (`NATIVE_ONLY`, sp1) is the same route with no FP8:
there about 0.88-0.90 of the gain is the native kernels and FP8 adds ~0.90-0.91x on top.

Per stage at S1024 N768 (profiler, us; default -> FP8 e4m3): `ln_vg` 217 -> `ln_vt8n` 259 (+42: the natural-v byte scatter; launch tuned
1 block/SM x 4 stages, 310 at 2 x 3); o 448 -> 293; `pwa_glue3` 652 -> `pwa_glue3_f8` 658; `w_quant_t` +16 (+ `w_quant` ~7); dv 448 -> 293;
dw 387 -> 233; `scale_update` 2 x ~2; every other stage unchanged. Step kernel sum 3273 -> 2872 us (0.877). Against the upper bound
(~0.86x) the producers give back ~2 % (the natural v / do^T byte stores).

### Gradient error against fp64 PyTorch (`err.py`, `res/err1.jsonl`)

Module in float64 on the PyTorch backend, same inputs; random weights (fan-in^-0.5, `to_out` included), LN weights 1 +- 0.1, biases +- 0.1,
3 masked keys, p_drop 0, upstream gradient N(0, 1); FP8 = second call (delayed scale). rms relative error in % (max relative in the raw rows,
same order of magnitude). upd = the output update y - x.

S1024 N768, d_pair 128:

| tensor | PyTorch bf16 | default 2.5.14 | FP8 do e4m3 | FP8 do e5m2 | FP8 dv+dw only (do e4m3) |
|---|---|---|---|---|---|
| update (fwd) | 2.38 | 2.37 | 2.89 | 2.89 | 2.37 |
| d msa | 0.17 | 0.17 | 0.21 | 0.25 | 0.21 |
| d pair | 0.71 | 0.58 | 3.80 | 5.93 | 3.82 |
| d ln_msa.weight / bias | 0.46 / 0.39 | 0.36 / 0.35 | 2.50 / 2.57 | 4.17 / 4.82 | 2.44 / 2.67 |
| d to_value | 0.43 | 0.28 | 2.75 | 5.36 | 2.79 |
| d ln_pair.weight | 0.81 | 0.59 | 4.14 | 6.97 | 4.23 |
| d to_bias | 0.69 | 0.55 | 3.80 | 5.96 | 3.83 |
| d to_gate | 0.59 | 0.40 | 1.69 | 1.69 | 0.40 |
| d to_out | 0.45 | 0.32 | 1.56 | 1.56 | 0.32 |

The same picture at every case (1024:384, 2048:768, 384:768, 1024:512, 1024:768:256): FP8 e4m3 do 2.1-5.0 % rms on the gradients behind dv
(to_value, ln_msa) and dw (pair, to_bias, ln_pair), 1.6-2.2 % on to_gate / to_out (from the FP8 forward o), d msa 0.21-0.25 %; e5m2 do
4-7 %. Default 0.3-0.7 %. No NaN / inf, no saturation. The error is the per-element rounding of the e4m3 / e5m2 operands (rel. ulp/2 = 2^-4 /
2^-3), which a dot product does not average away (random signs: the error grows with sqrt(K) like the sum) -- so it does not shrink with S or N.
Placement attribution (`proto` per placement, 1024:384): fwd alone -> to_gate / to_out 2.2 %, update 3.1 %; dv alone -> to_value / ln_msa
~5.8 % (e5m2); dw alone -> pair / ln_pair / to_bias ~6 % (e5m2).

### Training dynamics (`train.py`, `res/tr3*.jsonl`, `res/tr2*.jsonl`)

2-layer PWA stack (cuda backend, master weights, 8 x 32, p_drop 0.15) distilled onto a random fp32 PyTorch teacher stack; fresh random msa /
pair per step (same stream for every variant), loss = relative MSE of the update, Adam lr 1e-3, 1000 steps. Mean loss per 200-step window:

| S:N | variant | 1-200 | 201-400 | 401-600 | 601-800 | 801-1000 |
|---|---|---|---|---|---|---|
| 1024:384 (native default) | default (dropout seed 5) | 0.6265 | 0.3260 | 0.2084 | 0.1642 | 0.1568 |
| | default (dropout seed 6) | 0.6268 | 0.3270 | 0.2103 | 0.1660 | 0.1573 |
| | FP8 e5m2 (seed 5) | 0.6269 | 0.3264 | 0.2084 | 0.1656 | 0.1585 |
| | FP8 e5m2 (seed 6) | 0.6272 | 0.3281 | 0.2098 | 0.1665 | 0.1585 |
| | FP8 e4m3 (seed 5) | 0.6269 | 0.3263 | 0.2085 | 0.1656 | 0.1585 |
| | FP8 forward o only (proto) | 0.6269 | 0.3255 | 0.2077 | 0.1657 | 0.1587 |
| | FP8 dv + dw only (proto, e4m3 do) | 0.6265 | 0.3259 | 0.2082 | 0.1643 | 0.1568 |
| 512:384 (plain default) | default | 0.6330 | 0.3322 | 0.2213 | 0.1757 | 0.1618 |
| | FP8 e5m2 | 0.6332 | 0.3304 | 0.2211 | 0.1767 | 0.1637 |
| | FP8 e4m3 | 0.6332 | 0.3304 | 0.2211 | 0.1766 | 0.1637 |

Gap FP8 / default in the last window: +1.1 % (seed 5), +0.8 % (seed 6) at S1024 N384; +1.2 % at S512 N384; the dropout-seed spread of the
default is 0.3 % there (and 0.7 % in window 4). So FP8 trains to a ~1 % higher loss on this task -- small, systematic, the same for e4m3 and
e5m2 do; no divergence, NaN or overflow. The placement runs locate it: **the gap is the forward e4m3 contraction** (o only: +1.2 %); FP8 dv + dw alone end equal to the default (0.1568 vs 0.1568) despite their 2-4 % gradient error. Synthetic task, 1000 steps: not evidence for a real model's training (that needs an end-level run).

### Bit-identity of the default (switch off)

`bit.py` (sha1 of the output and all 10 gradients, training with p_drop 0.15 + inference, separate processes, deterministic parameters):
this branch vs the base tree (1a2bbe42e = rel/3.0.0 = 2.5.14 + inference FP8), 7 cases -- 8 x 32 S1024 N384 / N768 (native), d_pair 256
N768, AF3 64 x 8 N384 (wide), 128 x 16 d_pair 256 (wide): every hash equal (`res/bit_wt3.jsonl` vs `res/bit_base3.jsonl`). The two plain-build
cases (S1024 N512, S384 N768) are equal in output, d msa, d pair and 7 of 8 parameter gradients; d ln_msa.weight / bias differ -- and differ
between two runs of the base tree too (`bit_base3` vs `bit_base4`): the plain `dgv_bwd` build sums them with float atomics (documented, not a
change). The inference FP8 producer `ln_vt8` (`pwa_vt8.cu` without `-DVT_NAT`) compiles from the same preprocessed text as before.

### Tests

`tests/gpu/test_pwa_fp8_train.py` (new): fused producers vs PyTorch quantization (vT8 = transpose of v8, amax, w_quant_t = w_quant^T),
fused vs proto error against fp64 (both formats), route gradients within max(8 %, 6 x default error) of fp64 at 3 shapes (d_pair 128 / 256)
and the default bytes unchanged after the route ran (qualified shape), overflow 0, graph replay == eager == eager (deterministic). Plus the
files importing what changed: `test_pwa_fp8.py`, `test_pwa_fp8_wide.py`, `test_pwa_long_inference.py`, `test_i64_offsets.py::test_pwa_train_pair_past_2_31`;
CPU: `test_capture_api.py` (the new module dicts listed), `test_fp8_switch.py`; ruff check / format, pyright clean on the touched files.
Results (job 57855, final tree): `test_pwa_fp8_train.py` 7 passed; `test_pwa_fp8.py` + `test_pwa_fp8_wide.py` + `test_pwa_long_inference.py` 28 passed;
`test_i64_offsets.py::test_pwa_train_pair_past_2_31` 1 passed; CPU `test_capture_api.py` + `test_fp8_switch.py` 25 passed (job 57856).

### What did not pay / not done (numbers)

- **Wide widths (AF3 8 x 8, 128 x 8 / 16, BIG)**: the three contractions are 18 % (AF3 S1024 N768: 120 + 112 + 90 of 1823 us), 11 %
  (128 x 8 d_pair 256) and 12 % (BIG) of the step; FP8 GEMMs alone are 0.64x -> upper bound 4-7 % before the producers (new `ln_vg_w` / `gate_bwd_w`
  FP8 epilogues in `pwa_wide.cu`, which would re-key the default build or need copies). Not built. A larger, accuracy-free target there: the
  dropout mask is applied by two PyTorch elementwise passes over [S, N, D] in the backward (130 + 66 us = 11 % of the AF3 step; 13 % at 128 x 8;
  BIG also a 596 us `dgv` permute-copy) -- folding it into `gate_bwd_w` like `pwa_glue3` does is a bf16 change, outside this FP8 task.
- **Swapped-role wgmma for the natural v** (second m64n64 MMA, consecutive channels pack into halfwords): 335 us vs 290 for the byte scatter;
  reverted. Launch tuning of the byte-scatter build: 1 x 4 -> 245-259 us (from 310). Still +42 us over `ln_vg`.
- **Unfused quantize / transpose passes** (`IMPL=proto`): 2.1 ms (quantize) + 1.4 ms (byte transpose) per [H, N, S C] tensor at S1024 N768
  against a 0.5 ms GEMM saving -- the producers must write the FP8 layouts.
- **FP8 in the projection GEMMs** (Wv / Wg / Wo dgrad / wgrad, 25-51 GFLOP each): they live inside memory / latency-bound fused kernels at
  15-20 % tensor use; no MMA-bound time to remove (as in the inference study).
- **Forward contraction at N < 512**: the default's fused `pwa_fwd2/3` forward is replaced by split FP8 + `pwa_gateout`; still a win in the
  step (0.91 at S1024 N384) because the backward gains dominate.
- Not measured: fast accumulation (`FOLDJET_ENGINE_PWA_FP8_TRAIN_FASTACC=1` exists, 1-3 % in the inference study), end-level training.

### Recommendation

Speed: 0.85-0.91x of the 2.5.14 step at the 8 x 32 training shapes (cool; 0.88-0.92 sustained), 0.80-0.83 at shapes the default serves with
the plain builds (of which ~0.88-0.90 is the native kernels alone -- a bf16 change worth making separately). Accuracy: gradient error
2-5 % rms (e4m3 do) against 0.3-0.7 % default, ~1 % higher loss on a synthetic distillation run. Keep it opt-in (3.1.0, e4m3 do). The loss gap comes from the forward FP8 contraction, not the backward: a backward-only variant (bf16 forward, FP8 dv / dw) would keep the default loss curve here but needs a producer that writes bf16 v for the forward and e4m3 v natural for dw (`ln_vg` + an e4m3 copy, ~+0.07 ms) and keeps ~2/3 of the GEMM saving -- the next step if the full route's 1 % does not pass an end-level check; ship as
default only after an end-level training run (e.g. Boltz-2 / MiniWorld trunk fine-tune loss curves) shows no gap. The wide (AF3-family)
widths are not worth an FP8 route (<= 4-7 % bound); their dropout passes are the better bf16 target.

### Progress log

- 2026-11-18 00:20 branch `fp8t/pwa` from 1a2bbe42e; 01:05 stage profile + GEMM bench; 01:10 first fused route (0.886 / 0.882 at S1024 N768);
  01:25 swapped-role natural-v store (slower, reverted); 01:33 launch tuning (1 x 4); 01:35 sp2 / training curves / bit-identity; 08:00
  (after an API outage 02:00-05:10) tests fixed (fused-vs-proto against fp64; default bytes checked at a qualified shape), docs.

## Rounds 2 and 3: backward-only variant, baseline = the 3.0.1 default (branch `fp8t3/pwa` = `fp8t2/pwa` + origin/main f5be1df46)

Everything below is measured against the **3.0.1 default** (the branch was merged with origin/main 3.0.1 first; the plain-build shapes are
a little faster there than in 2.5.14: S1024 N512 1.94 ms against 2.14). Scripts / raw rows: `~/miniworld-engine-scratch/fp8t3-pwa/{scripts,res}`;
the step-time benchmark is `tools/fp8_train/pwa_step_bench.py` (+ `run_pwa_step_bench.sbatch`).

### Enable (added)

| env / attribute | effect |
|---|---|
| `FOLDJET_ENGINE_PWA_FP8_TRAIN=bwd` | **backward-only**: bf16 forward (default numerics), FP8 dv / dw (producers `ln_vt8n -DVT_BF` + `pwa_glue3_f8` + `w_quant_t`) |
| `FOLDJET_ENGINE_PWA_FP8_TRAIN_MIN_S` / `pwa_fp8_train.MIN_S` | smallest MSA depth served (default 1024: measured S 1024 .. 4096; below it the default route runs) |

Gates (`serves`): H100, N % 128 == 0, N <= 1024, S >= 1024, S even; `bwd` additionally N >= 512 (its forward needs the split head-major
contraction, N >= 512). Everything else runs the default route (bit-identical, see below).

### What was wrong with the round-2 producer, and the fix

Round 2's `bwd` variant (bf16 v for the forward + e4m3 natural v for dw from one swapped-role wgmma) gave only 0.994 of the default
(S1024 N768, cool): GEMMs saved 151 (dv 447 -> 297) + 153 (dw 388 -> 235) us, but `ln_vt8` (VT_BF) took 443 us against `ln_vg`'s 218.
Micro-benchmark of the producer alone (`mb.py`, S1024 N768, us): base 460-470; **no smem staging stores 468** (the bank conflicts of the
scatter are NOT the cost); **no v TMA stores 225** (the global stores are); swizzled staging (TMA 64B / 32B swizzle) 468 (no change).
The stores are 64 B (bf16) / 32 B (e4m3) row pieces at a stride of S*C bytes, and items were ordered n-block fastest, so the 132 concurrent CTAs
wrote 12 distant rows. Ordering the items **s fastest** (`VT_MAP=1`: concurrent CTAs write adjacent pieces of the same rows, which the L2
merges) takes the producer to **335 us** (swizzled staging 335, plain 337; cfg (1,4) best), bytes identical. Applying the same order to the
FP8 forward's `ln_vt8n` made it slower (the full route's step 0.904 against 0.887): left off there. The same order in `pwa_glue3_f8`
(`G3_MAP`): 662 against 659 us, no effect, off. 335 us against the memory-bound ideal ~290 (806 MB of v at ~2.8 TB/s): ~45 us left.

### Step time, fwd + bwd of one module call, against the 3.0.1 default (`bench_final.jsonl`, job 58927)

One process; `autograd.grad` of the output w.r.t. msa, pair and every parameter, master weights, p_drop 0.15, each variant one CUDA graph,
def / ctl (same-cubin control) / variants interleaved; cool = 2 s idle + short burst, median of 3; sustained = after 1 s of back-to-back replays,
mean of 3. Cells = variant / default, `cool / sustained` (ctl = control, 1.00 means the measurement is clean; 1.019 once at 2048:384).

| S:N[:d_pair] | default ms (cool) | ctl cool / sus | `bwd` (bf16 fwd, FP8 dv + dw) | full (`1`, FP8 o + dv + dw, do e4m3) |
|---|---|---|---|---|
| 1024:768 | 3.256 | 0.993 / 1.003 | **0.954 / 0.964** | 0.886 / 0.899 |
| 2048:768 | 6.075 | 0.993 / 0.997 | **0.951 / 0.958** | 0.881 / 0.892 |
| 4096:768 | 11.732 | 0.993 / 1.002 | **0.958 / 0.968** | 0.889 / 0.914 |
| 1024:768:256 (Chai-1) | 3.467 | 0.996 / 1.006 | **0.956 / 0.971** | 0.894 / 0.911 |
| 1024:512 | 1.942 | 0.998 / 1.005 | **0.972 / 0.982** | 0.914 / 0.925 |
| 1024:384 | 1.398 | 0.992 / 0.998 | not served (N < 512): default | 0.940 / 0.949 |
| 2048:384 | 2.620 | 1.019 / 1.004 | not served (N < 512): default | 0.922 / 0.927 |

Per stage at S1024 N768 (profiler, us; default -> `bwd`): `ln_vg` 219 -> `ln_vt8` (VT_BF, mapped) 329; contraction dv 447 -> 297; dw 388 -> 235;
`pwa_glue3` 612 -> `pwa_glue3_f8` 658; `w_quant_t` +16; `scale_update` 2 x 3. Gain 304 us of GEMM, cost 110 + 46 + 16 + 6 = 178 us of producers
(net ~126 us = 3.9 %; the measured 4.6 % cool also includes the smaller v bytes the dw GEMM reads).

### Gradient error against fp64 (`err3.jsonl`, rms relative error in %, same protocol as above; default / PyTorch / `bwd` / full e4m3)

| S:N | tensor | PyTorch bf16 | default 3.0.1 | `bwd` | full |
|---|---|---|---|---|---|
| 1024:768 | update (fwd) | 2.38 | 2.37 | 2.37 | 2.89 |
| | d msa | 0.17 | 0.17 | 0.21 | 0.21 |
| | d pair | 0.71 | 0.58 | 3.80 | 3.80 |
| | d to_value | 0.43 | 0.28 | 2.75 | 2.75 |
| | d ln_msa.weight | 0.46 | 0.36 | 2.30 | 2.50 |
| | d to_gate / to_out | 0.59 / 0.45 | 0.40 / 0.32 | 0.40 / 0.32 | 1.69 / 1.56 |
| 2048:768 | d pair / d to_value / d ln_msa.w | 0.71 / 0.45 / 0.62 | 0.58 / 0.31 / 0.44 | 3.79 / 3.01 / 2.41 | 3.79 / 3.01 / 2.66 |
| 1024:512 | d pair / d to_value / d ln_msa.w | 0.71 / 0.44 / 0.52 | 0.58 / 0.29 / 0.39 | 3.79 / 2.71 / 2.78 | 3.79 / 2.71 / 3.22 |

The `bwd` variant's forward (update, to_gate, to_out) equals the default's; the dv / dw gradients (pair, to_bias, ln_pair, to_value, ln_msa) carry
the e4m3 rounding (2.3-4.1 % rms, max 2-3 %), 4-8x the default's error; the full route adds the forward contraction (update 2.89, to_gate / to_out 1.6-2.0 %).

### Training curve (`tr3.jsonl`, `train.py`: 2-layer PWA stack distilled to an fp32 PyTorch teacher, S1024 N512, p_drop 0.15, Adam 1e-3, 1000 steps, mean loss per 200 steps)

| variant | 1-200 | 201-400 | 401-600 | 601-800 | 801-1000 |
|---|---|---|---|---|---|
| default | 0.6270 | 0.3329 | 0.2364 | 0.1950 | 0.1666 |
| `bwd` (FP8 dv + dw) | 0.6271 | 0.3333 | 0.2365 | 0.1952 | 0.1668 |
| full (FP8 o + dv + dw, e4m3) | 0.6273 | 0.3326 | 0.2374 | 0.1955 | 0.1671 |

Final window: `bwd` +0.1 %, full +0.3 % (one dropout seed; the default's own seed spread is ~0.3 %); no divergence, overflow 0. (The 3.0.0-line runs
above saw +1 % for the full route at S1024 N384; same direction, `bwd` indistinguishable from the default in both.) Synthetic task: not an end-level check.

### Default route bit-identical with the switch off

`bit.py` (sha1 of output + all gradients, training p_drop 0.15 + inference, 7 cases: native, d_pair 256, plain builds, wide): this branch against a
detached origin/main 3.0.1 worktree: `cmp` of the two result files equal (`bit_wt5` / `bit_base5`). Tests (job 58922): `test_pwa_fp8_train.py`
+ `test_pwa_fp8.py` + `test_pwa_fp8_wide.py` + `test_i64_offsets.py::test_pwa_train_pair_past_2_31` + CPU `test_capture_api.py`, `test_fp8_switch.py`: 57 passed;
ruff format / check, pyright clean.

### Verdict

- `bwd`: wins 2.8-4.9 % (cool; 1.8-4.2 % sustained) at S >= 1024, N 512-1024, d_pair 128 / 256, forward numerics unchanged, training curve = default;
  gradient error 2.3-4 % rms on the dv / dw side. Gated off for N < 512 (no win there: 0.99-1.00) and S < 1024.
- full: wins 6-12 % at every gated shape incl. N 384; forward error and +0.3-1 % loss.
- Remaining producer headroom: `ln_vt8` (VT_BF) 335 us vs ~290 ideal, `pwa_glue3_f8` +46 us over `pwa_glue3` (its do / doT stores are 64 B pieces,
  tile order did not help); worth ~1.5 % more at most. Not a default: opt-in, off. FP8 projection GEMMs, wide widths: unchanged no-go (above).

### Progress log (rounds 2-3)
- 2026-11-18 18:33 branch fp8t3/pwa = fp8t2/pwa + 3.0.1; 18:36 baseline: `bwd` 0.994 of default (producer 443 us); 18:52 producer micro-benchmark:
  smem scatter / swizzle irrelevant, stores dominate; 23:47 s-fastest item order 460 -> 335 us; 00:10 gradient error, training curve; 00:22 tests, bit-identity;
  00:23 `bench_final` (step table above); S gate (MIN_S 1024) added after.

## Round 4: saved-activation bytes, memory, finer scales (branch `fp8t4/pwa` = `fp8t3/pwa` 1269b85c6 + round 4)

Scripts / raw rows: `~/miniworld-engine-scratch/fp8t4-pwa/{scripts,res}` (`mem.py`, `errstudy.py`, `speed.py`, `err.py`, `train.py`). Baseline = the 3.0.1 default.

### 1. Saved activation bytes and peak memory (`mem.py`; eager fwd + bwd, MiB above the inputs + parameters; `saved` = allocated after the forward minus before, output excluded)

Saved by the module for the backward (S1024 N768, default): w 9 + v 384 + o 384 + y 96 = 873 MiB; the inputs (m 96, z 144) are the caller's. v and o are each S x N x 256 x 2 B and are 88 % of it.

| S:N[:d_pair] | default saved / peak | `bwd` saved / peak | `bwd` + O8 saved / peak | peak vs default (`bwd` / `bwd`+O8) |
|---|---|---|---|---|
| 1024:768 | 873 / 2776 | 681 / **2318** | 489 / **2130** | -16.5 % / -23.3 % |
| 4096:768 | 3465 / 10552 | 2697 / 9230 | 1929 / 8466 | -12.5 % / -19.8 % |
| 1024:512 | 580 / 1813 | 452 / 1542 | 324 / 1421 | -14.9 % / -21.6 % |
| 1024:384, 2048:384 (`bwd` not served; full route) | 434 / 1343, 866 / 2639 | full: 338 / 1247, 674 / 2447 | | -7 % |
| 1024:1024, 1024:768:256 | 1168 / 3785, 873 / 2924 | 912 / 3529, 681 / 2732 (before the frees below) | | |

What moved the peak: (a) FP8 v (round 1-3): saved -192 MiB (-22 %), peak -7 %; (b) **new, free**: the FP8 backward drops `doT8` right after the
dv GEMM and `do8` right after the dw GEMM (they were held until the function returned): peak -266 MiB (2584 -> 2318 at S1024 N768), no time
cost; (c) **O8** (below): saved o as e4m3, a further -192 MiB saved and -190 MiB peak. DRAM bytes per step (ncu, round 2 on the 2.5.14 base, S1024 N768,
read / write MB): default 4425 / 2376; `bwd` 3859 / 2579; full 3662 / 2375 -- the bytes drop only by the half-size v re-read (dw) and
the smaller dv / dw operand reads; the write side does not shrink (v bf16 + e4m3 are both written).

**Largest S that fits one H100 80 GB** (training step, `mem.py` OOM sweep, step 2048; N > 1024 is not served by the cuda training backend
(`BackendUnsupported`: pair backward holds N <= 1024), so there is no N sweep):

| N | default | `bwd` (frees + FP8 v) | `bwd` + O8 |
|---|---|---|---|
| 768 | 26624 (28672 OOM) | 30720 (32768 OOM) = +15 % | 32768 (34816 OOM) = +23 % |
| 1024 | 20480 (22528 OOM) | 22528 (24576 OOM) = +10 % | 24576 (26624 OOM) = +20 % |

### 2. FP8 storage of o (`O8`, `FOLDJET_ENGINE_PWA_FP8_TRAIN_O8=1`, needs `bwd` or `1`; `csrc/pwa_o8.cu`)

o ([H, N, S*C], head-major) is quantised to e4m3 (delayed per-tensor pow2 scale, amax recorded by the kernel, same site machinery as v / do)
right before `save_for_backward`, and dequantised to bf16 in the backward just before `pwa_glue3_f8` (freed right after it). **A memory mode, not a speed
mode**: two memory-bound passes (403 MB read + 201 MB write each) cost ~420 us at S1024 N768 (step 1.07-1.11x of the default, i.e. +12-16 points over `bwd`).
Fusing the quantisation into `pwa_gateout` (which reads o anyway) and the dequantisation into `pwa_glue3_f8` (o arrives by TMA into swizzled smem
tiles that the consumers read as bf16) would remove most of it but means new copies of two persistent TMA kernels; not done (default `pwa_gateout`
cannot be re-keyed). Error (below) and curve: gate / out gradients go from 0.4 / 0.3 % to 2.6 / 2.5 % rms (same level as the other FP8 gradients).

### 3. Finer-grained scales: no-go (`errstudy.py`, `res/errstudy.json`)

Quantise-dequantise emulation, exact fp64 products, w = softmax(N(0, 1.5)), v with outlier channels (every 97th x8), do with lognormal row scales; rms relative error of the
contraction (S*C = 4096, N = 768, 8 heads), e4m3 both operands:

| contraction | per-tensor scale | per-row scale | per-128-block scale (not one GEMM) |
|---|---|---|---|
| dw = do v^T | 3.748 % | 3.748 % | 3.748 % |
| dv = w^T do | 3.713 % | 3.713 % | 3.713 % |
| o = w v | 3.730 % | 3.730 % | 3.730 % |

The error is the **mantissa rounding**, independent of the scale grain: one e4m3 rounding of a Gaussian vector is 2.65 % relative rms, two operands give sqrt(2) x 2.65 = 3.75 %.
An exact v alone: 2.65 % (the rounding of do remains); v as e4m3 hi + lo (2 GEMMs): 2.65 %; both operands hi + lo (4 GEMMs): 0.09 %. DeepSeek-V3's fine-grained
scales fix outliers / dynamic range and keep the loss within 0.25 %; they do not change the 3 mantissa bits, and the module's own error (3.8 % on d pair) is that floor.
Reaching <= 1.5 % rms needs >= 3 FP8 GEMMs per contraction (hi*hi + hi*lo + lo*hi) = more than the bf16 GEMM time saved. Per-row scales are also
more traffic (scale vectors) for no gain. No-go; the delayed per-tensor scale stays.

### 4. The remaining producer gaps

- `ln_vt8` (VT_BF) 335 us: bytes x 100 + y 100 + v bf16 403 + v e4m3 201 = 804 MB = 2.4 TB/s, against `ln_vg`'s 603 MB in 218 us = 2.77 TB/s: ~290 us at the same rate, i.e. ~45 us (1.4 % of the step) left. The stores are 64 B / 32 B row pieces (item = 64 tokens x one s); an item of 2 s would double the pieces (128 B / 64 B) but the smem staging then grows to 96 KiB: not tried, <= 1 % of the step.
- `pwa_glue3_f8` +46 us over `pwa_glue3`: the byte scatter of do / doT into smem has 4-way bank conflicts; 64B-swizzled staging tiles (TMA SWIZZLE_64B, `FOLDJET_ENGINE_PWA_FP8_TRAIN_G3SWZ=1`)
  gave 664 us against 659: not the cost (the kernel is latency-bound on its o / dout stream); left off. Tile order (round 3): no effect.

### 5. Step time against the 3.0.1 default (`bench_final4.jsonl`, `tools/fp8_train/pwa_step_bench.py`; same-process interleaved CUDA graphs, ctl = same-cubin control)

Cells = variant / default, `cool / sustained`; round 3 in brackets for `bwd`.

| S:N[:d_pair] | default ms (cool) | ctl cool / sus | `bwd` (round 4 [round 3]) | `bwd` + O8 | full (`1`) |
|---|---|---|---|---|---|
| 1024:768 | 3.231 | 0.991 / 1.003 | **0.951 / 0.962** [0.954 / 0.964] | 1.081 / 1.088 | 0.890 / 0.897 |
| 2048:768 | 6.059 | 0.990 / 1.002 | **0.947 / 0.961** [0.951 / 0.958] | 1.083 / 1.098 | 0.875 / 0.889 |
| 4096:768 | 11.668 | 0.992 / 0.993 | **0.949 / 0.959** [0.958 / 0.968] | 1.093 / 1.106 | 0.878 / 0.895 |
| 1024:768:256 | 3.455 | 0.990 / 0.996 | **0.952 / 0.962** [0.956 / 0.971] | 1.071 / 1.081 | 0.888 / 0.902 |
| 1024:512 | 1.950 | 0.992 / 0.997 | **0.962 / 0.978** [0.972 / 0.982] | 1.106 / 1.117 | 0.906 / 0.912 |
| 1024:384 | 1.419 | 0.998 / 0.995 | default (N < 512) | default | 0.927 / 0.930 |
| 2048:384 | 2.716 | 0.990 / 0.995 | default (N < 512) | default | 0.900 / 0.902 |

`bwd` is 0.4-1 point better than round 3 (fewer live buffers); full unchanged.

### 6. Accuracy and curve of O8 (`err5.jsonl`, `tr5.jsonl`)

rms % against fp64 (S1024 N768): `bwd`+O8: update 2.37, d msa 0.22, d pair 3.80, d to_value 2.75, d ln_msa.w 2.71, d to_gate 2.63, d to_out 2.50 (the `bwd` route without O8: 0.40 / 0.32 on the last two;
full + O8: update 2.89, to_gate 3.12, to_out 2.90). 2048:768 and 1024:512 the same (to_gate 2.7, to_out 2.5-2.6).
Distillation curve (S1024 N512, 1000 steps, window means): default 0.6270 / 0.3329 / 0.2364 / 0.1950 / 0.1666; `bwd` + O8 0.6271 / 0.3361 / 0.2379 / 0.1965 / 0.1678 (+0.7 % in the last window,
`bwd` alone +0.1 %, full +0.3 %; seed spread ~0.3 %).

### 7. Verdict (round 4)

- Ship (opt-in) the `bwd` route as is, now with the free peak-memory gain (-12 .. -17 % peak, +10-15 % max S) at 0.947-0.962 of the default step.
- O8 = a memory mode (peak -20 .. -23 %, max S +20-23 %) at +8-11 % step time over the default and +0.7 % loss; useful only when S is the limit.
- Finer scales: no-go (rounding floor); <= 1.5 % rms is not reachable with e4m3 x e4m3 in one GEMM.
- Producers: `ln_vt8` is ~45 us (1.4 %) above its memory rate, `pwa_glue3_f8` latency-bound (swizzle tried): no further cheap gain.
- Default route bit-identical with the switch off: `bit.py` sha1 of all 7 cases equal to the 3.0.1 worktree (`bit_wt6` vs `bit_base5`).
