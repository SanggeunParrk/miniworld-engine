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

## Token DiT round 3 (branch `fp8t3/tokendit`): against the 3.0.1 default

Base: `fp8t2/tokendit` (round 2: `nox` / `hsave` byte reductions, behind `MINIWORLD_TOKEN_DIT_FP8_TRAIN`) merged with origin/main 3.0.1 (shared
conditioning on L rows, pair bias on the APB kernels). The baseline is now the 3.0.1 default (A48 L384 3.03 ms; it was 3.43 ms on 2.5.14).
Training placements live in `token_dit_fp8_train.py`, separate from the inference gate `token_dit_fp8.active()`.
Scripts: `tools/fp8_train_bench/` (`speed.py`, `run.sh`); raw rows `~/miniworld-engine-scratch/fp8t3-tokendit/res/`.

Placements: `nox` (no fp32 residual copy: the backward reads the block input) and `hsave` (the forward keeps h; swiglu_bwd does not rewrite it)
are byte reductions, bit-identical (no FP8; the fp64 error table below shows identical numbers); `expand` / `squeeze` are the e4m3 forward GEMMs.
`RECOMMENDED` (= `MINIWORLD_FP8_TRAIN=1`, preset `recommended`) is now `expand + nox + hsave`; `squeeze` is dropped (see below). Gate: the byte reductions apply at
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
  stays opt-in (`MINIWORLD_TOKEN_DIT_FP8_TRAIN=expand`, gate 16384) until an end-level training curve exists. Candidates for the main line: `nox` + `hsave` (bit-identical, 1.0-1.5 %) as default-route changes with a
  `backend_versions` bump (cuda only) — a decision for the release owner, not done here.
- Does not pay (unchanged): dgrad placements, attention-side projections, wgrad, attention core.

## Token DiT round 4 (branch `fp8t4/tokendit`): bytes, finer scales, fused cast, real end-level check

Base `fp8t3/tokendit` (71d7d86d4). Scripts `~/miniworld-engine-scratch/fp8t4-tokendit/{scripts,res,ft}`; same-process CUDA-graph timing as round 3
(`tools/fp8_train_bench/`), baseline = the 3.0.1 default. H100 node01 / node02, 2026-11-19.

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
the 2 u write it saves; bytes 0.978 / 0.989 -> 0.983 / 0.992 at M 18432). Memory option only (`MINIWORLD_TOKEN_DIT_FP8_TRAIN=...,rx1`), not in `recommended`.

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
unfused 0.973 / 0.983 -> fused 0.971 / 0.980 at M 18432 (0.981 / 0.987 -> 0.980 / 0.985 at 36864): -0.2 / -0.3 %, as predicted. `MINIWORLD_TOKEN_DIT_FP8_TRAIN_FUSE_Q8=0` restores the pass.

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
| C `af3_default` (fp32 diffusion head), 6 steps | 0.0775 (A over the same 6: 0.0817) | -5.1 % |

Bit-identical byte reductions leave the loss unchanged by construction. The FP8 `expand` in 24 DiT blocks adds a systematic +2 % to the diffusion loss at fixed weights
(about a third of the penalty of running the diffusion head in bf16 instead of fp32 at all, which the production recipe forbids). The synthetic distillation (<= 0.02 %)
did not predict this: the FoldBench lesson again. Round-trip: the first run (job 59009) gave NaN from step 1400: a real weight output channel with amax ~ 1e-37 made
the per-channel scale 1e-39 and 1 / scale = inf; every quantizer now floors the scale at 1e-30 (`test_tiny_and_zero_rows_stay_finite`), the unit tests and the random-weight runs had never seen such a channel.

### Verdict (round 4)

- FP8 `expand` for training: NO-GO as a recommendation. The step gain is 0.7-0.9 % (inside the 1 % control spread), the gradient error of the transition 4-5 %, and the real
  fine-tune shows +2 % diffusion loss at fixed weights. Kept opt-in (`MINIWORLD_TOKEN_DIT_FP8_TRAIN=expand`, gate 16384, default off) for the record; `RECOMMENDED` stays `expand + nox + hsave` only as the opt-in preset of
  the switch `MINIWORLD_FP8_TRAIN=1` (a release owner should drop `expand` from it: see the table, a one-line change).
- The bit-identical byte reductions `nox` + `hsave` are the whole win: 0.9-1.1 % sustained (1.7-2.2 % cool) at M >= 18432, -0.2 .. -1.1 % below, peak -3 %, largest A at L 384 1536 -> 1792 (+17 %), output and 27 gradients
  bit-identical to 3.0.1. Release candidate: branch `fp8t4/tokendit-bytes-default` (see the report: diff, `TokenDiTProtenix.backend_versions["cuda:h100_sm90"]` 5 -> 6, 292 accuracy + 292 speed cells owed on h100_sm90).
- Finer scales, 8-bit saved activations (ab), `rx1`: no-go (above). Fused q8: kept (-0.2 / -0.3 %, bit-identical).
