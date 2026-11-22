# E5: other public kernels (all operator classes, all candidates)

Source: `E5_<card>.csv` (collect.py) from the existing records only: the first E5 run (`ext`: TriAttn + ds4sci / trifast), the third-party-trimul run (`ext_trimul`), the third-party-attn run (`ext_attn`) and a few small runs of this integration (`e5`: cueq for TriMul on every card, all TriMul backends on the 5090 at 20 iterations). AF3 setup, L = 512 / 1024 / 2048, eval and train, 3 repeats (median of the repeats' medians), 20 timed iterations (warmup 5, L2 flushed, CUDA-graph timing) unless the `iterations` column of the CSV says otherwise. Median ms per call of the whole module (LayerNorm, projections, gate, out-projection and residual are the module's own; only the core differs for the attention libraries, the whole update for the TriMul libraries). Cells: `ms (xR)` with R = candidate / engine cuda (R < 1: faster than the engine). `e` after a number = eager timing (CUDA events) of a cell whose graph timing failed the CUDA-graph validity check (section 0), compared with the engine's eager time. Codes of cells without a number: `OOM` out of memory on this card (skipped; the case is not re-routed); `FAIL` the library failed on this card (error recorded); `NOBWD` no backward kernel: training not served (forward-only or inference-only library); `HW` hardware / software not supported on this card (see the footnotes); `SHAPE` the library (or, for the engine, its kernel table) does not serve this shape / dtype / layout.

## 0. CUDA-graph validity of every candidate (graph_check.py)

The measurer times `graph.replay()` of a step captured on a side stream. A library that launches its kernels on the legacy default stream (DeepSpeed `evoformer_attn`: attention_cu.cu / attention_back.cu launch without a stream argument) leaves the capture: the replay then times an empty graph. `graph_check.py` runs each (backend, case) with the same step and checks (a) two replays give the eager output, (b) train: parameter gradients non-zero and of the eager norm, (c) replay time not below half of the eager GPU kernel time, (d) the largest eager kernels appear in the profile of the replay. A cell that fails is `invalid_graph` in the CSV (excluded from the tables' graph columns); its corrected eager time (CUDA events, same warmup / iterations / L2 flush) is shown with an `e` and compared with the engine's eager time. `pytorch` is not checked (plain torch ops).

| backend | H100 (sm_90) | A100 (sm_80) | RTX 5090 (sm_120) | B200 (sm_100) |
|---|---|---|---|---|
| `cuda` | valid (47 cells) | valid (52 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (46 cells); 6 ok cells: check could not run (out of memory / crash / library error) | valid (48 cells) |
| `cueq` | valid (12 cells) | valid (12 cells) | valid (9 cells); 3 ok cells: check could not run (out of memory / crash / library error) | valid (30 cells) |
| `ds4sci` | INVALID 6 of 6 cells ((a) graph output differs from eager: relative rms 1.27e-01); valid 0 | INVALID 6 of 6 cells ((a) graph output differs from eager: relative rms 1.27e-01); valid 0 | no ok cell (unsupported / failed / skipped) | INVALID 6 of 6 cells ((a) graph output differs from eager: relative rms 1.04e-01); valid 0 |
| `trifast` | valid (6 cells) | valid (6 cells) | valid (5 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (6 cells) |
| `fast_trimul` | valid (6 cells) | valid (6 cells) | valid (4 cells); 2 ok cells: check could not run (out of memory / crash / library error) | valid (6 cells) |
| `gpumode` | valid (3 cells) | valid (3 cells) | valid (2 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (3 cells) |
| `geonn` | valid (3 cells) | valid (3 cells) | valid (2 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (3 cells) |
| `gnops` | valid (24 cells) | valid (24 cells) | valid (23 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (24 cells) |
| `of3tri` | valid (24 cells) | valid (24 cells) | valid (23 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (24 cells) |
| `megafold` | valid (24 cells) | valid (24 cells) | valid (22 cells); 2 ok cells: check could not run (out of memory / crash / library error) | valid (24 cells) |
| `bionemo_ir` | valid (3 cells) | valid (3 cells) | valid (8 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (15 cells) |
| `turbopf` | valid (5 cells); 1 ok cells: check could not run (out of memory / crash / library error) | no ok cell (unsupported / failed / skipped) | no ok cell (unsupported / failed / skipped) | no ok cell (unsupported / failed / skipped) |
| `ligo_pwa` | valid (6 cells) | valid (6 cells) | valid (5 cells); 1 ok cells: check could not run (out of memory / crash / library error) | valid (6 cells) |
| `megafold_tr` | valid (12 cells) | valid (12 cells) | valid (10 cells); 2 ok cells: check could not run (out of memory / crash / library error) | valid (12 cells) |

## 1. Cells where a candidate is faster than the engine

Engine = the engine's `cuda` backend. Faster: candidate <= 0.97 x engine; equal: within +-3 %. Every class on every card was checked (all `ok` cells with an engine `cuda` number).

No cell. (An earlier reading that `ds4sci` beats the engine at TriAttn train L = 1024 / 2048 and eval L = 2048 on H100 and A100 is withdrawn: those graph timings were of an empty graph, see section 0; the eager timings of `ds4sci` are slower than the engine in every TriAttn cell.)

Cells where the engine's `cuda` backend has no number (no kernel for the shape: any candidate that runs is ahead by default):

| card | class | mode | L | engine cuda | best candidate | its ms |
|---|---|---|---|---|---|---|
| a100 | PWA | train | 2048 | unsupported | `ligo_pwa` | 2578.6 |
| b200 | AttentionPairBias-768 | eval | 512 | unsupported | `megafold` | 0.514 |
| b200 | AttentionPairBias-768 | eval | 1024 | unsupported | `of3tri` | 1.785 |
| b200 | AttentionPairBias-768 | eval | 2048 | unsupported | `of3tri` | 6.946 |
| b200 | AttentionPairBias-768 | train | 512 | unsupported | `megafold` | 1.149 |
| b200 | AttentionPairBias-768 | train | 1024 | unsupported | `of3tri` | 3.759 |
| b200 | AttentionPairBias-768 | train | 2048 | unsupported | `of3tri` | 14.348 |
| h100 | AttentionPairBias-768 | eval | 512 | unsupported | `megafold` | 0.712 |
| h100 | AttentionPairBias-768 | eval | 1024 | unsupported | `of3tri` | 2.548 |
| h100 | AttentionPairBias-768 | eval | 2048 | unsupported | `of3tri` | 10.228 |
| h100 | AttentionPairBias-768 | train | 512 | unsupported | `megafold` | 1.507 |
| h100 | AttentionPairBias-768 | train | 1024 | unsupported | `of3tri` | 5.196 |
| h100 | AttentionPairBias-768 | train | 2048 | unsupported | `of3tri` | 20.040 |
| h100 | PWA | train | 2048 | unsupported | `ligo_pwa` | 1114.8 |
| rtx5090 | TriAttn | train | 2048 | skipped | `cueq` | 244.6 |
| rtx5090 | PWA | train | 2048 | unsupported | `ligo_pwa` | 579.1 |

## 2. Engine against the best candidate (per class, per card)

Best candidate = the fastest `ok` backend other than `pytorch` and the engine's `cuda` (cuEquivariance counts). Ratio = best candidate / engine (> 1: the engine is ahead).

| card | class | eval 512 | eval 1024 | eval 2048 | train 512 | train 1024 | train 2048 |
|---|---|---|---|---|---|---|---|
| h100 | TriAttn | `of3tri` x4.03 | `of3tri` x3.85 | `turbopf` x3.38 | `turbopf` x3.27 | `of3tri` x2.93 | `of3tri` x2.74 |
| h100 | TriMul | `cueq` x2.06 | `cueq` x2.04 | `cueq` x1.59 | `cueq` x2.49 | `cueq` x2.34 | `cueq` x2.00 |
| h100 | AttentionPairBias-384 | `megafold` x10.65 | `of3tri` x16.57 | `of3tri` x20.71 | `of3tri` x8.06 | `of3tri` x10.65 | `of3tri` x12.01 |
| h100 | AttentionPairBias-768 | no engine number; `megafold` 0.712 | no engine number; `of3tri` 2.548 | no engine number; `of3tri` 10.228 | no engine number; `megafold` 1.507 | no engine number; `of3tri` 5.196 | no engine number; `of3tri` 20.040 |
| h100 | OPM | - | - | - | - | - | - |
| h100 | PWA | `bionemo_ir` x11.50 | `bionemo_ir` x12.08 | `bionemo_ir` x11.45 | `ligo_pwa` x36.31 | `ligo_pwa` x52.58 | no engine number; `ligo_pwa` 1114.8 |
| h100 | Transition-d128 | `megafold_tr` x5.09 | `megafold_tr` x5.66 | `megafold_tr` x5.38 | `megafold_tr` x3.49 | `megafold_tr` x3.46 | `megafold_tr` x3.22 |
| h100 | Transition-d64 | `megafold_tr` x7.60 | `megafold_tr` x7.71 | `megafold_tr` x7.33 | `megafold_tr` x3.72 | `megafold_tr` x3.73 | `megafold_tr` x3.79 |
| a100 | TriAttn | `gnops` x3.15 | `of3tri` x2.81 | `trifast` x2.80 | `trifast` x2.34 | `trifast` x2.16 | `trifast` x2.08 |
| a100 | TriMul | `cueq` x1.63 | `cueq` x1.43 | `cueq` x1.29 | `cueq` x1.93 | `cueq` x1.71 | `cueq` x1.68 |
| a100 | AttentionPairBias-384 | `of3tri` x10.98 | `of3tri` x14.99 | `of3tri` x17.10 | `of3tri` x7.32 | `of3tri` x9.38 | `of3tri` x10.19 |
| a100 | AttentionPairBias-768 | `of3tri` x6.63 | `of3tri` x9.41 | `of3tri` x10.77 | `megafold` x3.20 | `megafold` x4.08 | `megafold` x4.40 |
| a100 | OPM | - | - | - | - | - | - |
| a100 | PWA | `bionemo_ir` x10.23 | `bionemo_ir` x9.40 | `bionemo_ir` x7.85 | `ligo_pwa` x47.68 | `ligo_pwa` x62.09 | no engine number; `ligo_pwa` 2578.6 |
| a100 | Transition-d128 | `megafold_tr` x4.33 | `megafold_tr` x4.40 | `megafold_tr` x4.14 | `megafold_tr` x2.39 | `megafold_tr` x2.34 | `megafold_tr` x2.34 |
| a100 | Transition-d64 | `megafold_tr` x6.23 | `megafold_tr` x6.68 | `megafold_tr` x7.00 | `megafold_tr` x2.91 | `megafold_tr` x3.21 | `megafold_tr` x3.23 |
| rtx5090 | TriAttn | `cueq` x1.53 | `cueq` x1.38 | `cueq` x1.27 | `cueq` x1.52 | `cueq` x1.45 | no engine number; `cueq` 244.6 |
| rtx5090 | TriMul | `cueq` x1.61 | `cueq` x1.49 | `cueq` x1.36 | `cueq` x1.69 | `cueq` x1.39 | `cueq` x1.33 |
| rtx5090 | AttentionPairBias-384 | `of3tri` x8.03 | `megafold` x10.01 | `megafold` x11.31 | `megafold` x6.88 | `of3tri` x7.31 | `of3tri` x7.95 |
| rtx5090 | AttentionPairBias-768 | `of3tri` x6.72 | `of3tri` x8.93 | `gnops` x11.26 | `megafold` x3.08 | `megafold` x4.97 | `megafold` x5.86 |
| rtx5090 | OPM | `bionemo_ir` x1.48 | `bionemo_ir` x1.22 | `bionemo_ir` x1.14 | - | - | - |
| rtx5090 | PWA | `bionemo_ir` x7.32 | `bionemo_ir` x6.27 | `bionemo_ir` x5.60 | `ligo_pwa` x34.13 | `ligo_pwa` x41.35 | no engine number; `ligo_pwa` 579.1 |
| rtx5090 | Transition-d128 | `megafold_tr` x3.37 | `megafold_tr` x4.07 | `megafold_tr` x4.31 | `megafold_tr` x2.30 | `megafold_tr` x2.35 | `megafold_tr` x2.35 |
| rtx5090 | Transition-d64 | `megafold_tr` x6.12 | `megafold_tr` x6.96 | `megafold_tr` x7.56 | `megafold_tr` x2.97 | `megafold_tr` x2.97 | `megafold_tr` x3.16 |
| b200 | TriAttn | `bionemo_ir` x2.96 | `bionemo_ir` x2.57 | `bionemo_ir` x1.82 | `trifast` x4.42 | `trifast` x4.12 | `trifast` x3.70 |
| b200 | TriMul | `cueq` x2.35 | `cueq` x2.22 | `cueq` x1.91 | `cueq` x4.11 | `cueq` x3.78 | `cueq` x2.93 |
| b200 | AttentionPairBias-384 | `megafold` x12.50 | `of3tri` x21.12 | `of3tri` x27.95 | `megafold` x8.63 | `of3tri` x11.14 | `of3tri` x11.95 |
| b200 | AttentionPairBias-768 | no engine number; `megafold` 0.514 | no engine number; `of3tri` 1.785 | no engine number; `of3tri` 6.946 | no engine number; `megafold` 1.149 | no engine number; `of3tri` 3.759 | no engine number; `of3tri` 14.348 |
| b200 | OPM | `bionemo_ir` x2.80 | `bionemo_ir` x1.88 | `bionemo_ir` x1.25 | - | - | - |
| b200 | PWA | `bionemo_ir` x15.30 | `bionemo_ir` x16.45 | `bionemo_ir` x16.57 | `ligo_pwa` x72.72 | `ligo_pwa` x87.55 | `ligo_pwa` x139.55 |
| b200 | Transition-d128 | `bionemo_ir` x5.85 | `bionemo_ir` x6.36 | `bionemo_ir` x6.14 | `megafold_tr` x3.91 | `megafold_tr` x3.82 | `megafold_tr` x3.38 |
| b200 | Transition-d64 | `bionemo_ir` x7.57 | `bionemo_ir` x9.13 | `bionemo_ir` x10.14 | `megafold_tr` x5.86 | `megafold_tr` x6.15 | `megafold_tr` x6.25 |

## 3. Tables per operator class and card

### H100 (sm_90)

**TriAttn (TriangleAttention, starting, d_pair 128, 4 heads of 32)**

| mode | L | `pytorch` | `cuda` | `cueq` | `ds4sci` | `trifast` | `gnops` | `of3tri` | `megafold` | `bionemo_ir` | `turbopf` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | 512 | 8.012 | 0.435 | 2.283 (x5.25) | 3.003e (x6.77) INVALID-GRAPH | 2.069 (x4.76) | 1.762 (x4.06) | 1.750 (x4.03) | 2.624 (x6.04) | HW | 2.029 (x4.67) |
| eval | 1024 | 56.041 | 2.389 | 11.950 (x5.00) | 18.638e (x7.73) INVALID-GRAPH | 9.978 (x4.18) | 9.281 (x3.88) | 9.189 (x3.85) | 14.346 (x6.01) | HW | 9.649 (x4.04) |
| eval | 2048 | OOM | 16.487 | 84.064 (x5.10) | 134.2e (x7.95) INVALID-GRAPH | 56.753 (x3.44) | 57.934 (x3.51) | 58.033 (x3.52) | 92.468 (x5.61) | HW | 55.749 (x3.38) |
| train | 512 | 19.486 | 2.038 | 8.434 (x4.14) | 12.507e (x6.04) INVALID-GRAPH | 6.870 (x3.37) | 7.757 (x3.81) | 7.770 (x3.81) | 8.026 (x3.94) | NOBWD | 6.659 (x3.27) |
| train | 1024 | OOM | 11.905 | 51.068 (x4.29) | 90.148e (x7.55) INVALID-GRAPH | 37.592 (x3.16) | 34.986 (x2.94) | 34.923 (x2.93) | 46.741 (x3.93) | NOBWD | 35.892 (x3.01) |
| train | 2048 | OOM | 82.921 | 380.5 (x4.59) | 725.3e (x8.61) INVALID-GRAPH | 244.7 (x2.95) | 227.3 (x2.74) | 227.3 (x2.74) | 372.7 (x4.49) | NOBWD | 228.9 (x2.76) |

Faster than the engine: none.

**TriMul (TriangleMultiplication, outgoing, d_pair 128)**

| mode | L | `pytorch` | `cuda` | `cueq` | `fast_trimul` | `gpumode` | `geonn` |
|---|---|---|---|---|---|---|---|
| eval | 512 | 3.182 | 0.249 | 0.512 (x2.06) | 2.724 (x10.93) | 1.920 (x7.70) | 2.168 (x8.70) |
| eval | 1024 | 19.533 | 0.980 | 2.002 (x2.04) | 17.013 (x17.36) | 10.977 (x11.20) | 14.815 (x15.12) |
| eval | 2048 | 80.361 | 5.787 | 9.219 (x1.59) | 71.826 (x12.41) | 53.479 (x9.24) | 62.140 (x10.74) |
| train | 512 | 10.745 | 1.128 | 2.807 (x2.49) | 12.777 (x11.32) | NOBWD | NOBWD |
| train | 1024 | 80.993 | 4.596 | 10.777 (x2.34) | 91.505 (x19.91) | NOBWD | NOBWD |
| train | 2048 | 336.7 | 23.313 | 46.714 (x2.00) | 384.2 (x16.48) | NOBWD | NOBWD |

Faster than the engine: none.

**AttentionPairBias d_single 384, 16 heads (confidence / Pairformer single attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|
| eval | 512 | 0.709 | 0.066 | 0.704 (x10.69) | 0.701 (x10.65) | 0.701 (x10.65) |
| eval | 1024 | 2.586 | 0.153 | 2.536 (x16.60) | 2.533 (x16.57) | 2.558 (x16.73) |
| eval | 2048 | 10.454 | 0.493 | 10.207 (x20.72) | 10.204 (x20.71) | 10.384 (x21.08) |
| train | 512 | 1.487 | 0.183 | 1.474 (x8.07) | 1.473 (x8.06) | 1.484 (x8.12) |
| train | 1024 | 5.196 | 0.482 | 5.131 (x10.65) | 5.128 (x10.65) | 5.256 (x10.91) |
| train | 2048 | 20.406 | 1.658 | 19.919 (x12.02) | 19.911 (x12.01) | 20.622 (x12.44) |

Faster than the engine: none.

**AttentionPairBias d_single 768, 16 heads x 48 channels (AF3 token DiT attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|
| eval | 512 | 0.721 | SHAPE | 0.716 | 0.714 | 0.712 |
| eval | 1024 | 2.611 | SHAPE | 2.552 | 2.548 | 2.565 |
| eval | 2048 | 10.471 | SHAPE | 10.229 | 10.228 | 10.329 |
| train | 512 | 1.520 | SHAPE | 1.525 | 1.522 | 1.507 |
| train | 1024 | 5.232 | SHAPE | 5.200 | 5.196 | 5.220 |
| train | 2048 | 20.510 | SHAPE | 20.051 | 20.040 | 20.342 |

Faster than the engine: none. Engine cuda without a number: eval L=512 (unsupported); eval L=1024 (unsupported); eval L=2048 (unsupported); train L=512 (unsupported); train L=1024 (unsupported); train L=2048 (unsupported).

**OuterProductMean (d_msa 64, d_pair 128, d_hidden 32, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` |
|---|---|---|---|---|
| eval | 512 | 3.009 | 0.961 | HW |
| eval | 1024 | 9.565 | 3.943 | HW |
| eval | 2048 | 33.907 | 15.763 | HW |
| train | 512 | 7.155 | 3.266 | NOBWD |
| train | 1024 | 23.045 | 12.910 | NOBWD |
| train | 2048 | 84.385 | 51.444 | NOBWD |

Faster than the engine: none.

**MSAPairWeightedAveraging (d_msa 64, d_pair 128, 8 heads, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `ligo_pwa` |
|---|---|---|---|---|---|
| eval | 512 | 2.267 | 0.250 | 2.873 (x11.50) | 30.859 (x123.48) |
| eval | 1024 | 5.565 | 0.572 | 6.918 (x12.08) | 100.7 (x175.94) |
| eval | 2048 | 16.442 | 1.661 | 19.015 (x11.45) | 1052.0 (x633.27) |
| train | 512 | 4.946 | 1.086 | NOBWD | 39.424 (x36.31) |
| train | 1024 | 12.205 | 2.364 | NOBWD | 124.3 (x52.58) |
| train | 2048 | 35.203 | SHAPE | NOBWD | 1114.8 |

Faster than the engine: none. Engine cuda without a number: train L=2048 (unsupported).

**Transition d_hidden 128, n 4 (pair transition)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 1.773 | 0.190 | SHAPE | 0.966 (x5.09) |
| eval | 1024 | 6.975 | 0.663 | SHAPE | 3.750 (x5.66) |
| eval | 2048 | 27.752 | 2.757 | SHAPE | 14.820 (x5.38) |
| train | 512 | 4.936 | 0.912 | NOBWD | 3.177 (x3.49) |
| train | 1024 | 19.211 | 3.521 | NOBWD | 12.183 (x3.46) |
| train | 2048 | 76.279 | 15.035 | NOBWD | 48.342 (x3.22) |

Faster than the engine: none.

**Transition d_hidden 64, n 4 (MSA transition, depth 384)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 0.849 | 0.059 | SHAPE | 0.445 (x7.60) |
| eval | 1024 | 1.637 | 0.111 | SHAPE | 0.856 (x7.71) |
| eval | 2048 | 3.217 | 0.230 | SHAPE | 1.684 (x7.33) |
| train | 512 | 2.164 | 0.282 | NOBWD | 1.049 (x3.72) |
| train | 1024 | 4.169 | 0.540 | NOBWD | 2.013 (x3.73) |
| train | 2048 | 8.161 | 1.038 | NOBWD | 3.928 (x3.79) |

Faster than the engine: none.

### A100 (sm_80)

**TriAttn (TriangleAttention, starting, d_pair 128, 4 heads of 32)**

| mode | L | `pytorch` | `cuda` | `cueq` | `ds4sci` | `trifast` | `gnops` | `of3tri` | `megafold` | `bionemo_ir` | `turbopf` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | 512 | 14.156 | 0.991 | 3.962 (x4.00) | 5.118e (x5.15) INVALID-GRAPH | 3.745 (x3.78) | 3.124 (x3.15) | 3.132 (x3.16) | 4.502 (x4.54) | HW | HW |
| eval | 1024 | 107.5 | 6.176 | 21.651 (x3.51) | 32.386e (x5.23) INVALID-GRAPH | 19.252 (x3.12) | 17.373 (x2.81) | 17.332 (x2.81) | 25.212 (x4.08) | HW | HW |
| eval | 2048 | OOM | 40.068 | 155.9 (x3.89) | 243.0e (x5.98) INVALID-GRAPH | 112.1 (x2.80) | 121.6 (x3.03) | 121.5 (x3.03) | 167.8 (x4.19) | HW | HW |
| train | 512 | 34.906 | 5.257 | 14.505 (x2.76) | 21.812e (x4.17) INVALID-GRAPH | 12.327 (x2.34) | 73.456 (x13.97) | 73.492 (x13.98) | 15.984 (x3.04) | NOBWD | HW |
| train | 1024 | OOM | 32.051 | 90.984 (x2.84) | 154.5e (x4.81) INVALID-GRAPH | 69.217 (x2.16) | 408.2 (x12.74) | 409.0 (x12.76) | 98.394 (x3.07) | NOBWD | HW |
| train | 2048 | OOM | 221.6 | 663.6 (x2.99) | 1230.1e (x5.55) INVALID-GRAPH | 460.4 (x2.08) | 2788.8 (x12.58) | 2789.0 (x12.58) | 788.1 (x3.56) | NOBWD | HW |

Faster than the engine: none.

**TriMul (TriangleMultiplication, outgoing, d_pair 128)**

| mode | L | `pytorch` | `cuda` | `cueq` | `fast_trimul` | `gpumode` | `geonn` |
|---|---|---|---|---|---|---|---|
| eval | 512 | 6.747 | 0.593 | 0.969 (x1.63) | 5.631 (x9.50) | 5.379 (x9.07) | 5.097 (x8.60) |
| eval | 1024 | 33.728 | 3.141 | 4.500 (x1.43) | 28.502 (x9.07) | 24.493 (x7.80) | 25.776 (x8.21) |
| eval | 2048 | 141.2 | 17.556 | 22.698 (x1.29) | 121.1 (x6.90) | 124.1 (x7.07) | 108.1 (x6.16) |
| train | 512 | 25.006 | 2.612 | 5.045 (x1.93) | 28.475 (x10.90) | NOBWD | NOBWD |
| train | 1024 | 139.1 | 12.690 | 21.643 (x1.71) | 158.2 (x12.46) | NOBWD | NOBWD |
| train | 2048 | 570.8 | 65.464 | 110.1 (x1.68) | 653.0 (x9.98) | NOBWD | NOBWD |

Faster than the engine: none.

**AttentionPairBias d_single 384, 16 heads (confidence / Pairformer single attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|
| eval | 512 | 1.141 | 0.101 | 1.115 (x11.00) | 1.113 (x10.98) | 1.126 (x11.11) |
| eval | 1024 | 4.413 | 0.291 | 4.366 (x15.01) | 4.360 (x14.99) | 4.440 (x15.27) |
| eval | 2048 | 17.511 | 1.000 | 17.117 (x17.11) | 17.104 (x17.10) | 17.321 (x17.31) |
| train | 512 | 2.396 | 0.328 | 2.400 (x7.32) | 2.398 (x7.32) | 2.433 (x7.42) |
| train | 1024 | 8.832 | 0.940 | 8.831 (x9.39) | 8.821 (x9.38) | 9.038 (x9.61) |
| train | 2048 | 34.680 | 3.368 | 34.336 (x10.19) | 34.316 (x10.19) | 35.128 (x10.43) |

Faster than the engine: none.

**AttentionPairBias d_single 768, 16 heads x 48 channels (AF3 token DiT attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|
| eval | 512 | 1.163 | 0.172 | 1.144 (x6.65) | 1.141 (x6.63) | 1.153 (x6.70) |
| eval | 1024 | 4.445 | 0.463 | 4.366 (x9.43) | 4.355 (x9.41) | 4.397 (x9.50) |
| eval | 2048 | 17.578 | 1.595 | 17.195 (x10.78) | 17.185 (x10.77) | 17.270 (x10.83) |
| train | 512 | 2.463 | 0.774 | 3.503 (x4.53) | 3.506 (x4.53) | 2.478 (x3.20) |
| train | 1024 | 8.947 | 2.212 | 11.807 (x5.34) | 11.794 (x5.33) | 9.029 (x4.08) |
| train | 2048 | 34.878 | 7.931 | 42.967 (x5.42) | 42.918 (x5.41) | 34.863 (x4.40) |

Faster than the engine: none.

**OuterProductMean (d_msa 64, d_pair 128, d_hidden 32, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` |
|---|---|---|---|---|
| eval | 512 | 6.284 | 3.021 | HW |
| eval | 1024 | 21.121 | 12.448 | HW |
| eval | 2048 | 78.788 | 48.804 | HW |
| train | 512 | 15.852 | 9.230 | NOBWD |
| train | 1024 | 55.326 | 36.336 | NOBWD |
| train | 2048 | 206.1 | 143.1 | NOBWD |

Faster than the engine: none.

**MSAPairWeightedAveraging (d_msa 64, d_pair 128, 8 heads, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `ligo_pwa` |
|---|---|---|---|---|---|
| eval | 512 | 3.865 | 0.484 | 4.957 (x10.23) | 65.829 (x135.91) |
| eval | 1024 | 9.756 | 1.269 | 11.926 (x9.40) | 262.4 (x206.79) |
| eval | 2048 | 29.051 | 4.255 | 33.388 (x7.85) | 2429.7 (x571.07) |
| train | 512 | 8.822 | 1.736 | NOBWD | 82.754 (x47.68) |
| train | 1024 | 22.003 | 4.966 | NOBWD | 308.4 (x62.09) |
| train | 2048 | 64.367 | SHAPE | NOBWD | 2578.6 |

Faster than the engine: none. Engine cuda without a number: train L=2048 (unsupported).

**Transition d_hidden 128, n 4 (pair transition)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 3.413 | 0.453 | SHAPE | 1.960 (x4.33) |
| eval | 1024 | 13.620 | 1.848 | SHAPE | 8.121 (x4.40) |
| eval | 2048 | 54.574 | 7.876 | SHAPE | 32.623 (x4.14) |
| train | 512 | 9.501 | 2.477 | NOBWD | 5.912 (x2.39) |
| train | 1024 | 36.615 | 9.915 | NOBWD | 23.204 (x2.34) |
| train | 2048 | 146.3 | 39.960 | NOBWD | 93.373 (x2.34) |

Faster than the engine: none.

**Transition d_hidden 64, n 4 (MSA transition, depth 384)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 1.544 | 0.123 | SHAPE | 0.765 (x6.23) |
| eval | 1024 | 2.978 | 0.225 | SHAPE | 1.499 (x6.68) |
| eval | 2048 | 5.884 | 0.432 | SHAPE | 3.019 (x7.00) |
| train | 512 | 3.998 | 0.655 | NOBWD | 1.907 (x2.91) |
| train | 1024 | 7.649 | 1.207 | NOBWD | 3.879 (x3.21) |
| train | 2048 | 15.021 | 2.397 | NOBWD | 7.735 (x3.23) |

Faster than the engine: none.

### RTX 5090 (sm_120)

**TriAttn (TriangleAttention, starting, d_pair 128, 4 heads of 32)**

| mode | L | `pytorch` | `cuda` | `cueq` | `ds4sci` | `trifast` | `gnops` | `of3tri` | `megafold` | `bionemo_ir` | `turbopf` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | 512 | 12.104 | 1.226 | 1.880 (x1.53) | FAIL | 2.366 (x1.93) | 1.998 (x1.63) | 1.996 (x1.63) | 2.827 (x2.31) | 2.246 (x1.83) | HW |
| eval | 1024 | OOM | 6.362 | 8.789 (x1.38) | FAIL | 11.263 (x1.77) | 9.908 (x1.56) | 9.924 (x1.56) | 14.980 (x2.35) | 10.121 (x1.59) | HW |
| eval | 2048 | OOM | 38.322 | 48.607 (x1.27) | FAIL | 61.535 (x1.61) | 56.199 (x1.47) | 56.158 (x1.47) | 89.395 (x2.33) | 53.701 (x1.40) | HW |
| train | 512 | 31.500 | 4.327 | 6.576 (x1.52) | FAIL | 7.830 (x1.81) | 10.788 (x2.49) | 10.642 (x2.46) | 8.150 (x1.88) | NOBWD | HW |
| train | 1024 | OOM | 25.377 | 36.697 (x1.45) | FAIL | 42.441 (x1.67) | 49.876 (x1.97) | 49.898 (x1.97) | 47.327 (x1.86) | NOBWD | HW |
| train | 2048 | OOM | OOM | 244.6 | FAIL | 255.8 | 298.3 | 298.5 | 378.1 | NOBWD | HW |

Faster than the engine: none. Engine cuda without a number: train L=2048 (skipped).

**TriMul (TriangleMultiplication, outgoing, d_pair 128)**

| mode | L | `pytorch` | `cuda` | `cueq` | `fast_trimul` | `gpumode` | `geonn` |
|---|---|---|---|---|---|---|---|
| eval | 512 | 3.920 | 0.592 | 0.950 (x1.61) | 3.003 (x5.07) | 2.721 (x4.60) | 2.788 (x4.71) |
| eval | 1024 | 25.349 | 2.577 | 3.830 (x1.49) | 18.885 (x7.33) | 13.349 (x5.18) | 19.428 (x7.54) |
| eval | 2048 | 112.5 | 15.441 | 20.962 (x1.36) | 85.666 (x5.55) | 66.098 (x4.28) | 96.041 (x6.22) |
| train | 512 | 12.584 | 2.670 | 4.515 (x1.69) | 13.510 (x5.06) | NOBWD | NOBWD |
| train | 1024 | 104.4 | 12.783 | 17.716 (x1.39) | 112.9 (x8.83) | NOBWD | NOBWD |
| train | 2048 | 468.3 | 66.299 | 88.038 (x1.33) | 516.5 (x7.79) | NOBWD | NOBWD |

Faster than the engine: none.

**AttentionPairBias d_single 384, 16 heads (confidence / Pairformer single attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|
| eval | 512 | 0.655 | 0.080 | 0.644 (x8.06) | 0.642 (x8.03) | 0.645 (x8.07) |
| eval | 1024 | 2.594 | 0.253 | 2.549 (x10.06) | 2.540 (x10.02) | 2.536 (x10.01) |
| eval | 2048 | 10.855 | 0.944 | 10.725 (x11.36) | 10.735 (x11.37) | 10.676 (x11.31) |
| train | 512 | 1.565 | 0.227 | 1.579 (x6.95) | 1.580 (x6.95) | 1.564 (x6.88) |
| train | 1024 | 5.945 | 0.805 | 5.881 (x7.31) | 5.880 (x7.31) | 5.923 (x7.36) |
| train | 2048 | 24.474 | 3.057 | 24.302 (x7.95) | 24.290 (x7.95) | 24.465 (x8.00) |

Faster than the engine: none.

**AttentionPairBias d_single 768, 16 heads x 48 channels (AF3 token DiT attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|
| eval | 512 | 0.655 | 0.102 | 0.690 (x6.74) | 0.688 (x6.72) | 0.690 (x6.74) |
| eval | 1024 | 2.640 | 0.289 | 2.580 (x8.94) | 2.579 (x8.93) | 2.606 (x9.03) |
| eval | 2048 | 11.013 | 0.992 | 11.169 (x11.26) | 11.300 (x11.39) | 11.329 (x11.42) |
| train | 512 | 1.705 | 0.544 | 1.752 (x3.22) | 1.742 (x3.20) | 1.672 (x3.08) |
| train | 1024 | 6.257 | 1.222 | 6.255 (x5.12) | 6.215 (x5.09) | 6.076 (x4.97) |
| train | 2048 | 24.933 | 4.229 | 25.273 (x5.98) | 25.271 (x5.98) | 24.796 (x5.86) |

Faster than the engine: none.

**OuterProductMean (d_msa 64, d_pair 128, d_hidden 32, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` |
|---|---|---|---|---|
| eval | 512 | 4.807 | 2.969 | 4.396 (x1.48) |
| eval | 1024 | 16.849 | 11.538 | 14.048 (x1.22) |
| eval | 2048 | 62.457 | 45.860 | 52.071 (x1.14) |
| train | 512 | 13.331 | 9.563 | NOBWD |
| train | 1024 | 46.809 | 36.443 | NOBWD |
| train | 2048 | OOM | 142.2 | NOBWD |

Faster than the engine: none.

**MSAPairWeightedAveraging (d_msa 64, d_pair 128, 8 heads, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `ligo_pwa` |
|---|---|---|---|---|---|
| eval | 512 | 2.406 | 0.463 | 3.387 (x7.32) | 41.858 (x90.44) |
| eval | 1024 | 6.268 | 1.294 | 8.108 (x6.27) | 140.6 (x108.65) |
| eval | 2048 | 18.947 | 4.054 | 22.688 (x5.60) | 495.0 (x122.11) |
| train | 512 | 6.204 | 1.479 | NOBWD | 50.487 (x34.13) |
| train | 1024 | 16.013 | 4.071 | NOBWD | 168.4 (x41.35) |
| train | 2048 | 47.858 | SHAPE | NOBWD | 579.1 |

Faster than the engine: none. Engine cuda without a number: train L=2048 (unsupported).

**Transition d_hidden 128, n 4 (pair transition)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 2.846 | 0.454 | SHAPE | 1.531 (x3.37) |
| eval | 1024 | 12.317 | 1.586 | SHAPE | 6.450 (x4.07) |
| eval | 2048 | 47.520 | 5.886 | SHAPE | 25.377 (x4.31) |
| train | 512 | 8.413 | 1.906 | NOBWD | 4.391 (x2.30) |
| train | 1024 | 34.929 | 7.521 | NOBWD | 17.647 (x2.35) |
| train | 2048 | OOM | 30.200 | NOBWD | 70.964 (x2.35) |

Faster than the engine: none.

**Transition d_hidden 64, n 4 (MSA transition, depth 384)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 1.023 | 0.088 | SHAPE | 0.539 (x6.12) |
| eval | 1024 | 2.221 | 0.160 | SHAPE | 1.112 (x6.96) |
| eval | 2048 | 4.588 | 0.311 | SHAPE | 2.355 (x7.56) |
| train | 512 | 3.100 | 0.481 | NOBWD | 1.425 (x2.97) |
| train | 1024 | 6.472 | 0.965 | NOBWD | 2.870 (x2.97) |
| train | 2048 | 13.128 | 1.852 | NOBWD | 5.851 (x3.16) |

Faster than the engine: none.

### B200 (sm_100)

**TriAttn (TriangleAttention, starting, d_pair 128, 4 heads of 32)**

| mode | L | `pytorch` | `cuda` | `cueq` | `ds4sci` | `trifast` | `gnops` | `of3tri` | `megafold` | `bionemo_ir` | `turbopf` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | 512 | 5.200 | 0.287 | 1.949 (x6.80) | 2.491e (x8.78) INVALID-GRAPH | 1.483 (x5.17) | 1.251 (x4.37) | 1.248 (x4.36) | 2.049 (x7.15) | 0.850 (x2.96) | HW |
| eval | 1024 | 36.388 | 1.497 | 11.019 (x7.36) | 16.231e (x10.86) INVALID-GRAPH | 7.650 (x5.11) | 6.905 (x4.61) | 6.902 (x4.61) | 11.799 (x7.88) | 3.854 (x2.57) | HW |
| eval | 2048 | OOM | 10.994 | 71.373 (x6.49) | 118.0e (x10.95) INVALID-GRAPH | 44.594 (x4.06) | 43.765 (x3.98) | 43.748 (x3.98) | 77.125 (x7.02) | 19.960 (x1.82) | HW |
| train | 512 | 11.534 | 1.194 | 7.062 (x5.91) | 9.126e (x7.32) INVALID-GRAPH | 5.274 (x4.42) | 7.084 (x5.93) | 7.084 (x5.93) | 6.392 (x5.35) | NOBWD | HW |
| train | 1024 | 79.053 | 7.505 | 43.751 (x5.83) | 61.367e (x8.36) INVALID-GRAPH | 30.913 (x4.12) | 31.635 (x4.22) | 31.622 (x4.21) | 38.586 (x5.14) | NOBWD | HW |
| train | 2048 | OOM | 54.836 | 331.8 (x6.05) | 457.7e (x8.59) INVALID-GRAPH | 203.1 (x3.70) | 203.2 (x3.71) | 203.2 (x3.70) | 266.8 (x4.87) | NOBWD | HW |

Faster than the engine: none.

**TriMul (TriangleMultiplication, outgoing, d_pair 128)**

| mode | L | `pytorch` | `cuda` | `cueq` | `fast_trimul` | `gpumode` | `geonn` |
|---|---|---|---|---|---|---|---|
| eval | 512 | 1.867 | 0.142 | 0.333 (x2.35) | 1.936 (x13.64) | 1.078 (x7.59) | 1.349 (x9.51) |
| eval | 1024 | 12.688 | 0.563 | 1.246 (x2.22) | 12.009 (x21.35) | 8.250 (x14.66) | 9.453 (x16.80) |
| eval | 2048 | 53.736 | 2.963 | 5.648 (x1.91) | 52.793 (x17.82) | 42.379 (x14.30) | 41.870 (x14.13) |
| train | 512 | 5.709 | 0.511 | 2.098 (x4.11) | 7.528 (x14.74) | NOBWD | NOBWD |
| train | 1024 | 50.346 | 2.075 | 7.844 (x3.78) | 58.087 (x27.99) | NOBWD | NOBWD |
| train | 2048 | 219.4 | 11.337 | 33.190 (x2.93) | 255.1 (x22.50) | NOBWD | NOBWD |

Faster than the engine: none.

**AttentionPairBias d_single 384, 16 heads (confidence / Pairformer single attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `cueq` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|---|
| eval | 512 | 0.518 | 0.041 | 0.565 (x13.80) | 0.518 (x12.65) | 0.514 (x12.55) | 0.512 (x12.50) |
| eval | 1024 | 1.807 | 0.084 | 1.897 (x22.61) | 1.778 (x21.19) | 1.773 (x21.12) | 1.789 (x21.31) |
| eval | 2048 | 7.144 | 0.248 | 7.399 (x29.85) | 6.931 (x27.96) | 6.927 (x27.95) | 7.092 (x28.61) |
| train | 512 | 1.157 | 0.133 | 1.204 (x9.08) | 1.148 (x8.66) | 1.145 (x8.63) | 1.145 (x8.63) |
| train | 1024 | 3.788 | 0.334 | 3.942 (x11.82) | 3.721 (x11.15) | 3.717 (x11.14) | 3.819 (x11.45) |
| train | 2048 | 14.714 | 1.187 | 15.297 (x12.89) | 14.183 (x11.95) | 14.179 (x11.95) | 14.723 (x12.41) |

Faster than the engine: none.

**AttentionPairBias d_single 768, 16 heads x 48 channels (AF3 token DiT attention; master dtype)**

| mode | L | `pytorch` | `cuda` | `cueq` | `gnops` | `of3tri` | `megafold` |
|---|---|---|---|---|---|---|---|
| eval | 512 | 0.527 | SHAPE | 0.575 | 0.528 | 0.524 | 0.514 |
| eval | 1024 | 1.816 | SHAPE | 1.910 | 1.791 | 1.785 | 1.789 |
| eval | 2048 | 7.166 | SHAPE | 7.430 | 6.949 | 6.946 | 7.025 |
| train | 512 | 1.176 | SHAPE | 1.226 | 1.168 | 1.165 | 1.149 |
| train | 1024 | 3.819 | SHAPE | 3.967 | 3.760 | 3.759 | 3.792 |
| train | 2048 | 14.764 | SHAPE | 15.345 | 14.353 | 14.348 | 14.463 |

Faster than the engine: none. Engine cuda without a number: eval L=512 (unsupported); eval L=1024 (unsupported); eval L=2048 (unsupported); train L=512 (unsupported); train L=1024 (unsupported); train L=2048 (unsupported).

**OuterProductMean (d_msa 64, d_pair 128, d_hidden 32, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` |
|---|---|---|---|---|
| eval | 512 | 2.069 | 0.467 | 1.306 (x2.80) |
| eval | 1024 | 6.512 | 1.827 | 3.433 (x1.88) |
| eval | 2048 | 22.823 | 8.272 | 10.321 (x1.25) |
| train | 512 | 4.938 | 1.429 | NOBWD |
| train | 1024 | 15.347 | 5.755 | NOBWD |
| train | 2048 | 53.988 | 25.406 | NOBWD |

Faster than the engine: none.

**MSAPairWeightedAveraging (d_msa 64, d_pair 128, 8 heads, depth 1024)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `ligo_pwa` |
|---|---|---|---|---|---|
| eval | 512 | 1.636 | 0.130 | 1.981 (x15.30) | 34.500 (x266.37) |
| eval | 1024 | 3.834 | 0.294 | 4.840 (x16.45) | 86.225 (x292.96) |
| eval | 2048 | 11.090 | 0.790 | 13.090 (x16.57) | 398.2 (x504.16) |
| train | 512 | 3.484 | 0.544 | NOBWD | 39.546 (x72.72) |
| train | 1024 | 8.611 | 1.196 | NOBWD | 104.7 (x87.55) |
| train | 2048 | 24.246 | 3.217 | NOBWD | 449.0 (x139.55) |

Faster than the engine: none.

**Transition d_hidden 128, n 4 (pair transition)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 0.987 | 0.086 | 0.504 (x5.85) | 0.741 (x8.61) |
| eval | 1024 | 3.784 | 0.307 | 1.954 (x6.36) | 2.841 (x9.25) |
| eval | 2048 | 14.969 | 1.259 | 7.734 (x6.14) | 11.247 (x8.93) |
| train | 512 | 2.662 | 0.436 | NOBWD | 1.707 (x3.91) |
| train | 1024 | 10.072 | 1.702 | NOBWD | 6.499 (x3.82) |
| train | 2048 | 39.704 | 7.571 | NOBWD | 25.554 (x3.38) |

Faster than the engine: none.

**Transition d_hidden 64, n 4 (MSA transition, depth 384)**

| mode | L | `pytorch` | `cuda` | `bionemo_ir` | `megafold_tr` |
|---|---|---|---|---|---|
| eval | 512 | 0.483 | 0.037 | 0.279 (x7.57) | 0.374 (x10.16) |
| eval | 1024 | 0.950 | 0.061 | 0.558 (x9.13) | 0.723 (x11.83) |
| eval | 2048 | 1.854 | 0.109 | 1.101 (x10.14) | 1.416 (x13.03) |
| train | 512 | 1.229 | 0.141 | NOBWD | 0.828 (x5.86) |
| train | 1024 | 2.395 | 0.256 | NOBWD | 1.574 (x6.15) |
| train | 2048 | 4.600 | 0.485 | NOBWD | 3.032 (x6.25) |

Faster than the engine: none.

Reasons behind the codes (backend, code: recorded reason, cards):

- `bionemo_ir` HW: native library bionemo_ir.libs._cutedsl_kernels cannot be loaded on this host (/lib64/libc.so.6: version `GLIBC_2.34' not found): its CuTeDSL kernels are unavailable (h100)
- `bionemo_ir` HW: native library bionemo_ir.libs._cutedsl_kernels cannot be loaded on this host (cuModuleLoadData(015e13181a6001abecf6) failed: CUDA_ERROR_INVALID_IMAGE: device kernel imag (a100)
- `bionemo_ir` NOBWD: BioIR kernels are inference only (no backward) (a100, b200, h100, rtx5090)
- `bionemo_ir` SHAPE: width 128, hidden 512: no shipped silu_gate (no bias / no mask / residual) variant on this SM (a100, h100, rtx5090)
- `bionemo_ir` SHAPE: width 64, hidden 256: no shipped silu_gate (no bias / no mask / residual) variant on this SM (a100, h100, rtx5090)
- `cuda` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (rtx5090)
- `cuda` SHAPE: 16 heads x d_single 768: the kernels are built for (heads, d_single) in ((8, 384), (12, 384), (16, 384), (24, 384), (16, 512)) with n_head * d_hidden == d_single (b200, h100)
- `cuda` SHAPE: pwa_sm80: L 2048: the sm_80 kernels take L a multiple of 16 up to 1024 in training (training 1024: the pair backward's shared-memory row; inference 8192) and a [1, L, L,  (a100, rtx5090)
- `cuda` SHAPE: training past 1024 tokens: the H100 pair backward (pair3.cu pair_bwd3) holds N <= 1024, got N=2048 (inference serves N <= 8192) (h100)
- `ds4sci` FAIL: RuntimeError: Unable to JIT load the evoformer_attn op due to it not being compatible due to hardware/software issue. None ; eager: RuntimeError: Unable to JIT load the e (rtx5090)
- `geonn` NOBWD: training: the library has no backward kernel (its autograd path is its PyTorch reference) (a100, b200, h100, rtx5090)
- `gpumode` NOBWD: training: forward-only kernels (no backward) (a100, b200, h100, rtx5090)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (b200)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (b200)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (rtx5090)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (rtx5090)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (h100)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (a100)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (rtx5090)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (rtx5090)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (rtx5090)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (h100)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (h100)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (a100)
- `pytorch` OOM: CUDA out of memory (allocation larger than the free memory of the card at that size) (rtx5090)
- `turbopf` HW: turbopf: ImportError: Hopper (sm_90) only; this device is capability (10, 0) (see third_party/ README, fetch.sh) (b200)
- `turbopf` HW: turbopf: ImportError: Hopper (sm_90) only; this device is capability (12, 0) (see third_party/ README, fetch.sh) (rtx5090)
- `turbopf` HW: turbopf: ImportError: Hopper (sm_90) only; this device is capability (8, 0) (see third_party/ README, fetch.sh) (a100)

## 4. Accuracy against the float64 gold, per candidate

Two harnesses, both against the module's own float64 PyTorch result with the same weights and inputs. Part A (`miniworld_engine.measure.accuracy`, the engine's accuracy standard; smallest length L = 512, master dtype): relative error of the checked tensor (`out` for eval, the worst gradient tensor for train) and the PyTorch backend's own error on the same case; the light compares with the tolerance (GREEN within, YELLOW slightly above, RED far above; backends that are external baselines are informational). The accuracy runs are eager (not graph) runs: the CUDA-graph problem of `ds4sci` does not touch them. Part B (TriMul, `acc_trimul.py`): relative rms of the output and of the gradients (`dx`, `dWo`) at L = 512, d_pair 128, no padding.

### H100 (sm_90): part A (L = 512)

| class | backend | eval error (light) | train error (light, tensor) | note |
|---|---|---|---|---|
| TriAttn | `pytorch` | 3.73e-03 (GREEN) | 8.76e-03 (GREEN, grad:pair) |  |
| TriAttn | `cuda` | 3.31e-03 (GREEN) [pytorch 3.73e-03] | 7.86e-03 (GREEN, grad:to_query.weight) [pytorch 8.57e-03] |  |
| TriAttn | `ds4sci` | 3.73e-03 (GREEN) [pytorch 3.73e-03] | 9.81e-03 (YELLOW, grad:pair) [pytorch 8.76e-03] | new in this integration |
| TriAttn | `trifast` | 3.73e-03 (GREEN) [pytorch 3.73e-03] | 9.81e-03 (YELLOW, grad:pair) [pytorch 8.76e-03] | new in this integration |
| TriAttn | `gnops` | 3.73e-03 (GREEN) [pytorch 3.73e-03] | 9.81e-03 (YELLOW, grad:pair) [pytorch 8.76e-03] |  |
| TriAttn | `of3tri` | 3.73e-03 (GREEN) [pytorch 3.73e-03] | 9.81e-03 (YELLOW, grad:pair) [pytorch 8.76e-03] |  |
| TriAttn | `megafold` | 3.73e-03 (GREEN) [pytorch 3.73e-03] | 9.81e-03 (YELLOW, grad:pair) [pytorch 8.76e-03] |  |
| TriAttn | `bionemo_ir` | unsupported: HW | unsupported: NOBWD |  |
| TriAttn | `turbopf` | 3.73e-03 (GREEN) [pytorch 3.73e-03] | 9.81e-03 (YELLOW, grad:pair) [pytorch 8.76e-03] |  |
| AttentionPairBias-384 | `pytorch` | 2.90e-03 (GREEN) | 9.78e-03 (GREEN, grad:pair) |  |
| AttentionPairBias-384 | `cuda` | 2.94e-03 (GREEN) [pytorch 2.90e-03] | 9.54e-03 (YELLOW, grad:ln_pair.weight) [pytorch 8.69e-03] |  |
| AttentionPairBias-384 | `gnops` | 2.90e-03 (GREEN) [pytorch 2.90e-03] | 1.07e-02 (YELLOW, grad:ln_pair.weight) [pytorch 8.69e-03] |  |
| AttentionPairBias-384 | `of3tri` | 2.90e-03 (GREEN) [pytorch 2.90e-03] | 1.07e-02 (YELLOW, grad:ln_pair.weight) [pytorch 8.69e-03] |  |
| AttentionPairBias-384 | `megafold` | 2.90e-03 (GREEN) [pytorch 2.90e-03] | 1.16e-02 (YELLOW, grad:ln_pair.weight) [pytorch 8.69e-03] |  |
| AttentionPairBias-768 | `pytorch` | 3.32e-03 (GREEN) | 1.24e-02 (GREEN, grad:pair) |  |
| AttentionPairBias-768 | `cuda` | unsupported: SHAPE | unsupported: SHAPE |  |
| AttentionPairBias-768 | `gnops` | 3.32e-03 (GREEN) [pytorch 3.32e-03] | 9.11e-03 (YELLOW, grad:to_key.weight) [pytorch 8.46e-03] |  |
| AttentionPairBias-768 | `of3tri` | 3.32e-03 (GREEN) [pytorch 3.32e-03] | 9.11e-03 (YELLOW, grad:to_key.weight) [pytorch 8.46e-03] |  |
| AttentionPairBias-768 | `megafold` | 3.32e-03 (GREEN) [pytorch 3.32e-03] | 9.16e-03 (YELLOW, grad:to_bias.weight) [pytorch 8.68e-03] |  |
| OPM | `pytorch` | 3.13e-03 (GREEN) | 6.80e-03 (GREEN, grad:msa) |  |
| OPM | `cuda` | 3.08e-03 (GREEN) [pytorch 3.13e-03] | 6.61e-03 (GREEN, grad:ln_msa.weight) [pytorch 6.08e-03] |  |
| OPM | `bionemo_ir` | unsupported: HW | unsupported: NOBWD |  |
| PWA | `pytorch` | 3.19e-03 (GREEN) | 1.23e-02 (GREEN, grad:pair) |  |
| PWA | `cuda` | 3.11e-03 (GREEN) [pytorch 3.19e-03] | 1.58e-02 (YELLOW, grad:pair) [pytorch 1.23e-02] |  |
| PWA | `bionemo_ir` | 7.00e-04 (GREEN) [pytorch 3.19e-03] | unsupported: NOBWD |  |
| PWA | `ligo_pwa` | 3.10e-03 (GREEN) [pytorch 3.19e-03] | 1.58e-02 (YELLOW, grad:pair) [pytorch 1.23e-02] |  |
| Transition-d128 | `pytorch` | 6.88e-03 (GREEN) | 8.62e-03 (GREEN, grad:ln_in.bias) |  |
| Transition-d128 | `cuda` | 3.80e-03 (GREEN) [pytorch 6.88e-03] | 7.50e-03 (GREEN, grad:x) [pytorch 7.93e-03] |  |
| Transition-d128 | `bionemo_ir` | unsupported: SHAPE | unsupported: NOBWD |  |
| Transition-d128 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.88e-03] | 1.09e-02 (YELLOW, grad:x) [pytorch 7.93e-03] |  |
| Transition-d64 | `pytorch` | 6.09e-03 (GREEN) | 8.46e-03 (GREEN, grad:x) |  |
| Transition-d64 | `cuda` | 4.46e-03 (GREEN) [pytorch 6.09e-03] | 6.18e-03 (GREEN, grad:x) [pytorch 8.46e-03] |  |
| Transition-d64 | `bionemo_ir` | unsupported: SHAPE | unsupported: NOBWD |  |
| Transition-d64 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.09e-03] | 1.04e-02 (YELLOW, grad:x) [pytorch 8.46e-03] |  |

H100 (sm_90): part B (TriMul, L = 512, d_pair 128)

| backend | output rel rms | output rel max | update rel rms | dx rel rms | dWo rel rms |
|---|---|---|---|---|---|
| `pytorch` | 3.65e-03 | 5.50e-03 | 7.69e-03 | 5.47e-03 | 6.64e-03 |
| `cuda` | 3.12e-03 | 4.66e-03 | 6.57e-03 | 4.61e-03 | 5.41e-03 |
| `fast_trimul` | 1.98e-03 | 3.80e-03 | 4.18e-03 | 2.75e-03 | 1.89e-03 |
| `gpumode` | 2.31e-03 | 4.25e-03 | 4.86e-03 | - | - |
| `geonn` | 3.46e-03 | 5.10e-03 | 7.30e-03 | - | - |

### A100 (sm_80): part A (L = 512)

| class | backend | eval error (light) | train error (light, tensor) | note |
|---|---|---|---|---|
| TriAttn | `pytorch` | 3.41e-03 (GREEN) | 1.06e-02 (GREEN, grad:ln_pair.weight) |  |
| TriAttn | `cuda` | 3.66e-03 (GREEN) [pytorch 3.41e-03] | 9.15e-03 (GREEN, grad:to_bias.weight) [pytorch 1.21e-02] |  |
| TriAttn | `ds4sci` | 3.37e-03 (GREEN) [pytorch 3.41e-03] | 1.04e-02 (GREEN, grad:ln_pair.weight) [pytorch 1.06e-02] | new in this integration |
| TriAttn | `trifast` | 3.37e-03 (GREEN) [pytorch 3.41e-03] | 6.79e-03 (GREEN, grad:pair) [pytorch 7.50e-03] | new in this integration |
| TriAttn | `gnops` | 3.37e-03 (GREEN) [pytorch 3.41e-03] | 6.77e-03 (GREEN, grad:pair) [pytorch 7.50e-03] |  |
| TriAttn | `of3tri` | 3.37e-03 (GREEN) [pytorch 3.41e-03] | 6.77e-03 (GREEN, grad:pair) [pytorch 7.50e-03] |  |
| TriAttn | `megafold` | 3.37e-03 (GREEN) [pytorch 3.41e-03] | 1.07e-02 (YELLOW, grad:ln_pair.weight) [pytorch 1.06e-02] |  |
| TriAttn | `bionemo_ir` | unsupported: HW | unsupported: NOBWD |  |
| TriAttn | `turbopf` | unsupported: HW | unsupported: HW |  |
| AttentionPairBias-384 | `pytorch` | 2.27e-03 (GREEN) | 9.53e-03 (GREEN, grad:ln_single.weight) |  |
| AttentionPairBias-384 | `cuda` | 2.27e-03 (GREEN) [pytorch 2.27e-03] | 1.12e-02 (YELLOW, grad:pair) [pytorch 7.98e-03] |  |
| AttentionPairBias-384 | `gnops` | 2.27e-03 (GREEN) [pytorch 2.27e-03] | 8.80e-03 (GREEN, grad:to_query.weight) [pytorch 8.80e-03] |  |
| AttentionPairBias-384 | `of3tri` | 2.27e-03 (GREEN) [pytorch 2.27e-03] | 8.80e-03 (GREEN, grad:to_query.weight) [pytorch 8.80e-03] |  |
| AttentionPairBias-384 | `megafold` | 2.27e-03 (GREEN) [pytorch 2.27e-03] | 1.00e-02 (YELLOW, grad:to_key.weight) [pytorch 8.67e-03] |  |
| AttentionPairBias-768 | `pytorch` | 3.23e-03 (GREEN) | 8.22e-03 (GREEN, grad:ln_single.weight) |  |
| AttentionPairBias-768 | `cuda` | 3.09e-03 (GREEN) [pytorch 3.23e-03] | 8.53e-03 (YELLOW, grad:ln_single.weight) [pytorch 8.22e-03] |  |
| AttentionPairBias-768 | `gnops` | 3.09e-03 (GREEN) [pytorch 3.23e-03] | 8.61e-03 (YELLOW, grad:to_query.weight) [pytorch 8.45e-03] |  |
| AttentionPairBias-768 | `of3tri` | 3.09e-03 (GREEN) [pytorch 3.23e-03] | 8.61e-03 (YELLOW, grad:to_query.weight) [pytorch 8.45e-03] |  |
| AttentionPairBias-768 | `megafold` | 3.09e-03 (GREEN) [pytorch 3.23e-03] | 8.20e-03 (YELLOW, grad:to_key.weight) [pytorch 7.06e-03] |  |
| OPM | `pytorch` | 3.13e-03 (GREEN) | 7.12e-03 (GREEN, grad:msa) |  |
| OPM | `cuda` | 3.08e-03 (GREEN) [pytorch 3.13e-03] | 6.26e-03 (GREEN, grad:ln_msa.bias) [pytorch 6.13e-03] |  |
| OPM | `bionemo_ir` | unsupported: HW | unsupported: NOBWD |  |
| PWA | `pytorch` | 3.22e-03 (GREEN) | 1.53e-02 (GREEN, grad:pair) |  |
| PWA | `cuda` | 3.22e-03 (GREEN) [pytorch 3.22e-03] | 1.53e-02 (GREEN, grad:pair) [pytorch 1.53e-02] |  |
| PWA | `bionemo_ir` | 7.12e-04 (GREEN) [pytorch 3.22e-03] | unsupported: NOBWD |  |
| PWA | `ligo_pwa` | 3.12e-03 (GREEN) [pytorch 3.22e-03] | 1.53e-02 (GREEN, grad:pair) [pytorch 1.53e-02] |  |
| Transition-d128 | `pytorch` | 6.88e-03 (GREEN) | 1.09e-02 (GREEN, grad:expand_a.weight) |  |
| Transition-d128 | `cuda` | 6.88e-03 (GREEN) [pytorch 6.88e-03] | 6.88e-03 (GREEN, out) [pytorch 6.88e-03] |  |
| Transition-d128 | `bionemo_ir` | unsupported: SHAPE | unsupported: NOBWD |  |
| Transition-d128 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.88e-03] | 1.09e-02 (YELLOW, grad:x) [pytorch 7.93e-03] |  |
| Transition-d64 | `pytorch` | 6.09e-03 (GREEN) | 8.46e-03 (GREEN, grad:x) |  |
| Transition-d64 | `cuda` | 4.12e-03 (GREEN) [pytorch 6.09e-03] | 4.52e-03 (GREEN, grad:expand_a.weight) [pytorch 6.61e-03] |  |
| Transition-d64 | `bionemo_ir` | unsupported: SHAPE | unsupported: NOBWD |  |
| Transition-d64 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.09e-03] | 9.84e-03 (YELLOW, grad:x) [pytorch 8.46e-03] |  |

A100 (sm_80): part B (TriMul, L = 512, d_pair 128)

| backend | output rel rms | output rel max | update rel rms | dx rel rms | dWo rel rms |
|---|---|---|---|---|---|
| `pytorch` | 3.65e-03 | 7.41e-03 | 7.69e-03 | 5.47e-03 | 6.88e-03 |
| `cuda` | 3.22e-03 | 5.44e-03 | 6.78e-03 | 4.26e-03 | 5.10e-03 |
| `fast_trimul` | 1.98e-03 | 3.85e-03 | 4.18e-03 | 2.75e-03 | 1.92e-03 |
| `gpumode` | 2.30e-03 | 5.38e-03 | 4.86e-03 | - | - |
| `geonn` | 3.46e-03 | 6.41e-03 | 7.30e-03 | - | - |

### RTX 5090 (sm_120): part A (L = 512)

| class | backend | eval error (light) | train error (light, tensor) | note |
|---|---|---|---|---|
| TriAttn | `pytorch` | 3.45e-03 (GREEN) | 9.90e-03 (GREEN, grad:ln_pair.weight) |  |
| TriAttn | `cuda` | 3.88e-03 (GREEN) [pytorch 3.45e-03] | 8.34e-03 (GREEN, grad:to_key.weight) [pytorch 9.04e-03] |  |
| TriAttn | `trifast` | 3.45e-03 (GREEN) [pytorch 3.45e-03] | 1.01e-02 (YELLOW, grad:to_key.weight) [pytorch 9.04e-03] | new in this integration |
| TriAttn | `gnops` | 3.45e-03 (GREEN) [pytorch 3.45e-03] | 8.84e-03 (GREEN, grad:ln_pair.weight) [pytorch 9.90e-03] |  |
| TriAttn | `of3tri` | 3.45e-03 (GREEN) [pytorch 3.45e-03] | 8.84e-03 (GREEN, grad:ln_pair.weight) [pytorch 9.90e-03] |  |
| TriAttn | `megafold` | 3.45e-03 (GREEN) [pytorch 3.45e-03] | 9.44e-03 (GREEN, grad:ln_pair.weight) [pytorch 9.90e-03] |  |
| TriAttn | `bionemo_ir` | 3.45e-03 (GREEN) [pytorch 3.45e-03] | unsupported: NOBWD |  |
| TriAttn | `turbopf` | unsupported: HW | unsupported: HW |  |
| AttentionPairBias-384 | `pytorch` | 2.94e-03 (GREEN) | 1.49e-02 (GREEN, grad:pair) |  |
| AttentionPairBias-384 | `cuda` | 3.06e-03 (GREEN) [pytorch 2.94e-03] | 8.52e-03 (YELLOW, grad:to_key.weight) [pytorch 7.59e-03] |  |
| AttentionPairBias-384 | `gnops` | 3.06e-03 (GREEN) [pytorch 2.94e-03] | 8.48e-03 (YELLOW, grad:ln_single.weight) [pytorch 8.16e-03] |  |
| AttentionPairBias-384 | `of3tri` | 3.06e-03 (GREEN) [pytorch 2.94e-03] | 8.48e-03 (YELLOW, grad:ln_single.weight) [pytorch 8.16e-03] |  |
| AttentionPairBias-384 | `megafold` | 3.06e-03 (GREEN) [pytorch 2.94e-03] | 1.04e-02 (YELLOW, grad:ln_single.weight) [pytorch 8.16e-03] |  |
| AttentionPairBias-768 | `pytorch` | 2.81e-03 (GREEN) | 8.28e-03 (GREEN, grad:pair) |  |
| AttentionPairBias-768 | `cuda` | 2.81e-03 (GREEN) [pytorch 2.81e-03] | 7.87e-03 (GREEN, grad:to_query.bias) [pytorch 8.23e-03] |  |
| AttentionPairBias-768 | `gnops` | 2.81e-03 (GREEN) [pytorch 2.81e-03] | 1.09e-02 (YELLOW, grad:to_query.weight) [pytorch 9.90e-03] |  |
| AttentionPairBias-768 | `of3tri` | 2.81e-03 (GREEN) [pytorch 2.81e-03] | 1.09e-02 (YELLOW, grad:to_query.weight) [pytorch 9.90e-03] |  |
| AttentionPairBias-768 | `megafold` | 2.81e-03 (GREEN) [pytorch 2.81e-03] | 1.11e-02 (YELLOW, grad:to_query.weight) [pytorch 9.90e-03] |  |
| OPM | `pytorch` | 3.13e-03 (GREEN) | 6.99e-03 (GREEN, grad:msa) |  |
| OPM | `cuda` | 3.08e-03 (GREEN) [pytorch 3.13e-03] | 5.24e-03 (GREEN, grad:to_right.weight) [pytorch 5.36e-03] |  |
| OPM | `bionemo_ir` | 3.13e-03 (GREEN) [pytorch 3.13e-03] | unsupported: NOBWD |  |
| PWA | `pytorch` | 3.01e-03 (GREEN) | 1.53e-02 (GREEN, grad:pair) |  |
| PWA | `cuda` | 3.04e-03 (GREEN) [pytorch 3.01e-03] | 7.53e-03 (GREEN, grad:to_bias.weight) [pytorch 8.05e-03] |  |
| PWA | `bionemo_ir` | 7.07e-04 (GREEN) [pytorch 3.01e-03] | unsupported: NOBWD |  |
| PWA | `ligo_pwa` | 3.04e-03 (GREEN) [pytorch 3.01e-03] | 1.12e-02 (YELLOW, grad:to_bias.weight) [pytorch 8.05e-03] |  |
| Transition-d128 | `pytorch` | 6.88e-03 (GREEN) | 1.06e-02 (GREEN, grad:ln_in.bias) |  |
| Transition-d128 | `cuda` | 4.86e-03 (GREEN) [pytorch 6.88e-03] | 4.94e-03 (GREEN, grad:ln_in.weight) [pytorch 4.79e-03] |  |
| Transition-d128 | `bionemo_ir` | unsupported: SHAPE | unsupported: NOBWD |  |
| Transition-d128 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.88e-03] | 1.09e-02 (YELLOW, grad:x) [pytorch 7.93e-03] |  |
| Transition-d64 | `pytorch` | 6.09e-03 (GREEN) | 8.46e-03 (GREEN, grad:x) |  |
| Transition-d64 | `cuda` | 4.45e-03 (GREEN) [pytorch 6.09e-03] | 5.28e-03 (GREEN, grad:ln_in.bias) [pytorch 6.65e-03] |  |
| Transition-d64 | `bionemo_ir` | unsupported: SHAPE | unsupported: NOBWD |  |
| Transition-d64 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.09e-03] | 1.04e-02 (YELLOW, grad:x) [pytorch 8.46e-03] |  |

RTX 5090 (sm_120): part B (TriMul, L = 512, d_pair 128)

| backend | output rel rms | output rel max | update rel rms | dx rel rms | dWo rel rms |
|---|---|---|---|---|---|
| `pytorch` | 3.65e-03 | 7.10e-03 | 7.69e-03 | 5.47e-03 | 6.59e-03 |
| `cuda` | 3.20e-03 | 5.10e-03 | 6.75e-03 | 4.39e-03 | 5.04e-03 |
| `fast_trimul` | 1.98e-03 | 3.63e-03 | 4.18e-03 | 2.75e-03 | 1.90e-03 |
| `gpumode` | 2.31e-03 | 5.40e-03 | 4.86e-03 | - | - |
| `geonn` | 3.46e-03 | 5.56e-03 | 7.30e-03 | - | - |

### B200 (sm_100): part A (L = 512)

| class | backend | eval error (light) | train error (light, tensor) | note |
|---|---|---|---|---|
| TriAttn | `pytorch` | 3.86e-03 (GREEN) | 8.67e-03 (GREEN, grad:ln_pair.weight) |  |
| TriAttn | `cuda` | 3.03e-03 (GREEN) [pytorch 3.86e-03] | 9.06e-03 (GREEN, grad:to_query.weight) [pytorch 1.04e-02] |  |
| TriAttn | `cueq` | 3.67e-03 (GREEN) [pytorch 3.86e-03] | 7.93e-03 (GREEN, grad:ln_pair.weight) [pytorch 8.67e-03] |  |
| TriAttn | `ds4sci` | 3.67e-03 (GREEN) [pytorch 3.86e-03] | 7.19e-03 (GREEN, grad:pair) [pytorch 7.50e-03] | new in this integration |
| TriAttn | `trifast` | 3.67e-03 (GREEN) [pytorch 3.86e-03] | 7.99e-03 (GREEN, grad:ln_pair.weight) [pytorch 8.67e-03] | new in this integration |
| TriAttn | `gnops` | 3.67e-03 (GREEN) [pytorch 3.86e-03] | 9.32e-03 (YELLOW, grad:ln_pair.weight) [pytorch 8.67e-03] |  |
| TriAttn | `of3tri` | 3.67e-03 (GREEN) [pytorch 3.86e-03] | 9.33e-03 (YELLOW, grad:ln_pair.weight) [pytorch 8.67e-03] |  |
| TriAttn | `megafold` | 3.67e-03 (GREEN) [pytorch 3.86e-03] | 7.19e-03 (GREEN, grad:pair) [pytorch 7.50e-03] |  |
| TriAttn | `bionemo_ir` | 3.67e-03 (GREEN) [pytorch 3.86e-03] | unsupported: NOBWD |  |
| TriAttn | `turbopf` | unsupported: HW | unsupported: HW |  |
| TriMul | `pytorch` | 7.01e-03 (GREEN) | 8.60e-03 (GREEN, grad:pair) |  |
| TriMul | `cuda` | 5.10e-03 (GREEN) [pytorch 7.01e-03] | 6.25e-03 (GREEN, grad:to_right.weight) [pytorch 6.27e-03] |  |
| TriMul | `cueq` | 5.10e-03 (GREEN) [pytorch 7.01e-03] | 7.77e-03 (GREEN, grad:pair) [pytorch 8.60e-03] |  |
| TriMul | `fast_trimul` | unsupported: SHAPE | unsupported: SHAPE |  |
| TriMul | `gpumode` | 7.01e-03 (GREEN) [pytorch 7.01e-03] | unsupported: NOBWD |  |
| TriMul | `geonn` | 5.75e-03 (GREEN) [pytorch 7.01e-03] | unsupported: NOBWD |  |
| AttentionPairBias-384 | `pytorch` | 2.69e-03 (GREEN) | 1.21e-02 (GREEN, grad:pair) |  |
| AttentionPairBias-384 | `cuda` | 2.69e-03 (GREEN) [pytorch 2.69e-03] | 8.40e-03 (YELLOW, grad:ln_single.weight) [pytorch 7.07e-03] |  |
| AttentionPairBias-384 | `cueq` | 2.69e-03 (GREEN) [pytorch 2.69e-03] | 2.01e-02 (YELLOW, grad:pair) [pytorch 1.21e-02] |  |
| AttentionPairBias-384 | `gnops` | 2.69e-03 (GREEN) [pytorch 2.69e-03] | 8.88e-03 (YELLOW, grad:ln_single.weight) [pytorch 7.07e-03] |  |
| AttentionPairBias-384 | `of3tri` | 2.69e-03 (GREEN) [pytorch 2.69e-03] | 8.88e-03 (YELLOW, grad:ln_single.weight) [pytorch 7.07e-03] |  |
| AttentionPairBias-384 | `megafold` | 2.69e-03 (GREEN) [pytorch 2.69e-03] | 8.30e-03 (YELLOW, grad:ln_single.weight) [pytorch 7.07e-03] |  |
| AttentionPairBias-768 | `pytorch` | 2.93e-03 (GREEN) | 8.63e-03 (GREEN, grad:ln_single.weight) |  |
| AttentionPairBias-768 | `cuda` | unsupported: SHAPE | unsupported: SHAPE |  |
| AttentionPairBias-768 | `cueq` | 2.93e-03 (GREEN) [pytorch 2.93e-03] | 1.64e-02 (YELLOW, grad:to_query.bias) [pytorch 7.09e-03] |  |
| AttentionPairBias-768 | `gnops` | 2.93e-03 (GREEN) [pytorch 2.93e-03] | 9.96e-03 (YELLOW, grad:pair) [pytorch 7.82e-03] |  |
| AttentionPairBias-768 | `of3tri` | 2.93e-03 (GREEN) [pytorch 2.93e-03] | 9.96e-03 (YELLOW, grad:pair) [pytorch 7.82e-03] |  |
| AttentionPairBias-768 | `megafold` | 2.93e-03 (GREEN) [pytorch 2.93e-03] | 9.96e-03 (YELLOW, grad:pair) [pytorch 7.82e-03] |  |
| OPM | `pytorch` | 3.13e-03 (GREEN) | 7.38e-03 (GREEN, grad:msa) |  |
| OPM | `cuda` | 3.08e-03 (GREEN) [pytorch 3.13e-03] | 5.41e-03 (GREEN, grad:msa) [pytorch 7.38e-03] |  |
| OPM | `bionemo_ir` | 3.13e-03 (GREEN) [pytorch 3.13e-03] | unsupported: NOBWD |  |
| PWA | `pytorch` | 3.01e-03 (GREEN) | 1.14e-02 (GREEN, grad:pair) |  |
| PWA | `cuda` | 3.04e-03 (GREEN) [pytorch 3.01e-03] | 8.69e-03 (GREEN, grad:pair) [pytorch 1.14e-02] |  |
| PWA | `bionemo_ir` | 7.58e-04 (GREEN) [pytorch 3.01e-03] | unsupported: NOBWD |  |
| PWA | `ligo_pwa` | 2.98e-03 (GREEN) [pytorch 3.01e-03] | 8.11e-03 (YELLOW, grad:ln_pair.weight) [pytorch 7.11e-03] |  |
| Transition-d128 | `pytorch` | 6.88e-03 (GREEN) | 7.93e-03 (GREEN, grad:x) |  |
| Transition-d128 | `cuda` | 3.80e-03 (GREEN) [pytorch 6.88e-03] | 7.50e-03 (GREEN, grad:x) [pytorch 7.93e-03] |  |
| Transition-d128 | `bionemo_ir` | 3.80e-03 (GREEN) [pytorch 6.88e-03] | unsupported: NOBWD |  |
| Transition-d128 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.88e-03] | 1.09e-02 (YELLOW, grad:x) [pytorch 7.93e-03] |  |
| Transition-d64 | `pytorch` | 6.09e-03 (GREEN) | 8.75e-03 (GREEN, grad:x) |  |
| Transition-d64 | `cuda` | 4.46e-03 (GREEN) [pytorch 6.09e-03] | 6.38e-03 (GREEN, grad:x) [pytorch 8.75e-03] |  |
| Transition-d64 | `bionemo_ir` | 4.46e-03 (GREEN) [pytorch 6.09e-03] | unsupported: NOBWD |  |
| Transition-d64 | `megafold_tr` | 8.12e-03 (YELLOW) [pytorch 6.09e-03] | 1.07e-02 (YELLOW, grad:x) [pytorch 8.75e-03] |  |

B200 (sm_100): part B (TriMul, L = 512, d_pair 128)

| backend | output rel rms | output rel max | update rel rms | dx rel rms | dWo rel rms |
|---|---|---|---|---|---|
| `pytorch` | 3.65e-03 | 6.51e-03 | 7.69e-03 | 5.47e-03 | 6.69e-03 |
| `cuda` | 3.12e-03 | 5.41e-03 | 6.57e-03 | 4.47e-03 | 5.12e-03 |
| `fast_trimul` | 1.98e-03 | 3.82e-03 | 4.18e-03 | 2.75e-03 | 1.88e-03 |
| `gpumode` | 2.31e-03 | 6.26e-03 | 4.86e-03 | - | - |
| `geonn` | 3.46e-03 | 6.26e-03 | 7.30e-03 | - | - |

## 5. Candidates

| candidate (`backend`) | library | license | classes it serves | cards with a measured cell (ok / cells) | why not elsewhere |
|---|---|---|---|---|---|
| `ds4sci` | DeepSpeed4Science `DS4Sci_EvoformerAttention` (deepspeed 0.20.0) | Apache-2.0 | TriAttn | h100 0/6 (+6 eager-timing only, graph invalid); a100 0/6 (+6 eager-timing only, graph invalid); rtx5090 0/6; b200 0/6 (+6 eager-timing only, graph invalid) | RTX 5090: the CUTLASS extension does not JIT-load on sm_120 (`evoformer_attn op ... not compatible`); needs CUTLASS headers + nvcc; B200: loads, but the graph timing is invalid (eager timings used, section 0) |
| `trifast` | trifast 0.1.13 (Triton) | MIT | TriAttn | h100 6/6; a100 6/6; rtx5090 6/6; b200 6/6 | none recorded |
| `cueq` | NVIDIA cuEquivariance (engine's external baseline) | Apache-2.0 | TriAttn, TriMul, AttentionPairBias-384, AttentionPairBias-768 | h100 12/12; a100 12/12; rtx5090 12/12; b200 30/30 | TriAttn and TriMul only |
| `fast_trimul` | fast_trimul 3.0.6 (CuTe DSL, fp16) | Apache-2.0 | TriMul | h100 6/6; a100 6/6; rtx5090 6/6; b200 6/6 | fp16 kernels (bf16 cast), L % 8 == 0, no mask; training = kernel forward + torch-recompute backward |
| `gpumode` | GPU MODE TriMul competition kernels (arseniivanov/trimul, Triton) | MIT | TriMul | h100 3/6; a100 3/6; rtx5090 3/6; b200 3/6 | forward only (no backward): training cells unsupported; its output-gate placement is remapped by the adapter |
| `geonn` | AMD geometric-nn-ops `triangle_multiplicative_update` (Triton) | Apache-2.0 | TriMul | h100 3/6; a100 3/6; rtx5090 3/6; b200 3/6 | forward only (autograd path is its PyTorch reference): training cells unsupported |
| `gnops` | AMD geometric-nn-ops `triangle_attention` (Triton) | Apache-2.0 | TriAttn, AttentionPairBias-384, AttentionPairBias-768 | h100 24/24; a100 24/24; rtx5090 24/24; b200 24/24 | none recorded (TriAttn + AttentionPairBias; its `attention_pair_bias` op is pure torch and not used) |
| `of3tri` | OpenFold3 Triton Evoformer attention | Apache-2.0 | TriAttn, AttentionPairBias-384, AttentionPairBias-768 | h100 24/24; a100 24/24; rtx5090 24/24; b200 24/24 | none recorded |
| `megafold` | MegaFold EvoAttention (Triton) | MIT | TriAttn, AttentionPairBias-384, AttentionPairBias-768 | h100 24/24; a100 24/24; rtx5090 24/24; b200 24/24 | none recorded |
| `bionemo_ir` | NVIDIA BioNeMo Inference Runtime (CuTeDSL CUBINs) | Apache-2.0 | TriAttn, OPM, PWA, Transition-d128, Transition-d64 | h100 3/30; a100 3/30; rtx5090 9/30; b200 15/30 | H100: the host glibc 2.28 < 2.34 required by its CuTeDSL library; A100: CUDA_ERROR_INVALID_IMAGE (its CUBINs do not load on driver 570); B200: runs (inference only); inference only (no backward); Transition: no shipped silu_gate variant for width 128 / hidden 512 or 64 / 256 |
| `turbopf` | TurboPairFormer Hopper triangle attention (CUDA) | Apache-2.0 | TriAttn | h100 6/6; a100 0/6; rtx5090 0/6; b200 0/6 | Hopper (sm_90) only: A100, RTX 5090 and B200 refuse; TriAttn only; d <= 64 |
| `ligo_pwa` | Ligo AlphaFold3 Triton PWA kernel (patched) | Apache-2.0 | PWA | h100 6/6; a100 6/6; rtx5090 6/6; b200 6/6 | PWA only |
| `megafold_tr` | MegaFold fused Transition (Triton) | MIT | Transition-d128, Transition-d64 | h100 12/12; a100 12/12; rtx5090 12/12; b200 12/12 | Transition only (bf16) |
| FlashBias | low-rank-bias attention | MIT | none | not connected | NOT APPLICABLE: the pair bias of AF3 attention is dense (not low-rank) |

Checked and not connected (documented in README_trimul.md / ext_attn_results.md): Hugging Face `trimul_gpumode` `kernel_global` (derived from a repository without a license), the TTT-Discover H100 TriMul kernel (printed in a paper appendix only), Anthropic kits (E1 / E2 baseline, outside E5).

